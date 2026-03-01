from flask import (
    Blueprint,
    render_template,
    request,
    jsonify,
    redirect,
    url_for,
    flash,
    session,
    current_app,
)
from sqlalchemy import func
from flask_login import login_user, login_required, logout_user, current_user
from models import db, User, Role, FAQ, File, Query, Settings, NewQuestion
from urllib.parse import urlparse,parse_qs

from werkzeug.security import generate_password_hash
from datetime import datetime
import os
from embed_and_search import (
    generate_and_store_embeddings,
    search_across_indices,
    split_into_chunks,
    create_embedding,
)
from openai import OpenAI
import json
import numpy as np
import uuid
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse
import re
from pathlib import Path
from werkzeug.utils import secure_filename
import time
from zotero_client import (
    get_libraries,
    get_items_with_attachments,
    download_attachment_file,
    file_like_for_zotero,
    ALLOWED_EXTENSIONS_LABEL,
)
try:
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.common.exceptions import TimeoutException, WebDriverException
    SELENIUM_AVAILABLE = True
except ImportError:
    SELENIUM_AVAILABLE = False

# Create blueprint
main = Blueprint("main", __name__)


# Public routes
@main.route("/")
def index():
    faqs = FAQ.query.all()
    settings = {s.key: s.value for s in Settings.query.all()}
    return render_template("public/index.html", faqs=faqs, settings=settings)


@main.route("/about")
def about():
    settings = {s.key: s.value for s in Settings.query.all()}
    return render_template("public/about.html", settings=settings)


@main.route("/contact")
def contact():
    settings = {s.key: s.value for s in Settings.query.all()}
    return render_template("public/contact.html", settings=settings)


@main.route("/api/chat", methods=["POST"])
def chat():
    data = request.json
    question = data.get("question")
    language = data.get("language", "en")  # default to English
    history = data.get("history", [])
    # check if history is a list of dicts with 'role' and 'content' and not empty
    if isinstance(history, list) and history and all(
        isinstance(h, dict) and "role" in h and "content" in h for h in history
    ):
        history_text = "\n".join(f"{h['role']}: {h['content']}" for h in history)
    else:
        history_text = ""

    if not question:
        return jsonify({"error": "No question provided", "type": "error"}), 400

    try:

        def save_query(answer_found=True, answer=None):
            query = Query(
                question=question,
                answer=answer,
                answer_found=answer_found,
                happy=True,
                language=language,
            )
            db.session.add(query)
            db.session.commit()
            return query

        # 1. Check if exact match exists in FAQ
        faq = FAQ.query.filter(func.lower(FAQ.question) == question.lower()).first()
        if faq:
            query = save_query(answer=faq.answer)
            return jsonify(
                {
                    "answer": faq.answer,
                    "type": "success",
                    "query_id": query.id,
                    "faq_id": faq.id,
                }
            )

        # 2. Ensure OpenAI key is set
        openai_key = Settings.query.filter_by(key="openai_key").first()
        openai_model = Settings.query.filter_by(key="openai_model").first()
        show_matched_text = Settings.query.filter_by(key="show_matched_text").first()
        show_matched_text = (
            show_matched_text.value.lower()
            if show_matched_text and show_matched_text.value
            else "yes"
        )

        if (
            not openai_key
            or not openai_key.value
            or not openai_model
            or not openai_model.value
        ):
            return (
                jsonify(
                    {
                        "answer": "This project is not configured properly. Please contact the responsible person.",
                        "type": "error",
                    }
                ),
                400,
            )

        # 3. Search across file embeddings
        files = File.query.all()
        file_ids = [f.file_identifier for f in files if f.file_identifier]
        file_id_to_title = {
            f.file_identifier: f.original_filename for f in files if f.file_identifier
        }

        if not file_ids:
            return handle_no_content(question, language)
        # get chunk number from settings
        chunk_setting = Settings.query.filter_by(key="chunk_number").first()
        chunk_number = chunk_setting.value if chunk_setting and chunk_setting.value else 3
        try:
            chunk_number = int(chunk_number)
            if chunk_number < 1 or chunk_number > 5:
                chunk_number = 3
        except (TypeError, ValueError):
            chunk_number = 3
        results = search_across_indices(question, file_ids, top_k=chunk_number)
        if not results and not history_text:
            # Save query with answer_found=False and return no content message
            message = {
                "fi": "Valitettavasti en löytänyt vastausta kysymykseesi. Olen tallentanut sen ja tiimimme tarkistaa sen.",
                "en": "I'm sorry, I couldn't find a specific answer to your question. I've noted it down and our team will review it.",
            }.get(language, "en")

            query = save_query(answer_found=False, answer=message)

            # Create new question for review
            new_question = NewQuestion(question=question)
            db.session.add(new_question)
            db.session.commit()

            return jsonify({"answer": message, "type": "info", "query_id": query.id})

        # 4. Build context from results
        context = "\n\n".join(
            [
                f"Relevant content from {file_id_to_title[r['file_id']]}:\n{r['chunk']}"
                for r in results
                if r["file_id"] in file_id_to_title
            ]
        )
        used_files = [
            {
                "name": file_id_to_title[r["file_id"]],
                "chunk": r.get("chunk", ""),  # adjust to your actual field name
            }
            for r in results
            if r["file_id"] in file_id_to_title
        ]

        # 5. Generate answer via OpenAI
        system_prompt = {
            "en": "You are a helpful assistant. Use the following context to answer the question in English. If the context doesn't contain enough information to answer the question, say so. And for casual greetings and chats, reply and ask how can i assist you.",
            "fi": "Olet avulias assistentti. Käytä seuraavaa kontekstia vastataksesi kysymykseen suomeksi. Jos kontekstissa ei ole tarpeeksi tietoa vastataksesi kysymykseen, kerro niin:",
        }
        prompt_content = f"""Conversation so far:{history_text} User's latest question:{question}Retrieved context:{context}"""
        client = OpenAI(api_key=openai_key.value)
        response = client.chat.completions.create(
            model=(
                openai_model.value if openai_model and openai_model.value else "gpt-4"
            ),
            messages=[
                {
                    "role": "system",
                    "content": system_prompt.get(language, system_prompt["en"]),
                },
                {
                    "role": "user",
                    # "content": f"Context: {context}\n\nQuestion: {question}",
                    "content": prompt_content,
                },
            ],
            temperature=0.8,
            max_tokens=600,
        )

        answer = response.choices[0].message.content
        used_file_names = []  # For PDF generation - just clean file names
        if used_files:
            files_list = []
            for f in used_files:
                fname = f["name"]  # file name / URL string
                chunk_text = f["chunk"].replace('"', "&quot;")  # escape quotes for HTML

                if isinstance(fname, str) and fname.startswith("http"):
                    parsed = urlparse(fname)
                    display = parsed.netloc
                    if parsed.path and parsed.path != "/":
                        parts = parsed.path.strip("/").split("/")
                        display += f"/{parts[0]}"
                    html = f"- <a href='{fname}' target='_blank'>{display}</a>"
                    used_file_names.append(fname)
                    if show_matched_text == "yes":
                        html += (
                            f"<details><summary style='cursor:pointer; color:gray;'>View Original Text</summary>"
                            f"<pre style='white-space: pre-wrap; margin:0; color:#247952 !important' >{chunk_text}</pre></details>"
                        )
                    files_list.append(html)
                else:
                    html = f"- {fname}"
                    used_file_names.append(fname)
                    if show_matched_text == "yes":
                        html += (
                            f"<details><summary style='cursor:pointer; color:gray;'>View Original Text</summary>"
                            f"<pre style='white-space: pre-wrap; margin:0;color:#247952 !important'>{chunk_text}</pre></details>"
                        )
                    files_list.append(html)

            files_html = "<br>".join(files_list)
            sources_used = f"<br><br><b>Sources used:</b><br>{files_html}"

        # 6. Save query with the generated answer
        query = save_query(answer=answer)
        # PRINT clean_file_names
        return jsonify({"answer": answer,"sources_used":sources_used, "used_file_names":used_file_names, "type": "success", "query_id": query.id})

    except Exception as e:
        print(f"Error in chat route: {str(e)}")
        return jsonify({"error": str(e), "type": "error"}), 500


def handle_no_content(question, language):
    new_question = NewQuestion(question=question)
    db.session.add(new_question)
    db.session.commit()

    message = {
        "fi": "Valitettavasti en löytänyt vastausta kysymykseesi. Olen tallentanut sen ja tiimimme tarkistaa sen.",
        "en": "I'm sorry, I couldn't find a specific answer to your question. I've noted it down and our team will review it.",
    }.get(language, language["en"])

    return jsonify({"answer": message, "type": "info"})


# Admin routes
@main.route("/admin/login", methods=["GET", "POST"])
def admin_login():
    if request.method == "POST":
        email = request.form.get("email")
        password = request.form.get("password")
        user = User.query.filter_by(email=email).first()

        if user and user.check_password(password):
            login_user(user)
            flash("Welcome back!", "success")
            return redirect(url_for("main.admin_dashboard"))
        flash("Invalid email or password", "error")
    return render_template("admin/login.html")


@main.route("/admin/logout")
@login_required
def admin_logout():
    logout_user()
    flash("You have been logged out successfully", "success")
    return redirect(url_for("main.admin_login"))


@main.route("/admin")
@login_required
def admin_dashboard():
    # Regenerate any missing embeddings
    regenerate_missing_embeddings()

    users = User.query.all()
    faqs = FAQ.query.all()
    files = File.query.all()
    new_questions = NewQuestion.query.order_by(NewQuestion.created_at.desc()).all()
    settings = {s.key: s.value for s in Settings.query.all()}
    return render_template(
        "admin/dashboard.html",
        users=users,
        faqs=faqs,
        files=files,
        new_questions=new_questions,
        settings=settings,
    )


# CRUD routes for admin
@main.route("/admin/profile", methods=["GET", "POST"])
@login_required
def admin_profile():
    if request.method == "POST":
        try:
            current_user.name = request.form.get("name")
            current_user.phone = request.form.get("phone")
            if request.form.get("password"):
                current_user.set_password(request.form.get("password"))
            db.session.commit()
            flash("Profile updated successfully", "success")
        except Exception as e:
            flash(f"Error updating profile: {str(e)}", "error")
    return render_template("admin/profile.html")


@main.route("/admin/roles")
@login_required
def admin_roles():
    roles = Role.query.all()
    settings = {s.key: s.value for s in Settings.query.all()}
    return render_template("admin/roles.html", roles=roles, settings=settings)


@main.route("/admin/users")
@login_required
def admin_users():
    users = User.query.all()
    roles = Role.query.all()
    settings = {s.key: s.value for s in Settings.query.all()}
    return render_template(
        "admin/users.html", users=users, roles=roles, settings=settings
    )


@main.route("/admin/faqs")
@login_required
def admin_faqs():
    faqs = FAQ.query.all()
    settings = {s.key: s.value for s in Settings.query.all()}
    return render_template("admin/faqs.html", faqs=faqs, settings=settings)


@main.route("/admin/files")
@login_required
def admin_files():
    settings = {s.key: s.value for s in Settings.query.all()}
    return render_template("admin/files.html", files=[], settings=settings)


@main.route("/admin/queries")
@login_required
def admin_queries():
    settings = {s.key: s.value for s in Settings.query.all()}
    return render_template("admin/queries.html", queries=[], settings=settings)


@main.route("/admin/zotero")
@login_required
def admin_zotero():
    settings = {s.key: s.value for s in Settings.query.all()}
    last_result = settings.get("zotero_last_sync_result")
    last_sync_summary = None
    if last_result:
        try:
            d = json.loads(last_result)
            last_sync_summary = f"Synced: {d.get('synced', 0)}, Skipped: {d.get('skipped', 0)}, Errors: {d.get('errors', 0)}"
        except Exception:
            pass
    return render_template(
        "admin/zotero.html",
        settings=settings,
        allowed_extensions_label=ALLOWED_EXTENSIONS_LABEL,
        last_sync_summary=last_sync_summary,
    )


@main.route("/api/zotero/libraries", methods=["GET"])
@login_required
def api_zotero_libraries():
    """Return list of Zotero libraries (My Library + groups) for the configured user."""
    try:
        api_key = Settings.query.filter_by(key="zotero_api_key").first()
        user_id = Settings.query.filter_by(key="zotero_user_id").first()
        api_key = api_key.value if api_key and api_key.value else None
        user_id = user_id.value if user_id and user_id.value else None
        if not api_key or not user_id:
            return jsonify({
                "libraries": [],
                "connected": False,
                "message": "Configure Zotero API key and User ID in Settings.",
            })
        libraries = get_libraries(user_id.strip(), api_key.strip())
        return jsonify({"libraries": libraries, "connected": True})
    except Exception as e:
        print(f"Zotero libraries error: {e}")
        return jsonify({
            "libraries": [],
            "connected": False,
            "message": str(e),
        }), 500


@main.route("/api/zotero/sync", methods=["POST"])
@login_required
def api_zotero_sync():
    """Sync allowed file types from My Library and all groups. Skips unsupported types."""
    try:
        api_key_setting = Settings.query.filter_by(key="zotero_api_key").first()
        user_id_setting = Settings.query.filter_by(key="zotero_user_id").first()
        api_key = api_key_setting.value if api_key_setting and api_key_setting.value else None
        user_id = user_id_setting.value if user_id_setting and user_id_setting.value else None
        if not api_key or not user_id:
            return jsonify({
                "type": "error",
                "message": "Configure Zotero API key and User ID in Settings first.",
            }), 400

        api_key = api_key.strip()
        user_id = user_id.strip()
        synced = 0
        skipped = 0
        errors = 0

        for lib in get_libraries(user_id, api_key):
            lib_type = lib["type"]
            lib_id = lib["id"]
            for att in get_items_with_attachments(lib_type, lib_id, api_key):
                item_key = att["item_key"]
                filename = att["filename"]
                external_id = f"zotero:{lib_type}/{lib_id}/{item_key}"
                if File.query.filter_by(external_id=external_id).first():
                    skipped += 1
                    continue
                data, down_filename = download_attachment_file(
                    lib_type, lib_id, item_key, api_key
                )
                if not data:
                    errors += 1
                    continue
                name = down_filename or filename
                file_like = file_like_for_zotero(data, name)
                text = extract_text_from_file(file_like)
                if not text:
                    skipped += 1
                    continue
                try:
                    file_identifier = generate_and_store_embeddings(text)
                    if not file_identifier:
                        errors += 1
                        continue
                    file_like.seek(0)
                    display_name = generate_filename(file_like, text)
                    db_file = File(
                        text=text,
                        user_id=current_user.id,
                        file_identifier=file_identifier,
                        original_filename=display_name,
                        source="zotero",
                        external_id=external_id,
                    )
                    db.session.add(db_file)
                    synced += 1
                except Exception as e:
                    print(f"Zotero sync error for {name}: {e}")
                    errors += 1
                    continue

        db.session.commit()

        # Update last sync time and result
        from datetime import datetime as dt
        now_str = dt.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
        result_json = json.dumps({"synced": synced, "skipped": skipped, "errors": errors})
        for key, value in [
            ("zotero_last_sync", now_str),
            ("zotero_last_sync_result", result_json),
        ]:
            setting = Settings.query.filter_by(key=key).first()
            if setting:
                setting.value = value
            else:
                db.session.add(Settings(key=key, value=value))
        db.session.commit()

        message = f"Synced {synced} file(s). Skipped: {skipped}. Errors: {errors}."
        return jsonify({
            "type": "success",
            "message": message,
            "synced": synced,
            "skipped": skipped,
            "errors": errors,
        })
    except Exception as e:
        print(f"Zotero sync error: {e}")
        db.session.rollback()
        return jsonify({"type": "error", "message": str(e)}), 500


def _apply_settings_update(settings_to_update, preserve_empty_keys=None):
    """Apply a dict of key->value to Settings table. Keys in preserve_empty_keys are skipped when value is empty."""
    preserve_empty_keys = preserve_empty_keys or []
    for key, value in settings_to_update.items():
        if value is None:
            continue
        if key in preserve_empty_keys and (value is None or str(value).strip() == ""):
            continue
        setting = Settings.query.filter_by(key=key).first()
        if setting:
            setting.value = value
        else:
            setting = Settings(key=key, value=value)
            db.session.add(setting)


@main.route("/admin/settings", methods=["GET", "POST"])
@login_required
def admin_settings():
    if request.method == "POST":
        try:
            section = request.form.get("settings_section")
            if not section or section not in ("site", "theme", "api", "pages"):
                return jsonify({"message": "Invalid settings section", "type": "error"}), 400

            upload_dir = os.path.join(current_app.static_folder, "uploads")
            os.makedirs(upload_dir, exist_ok=True)

            if section == "site":
                chunk_raw = request.form.get("chunk_number")
                msg_raw = request.form.get("message_history")
                try:
                    chunk_number = int(chunk_raw) if chunk_raw not in (None, "") else None
                    if chunk_number is not None and (chunk_number < 1 or chunk_number > 5):
                        chunk_number = None
                except (TypeError, ValueError):
                    chunk_number = None
                try:
                    message_history = int(msg_raw) if msg_raw not in (None, "") else None
                    if message_history is not None and (message_history < 1 or message_history > 15):
                        message_history = None
                except (TypeError, ValueError):
                    message_history = None
                settings_to_update = {
                    "logo": request.form.get("logo"),
                    "copyright": request.form.get("copyright"),
                    "show_matched_text": request.form.get("show_matched_text", "yes"),
                    "faq_search_enabled": request.form.get("faq_search_enabled", "yes"),
                }
                if chunk_number is not None:
                    settings_to_update["chunk_number"] = str(chunk_number)
                if message_history is not None:
                    settings_to_update["message_history"] = str(message_history)
                logo_file = request.files.get("logo_file")
                if logo_file and logo_file.filename:
                    old_logo = Settings.query.filter_by(key="logo_file").first()
                    if old_logo and old_logo.value:
                        old_logo_path = os.path.join(upload_dir, old_logo.value)
                        if os.path.exists(old_logo_path):
                            os.remove(old_logo_path)
                    filename = secure_filename(logo_file.filename)
                    logo_file.save(os.path.join(upload_dir, filename))
                    settings_to_update["logo_file"] = filename
                favicon_file = request.files.get("favicon_file")
                if favicon_file and favicon_file.filename:
                    old_favicon = Settings.query.filter_by(key="favicon_file").first()
                    if old_favicon and old_favicon.value:
                        old_favicon_path = os.path.join(upload_dir, old_favicon.value)
                        if os.path.exists(old_favicon_path):
                            os.remove(old_favicon_path)
                    filename = secure_filename(favicon_file.filename)
                    favicon_file.save(os.path.join(upload_dir, filename))
                    settings_to_update["favicon_file"] = filename
                _apply_settings_update(settings_to_update)

            elif section == "theme":
                settings_to_update = {
                    "theme_primary_color": request.form.get("theme_primary_color", "#5b1fa6"),
                    "theme_secondary_color": request.form.get("theme_secondary_color", "#d1aff3"),
                    "theme_bot_message_color": request.form.get("theme_bot_message_color", "#f3f0fa"),
                    "theme_background_start": request.form.get("theme_background_start", "#f5f0ff"),
                    "theme_background_end": request.form.get("theme_background_end", "#ede3f7"),
                    "theme_accent_color": request.form.get("theme_accent_color", "#5b1fa6"),
                }
                _apply_settings_update(settings_to_update)

            elif section == "api":
                openai_model = request.form.get("openai_model")
                settings_to_update = {
                    "openai_key": request.form.get("openai_key"),
                    "openai_model": openai_model if (openai_model and str(openai_model).strip()) else None,
                    "zotero_api_key": request.form.get("zotero_api_key"),
                    "zotero_user_id": request.form.get("zotero_user_id"),
                }
                _apply_settings_update(settings_to_update, preserve_empty_keys=["openai_key", "zotero_api_key"])

            elif section == "pages":
                about_val = request.form.get("about")
                contact_val = request.form.get("contact")
                # Avoid saving literal "undefined" from frontend if editors not ready
                if about_val == "undefined":
                    about_val = None
                if contact_val == "undefined":
                    contact_val = None
                settings_to_update = {
                    "about": about_val,
                    "contact": contact_val,
                }
                _apply_settings_update(settings_to_update)

            db.session.commit()
            return jsonify({"message": "Settings updated successfully", "type": "success"})
        except Exception as e:
            db.session.rollback()
            return jsonify({"message": str(e), "type": "error"}), 500

    settings = {s.key: s.value for s in Settings.query.all()}
    models = []

    if "openai_key" in settings and settings["openai_key"]:
        try:
            client = OpenAI(api_key=settings["openai_key"])
            resp = client.models.list()

            models = [
                m.id
                for m in resp.data
                if (
                    (m.id.startswith("gpt-") or m.id.startswith("o1-"))
                    and "embedding" not in m.id
                    and "audio" not in m.id
                    and ":" not in m.id
                )
            ]

            models = sorted(set(models), reverse=True)

            if not models:
                models = ["gpt-4"]
        except Exception as e:
            print("⚠️ Error fetching models:", e)
            models = ["gpt-4"]
    else:
        models = ["gpt-4"]

    # Do not pass API key values to template (avoid exposure in HTML/inspect)
    openai_key_set = bool(settings.get("openai_key"))
    zotero_api_key_set = bool(settings.get("zotero_api_key"))
    return render_template(
        "admin/settings.html",
        settings=settings,
        models=models,
        openai_key_set=openai_key_set,
        zotero_api_key_set=zotero_api_key_set,
    )


@main.route("/admin/new-questions")
@login_required
def admin_new_questions():
    new_questions = NewQuestion.query.order_by(NewQuestion.created_at.desc()).all()
    settings = {s.key: s.value for s in Settings.query.all()}
    return render_template(
        "admin/new_questions.html", new_questions=new_questions, settings=settings
    )


@main.route("/api/new-question/<int:id>/answer", methods=["POST"])
@login_required
def api_answer_new_question(id):
    try:
        new_question = NewQuestion.query.get_or_404(id)
        answer = request.form.get("answer")

        if not answer:
            return jsonify({"message": "Answer is required", "type": "error"}), 400

        # Create new FAQ
        faq = FAQ(question=new_question.question, answer=answer)
        db.session.add(faq)

        # Delete the new question
        db.session.delete(new_question)
        db.session.commit()

        return jsonify(
            {"message": "Question answered and added to FAQs", "type": "success"}
        )
    except Exception as e:
        print(f"Error in api_answer_new_question: {str(e)}")
        return jsonify({"message": str(e), "type": "error"}), 500


# API routes for CRUD operations
@main.route("/api/role", methods=["POST"])
@login_required
def api_create_role():
    try:
        name = request.form.get("name")
        if not name:
            return jsonify({"message": "Name is required", "type": "error"}), 400

        role = Role(name=name)
        db.session.add(role)
        db.session.commit()
        return jsonify({"message": "Role created successfully", "type": "success"})
    except Exception as e:
        return jsonify({"message": str(e), "type": "error"}), 500


@main.route("/api/role/<int:id>", methods=["PUT", "DELETE"])
@login_required
def api_manage_role(id):
    try:
        role = Role.query.get_or_404(id)

        if request.method == "DELETE":
            db.session.delete(role)
            db.session.commit()
            return jsonify({"message": "Role deleted successfully", "type": "success"})

        name = request.form.get("name")
        if not name:
            return jsonify({"message": "Name is required", "type": "error"}), 400

        role.name = name
        db.session.commit()
        return jsonify({"message": "Role updated successfully", "type": "success"})
    except Exception as e:
        return jsonify({"message": str(e), "type": "error"}), 500


@main.route("/api/user", methods=["POST"])
@login_required
def api_create_user():
    try:
        name = request.form.get("name")
        email = request.form.get("email")
        password = request.form.get("password")
        phone = request.form.get("phone")
        role_id = request.form.get("role_id")

        if not all([name, email, password]):
            return (
                jsonify(
                    {
                        "message": "Name, email, and password are required",
                        "type": "error",
                    }
                ),
                400,
            )

        user = User(
            name=name, email=email, phone=phone, role_id=role_id if role_id else None
        )
        user.set_password(password)

        db.session.add(user)
        db.session.commit()
        return jsonify({"message": "User created successfully", "type": "success"})
    except Exception as e:
        return jsonify({"message": str(e), "type": "error"}), 500


@main.route("/api/user/<int:id>", methods=["PUT", "DELETE"])
@login_required
def api_manage_user(id):
    try:
        user = User.query.get_or_404(id)

        if request.method == "DELETE":
            db.session.delete(user)
            db.session.commit()
            return jsonify({"message": "User deleted successfully", "type": "success"})

        user.name = request.form.get("name")
        user.email = request.form.get("email")
        user.phone = request.form.get("phone")
        user.role_id = request.form.get("role_id")

        if request.form.get("password"):
            user.set_password(request.form.get("password"))

        db.session.commit()
        return jsonify({"message": "User updated successfully", "type": "success"})
    except Exception as e:
        return jsonify({"message": str(e), "type": "error"}), 500


@main.route("/api/faq", methods=["POST"])
@login_required
def api_create_faq():
    try:
        if request.is_json:
            # Handle JSON request (from query conversion)
            data = request.get_json()
            if not data or "question" not in data or "answer" not in data:
                return (
                    jsonify(
                        {"type": "error", "message": "Question and answer are required"}
                    ),
                    400,
                )

            # Create new FAQ
            faq = FAQ(question=data["question"], answer=data["answer"])
            db.session.add(faq)

            # If query_id is provided, delete the query
            if "query_id" in data:
                query = Query.query.get(data["query_id"])
                if query:
                    db.session.delete(query)

            db.session.commit()
            return jsonify({"type": "success", "message": "FAQ created successfully"})
        else:
            # Handle form data request (from FAQ page)
            question = request.form.get("question")
            answer = request.form.get("answer")

            if not all([question, answer]):
                return (
                    jsonify(
                        {"message": "Question and answer are required", "type": "error"}
                    ),
                    400,
                )

            # Create FAQ record
            faq = FAQ(question=question, answer=answer)
            db.session.add(faq)
            db.session.commit()
            return jsonify({"message": "FAQ created successfully", "type": "success"})

    except Exception as e:
        db.session.rollback()
        print(f"Error creating FAQ: {str(e)}")
        return jsonify({"message": str(e), "type": "error"}), 500


def is_url(text: str) -> bool:
    try:
        parsed = urlparse(text)
        return all([parsed.scheme, parsed.netloc])
    except:
        return False


def extract_text_from_file(file_or_url):
    """Extract text from uploaded file, raw text, or URL based on its type."""
    try:
        # Handle string input: URL or raw text
        if isinstance(file_or_url, str):
            if is_url(file_or_url):
                parsed_url = urlparse(file_or_url)
                netloc = parsed_url.netloc.replace("www.", "").lower()
                if netloc in ["youtube.com", "youtu.be"]:
                    raise ValueError("YouTube URLs are not supported. Please use other types of URLs like web pages, PDFs, or documents.")
                return extract_webpage_text(file_or_url)
            else:
                return file_or_url.strip()  # raw text (e.g., from textarea)

        # Handle file uploads
        filename = file_or_url.filename.lower()

        if filename.endswith(".txt"):
            return file_or_url.read().decode("utf-8")

        elif filename.endswith(".pdf"):
            from PyPDF2 import PdfReader

            reader = PdfReader(file_or_url)
            return "\n".join(page.extract_text() or "" for page in reader.pages)

        elif filename.endswith(".docx"):
            from docx import Document

            doc = Document(file_or_url)
            return "\n".join(paragraph.text for paragraph in doc.paragraphs)

        else:
            return None  # unsupported file type

    except Exception as e:
        print(f"Error extracting text: {str(e)}")
        return None

def generate_filename(source, content=None):
    """Generate a unique filename based on the source type and content."""
    try:
        # Get current time for uniqueness
        current_time = datetime.now().strftime("%H%M")

        if isinstance(source, str):  # URL or raw text
            if source.startswith(("http://", "https://")):
                # # For URLs, use the URL itself as the base name
                # parsed_url = urlparse(source)
                # # Remove protocol and www
                # domain = parsed_url.netloc.replace("www.", "")
                # # Get path and remove leading/trailing slashes
                # path = parsed_url.path.strip("/")
                # # Combine domain and path, replace slashes with underscores
                # url_name = f"{domain}_{path}".replace("/", "_")
                # # Make it safe for filenames
                # safe_name = re.sub(r"[^\w\s-]", "", url_name)
                # safe_name = re.sub(r"[-\s]+", "_", safe_name)
                # return f"url_{safe_name}_{current_time}"
                return source
            else:
                # For raw text, use first line or generate random name
                if content:
                    first_line = content.split("\n")[0][:50].strip()
                    if first_line:
                        # Clean the first line to make it filename-safe
                        safe_name = re.sub(r"[^\w\s-]", "", first_line)
                        safe_name = re.sub(r"[-\s]+", "_", safe_name)
                        return f"text_{safe_name}_{current_time}"
                return f"text_{current_time}"
        else:  # File upload
            # Keep original filename but add timestamp
            original_name = os.path.splitext(source.filename)[0]
            extension = os.path.splitext(source.filename)[1]
            safe_name = re.sub(r"[^\w\s-]", "", original_name)
            safe_name = re.sub(r"[-\s]+", "_", safe_name)
            return f"{safe_name}_{current_time}{extension}"
    except Exception as e:
        print(f"Error generating filename: {str(e)}")
        return f"file_{current_time}"


def check_duplicate_filenames(files):
    """Check if any of the files already exist and return list of duplicate filenames."""
    duplicate_files = []
    for file in files:
        original_name = os.path.splitext(file.filename)[0]
        extension = os.path.splitext(file.filename)[1]
        safe_name = re.sub(r"[^\w\s-]", "", original_name)
        safe_name = re.sub(r"[-\s]+", "_", safe_name)
        base_filename = f"{safe_name}_"

        # Check if any existing file starts with this base filename
        existing = File.query.filter(
            File.original_filename.like(f"{base_filename}%")
        ).first()
        if existing:
            duplicate_files.append(
                {"original": file.filename, "existing": existing.original_filename}
            )

    return duplicate_files


def check_duplicate_url(url):
    """Check if a URL has already been processed."""
    try:
        # Normalize the URL by removing protocol and www
        parsed_url = urlparse(url)
        normalized_url = parsed_url.netloc.replace("www.", "") + parsed_url.path.rstrip(
            "/"
        )

        # Get all files that were created from URLs
        url_files = File.query.filter(File.original_filename.like("url_%")).all()

        for file in url_files:
            # Extract the URL part from the filename
            filename_parts = file.original_filename.split("_")
            if len(filename_parts) >= 3:  # url_domain_path_timestamp
                stored_url = "_".join(
                    filename_parts[1:-1]
                )  # Remove 'url_' prefix and timestamp
                if stored_url == normalized_url:
                    return True, file.original_filename

        return False, None
    except Exception as e:
        print(f"Error checking duplicate URL: {str(e)}")
        return False, None


@main.route("/api/file", methods=["POST"])
@login_required
def api_upload_file():
    try:
        upload_type = request.form.get("uploadType")
        # print("Upload Type:", upload_type)
        # print("Request form data:", request.form)
        # print("Request files:", request.files)

        if upload_type == "file":
            files = request.files.getlist("file")
            if not files or files[0].filename == "":
                return jsonify({"message": "No files selected", "type": "error"}), 400

            # Check for duplicate filenames first
            duplicate_files = check_duplicate_filenames(files)
            if duplicate_files:
                # Create a detailed error message listing all duplicates
                error_message = "The following files already exist:\n"
                for dup in duplicate_files:
                    error_message += (
                        f"- {dup['original']} (exists as: {dup['existing']})\n"
                    )
                return jsonify({"message": error_message, "type": "error"}), 400

            success_count = 0
            error_messages = []

            for file in files:
                try:
                    text = extract_text_from_file(file)
                    if not text:
                        error_messages.append(
                            f"Could not extract text from {file.filename}"
                        )
                        continue

                    file_identifier = generate_and_store_embeddings(text)
                    if not file_identifier:
                        error_messages.append(
                            f"Failed to generate embeddings for {file.filename}"
                        )
                        continue

                    filename = generate_filename(file, text)

                    db_file = File(
                        text=text,
                        user_id=current_user.id,
                        file_identifier=file_identifier,
                        original_filename=filename,
                    )
                    db.session.add(db_file)
                    success_count += 1
                except Exception as e:
                    error_messages.append(f"Error processing {file.filename}: {str(e)}")
                    continue

            if success_count > 0:
                db.session.commit()
                message = f"Successfully processed {success_count} file(s)"
                if error_messages:
                    message += f". Failed to process {len(error_messages)} file(s): {', '.join(error_messages)}"
                return jsonify({"message": message, "type": "success"})
            else:
                return (
                    jsonify(
                        {
                            "message": "Failed to process any files: "
                            + ", ".join(error_messages),
                            "type": "error",
                        }
                    ),
                    400,
                )

        elif upload_type == "url":
            url = request.form.get("url", "").strip()
            if not url:
                return jsonify({"message": "No URL provided", "type": "error"}), 400

            # Check for duplicate URL
            is_duplicate, existing_file = check_duplicate_url(url)
            if is_duplicate:
                return (
                    jsonify(
                        {
                            "message": f"This URL has already been processed (stored as: {existing_file})",
                            "type": "error",
                        }
                    ),
                    400,
                )

            # print(f"Processing URL: {url}")
            text = extract_text_from_file(url)
            if not text:
                return (
                    jsonify(
                        {"message": "Could not extract text from URL", "type": "error"}
                    ),
                    400,
                )

            file_identifier = generate_and_store_embeddings(text)
            if not file_identifier:
                return (
                    jsonify(
                        {"message": "Failed to generate embeddings", "type": "error"}
                    ),
                    500,
                )

            filename = generate_filename(url, text)

            db_file = File(
                text=text,
                user_id=current_user.id,
                file_identifier=file_identifier,
                original_filename=filename,
            )
            db.session.add(db_file)
            db.session.commit()

            return jsonify(
                {"message": "URL content processed successfully", "type": "success"}
            )

        elif upload_type == "text":
            text = request.form.get("text", "").strip()
            if not text:
                return jsonify({"message": "No text provided", "type": "error"}), 400

            file_identifier = generate_and_store_embeddings(text)
            if not file_identifier:
                return (
                    jsonify(
                        {"message": "Failed to generate embeddings", "type": "error"}
                    ),
                    500,
                )

            filename = generate_filename(text, text)

            db_file = File(
                text=text,
                user_id=current_user.id,
                file_identifier=file_identifier,
                original_filename=filename,
            )
            db.session.add(db_file)
            db.session.commit()

            return jsonify(
                {"message": "Text content processed successfully", "type": "success"}
            )

        else:
            return (
                jsonify({"message": "Invalid upload type provided", "type": "error"}),
                400,
            )

    except Exception as e:
        print(f"Error in api_upload_file: {str(e)}")
        return jsonify({"message": str(e), "type": "error"}), 500


def get_file_paths(file_identifier):
    """Get all associated file paths for a given file identifier."""
    base_path = os.path.join("dataembedding", file_identifier)
    return {
        "json": f"{base_path}_chunks.json",
        "numpy": f"{base_path}_embeddings.npy",
        "faiss": f"{base_path}.faiss",
    }


def delete_associated_files(file_identifier):
    """Delete all associated files for a given file identifier."""
    if not file_identifier or not str(file_identifier).strip():
        return
    try:
        file_paths = get_file_paths(file_identifier)
        for file_path in file_paths.values():
            if os.path.exists(file_path):
                os.remove(file_path)
    except Exception as e:
        print(f"Error deleting associated files: {str(e)}")


@main.route("/api/faq/<int:faq_id>", methods=["PUT", "DELETE"])
@login_required
def api_manage_faq(faq_id):
    try:
        faq = FAQ.query.get_or_404(faq_id)

        if request.method == "DELETE":
            db.session.delete(faq)
            db.session.commit()
            return jsonify({"message": "FAQ deleted successfully"})

        elif request.method == "PUT":
            data = request.get_json()
            if not data or "question" not in data or "answer" not in data:
                return jsonify({"error": "Question and answer are required"}), 400

            # Update FAQ content

            try:
                faq.question = data["question"]
                faq.answer = data["answer"]
                db.session.commit()
                return jsonify({"message": "FAQ updated successfully"})

            except Exception as e:
                db.session.rollback()
                # print(f"Error generating embeddings: {e}")
                return jsonify({"error": str(e)}), 500

    except Exception as e:
        db.session.rollback()
        # print(f"Error managing FAQ: {e}")
        return jsonify({"error": str(e)}), 500


@main.route("/api/faq/bulk-delete", methods=["DELETE"])
@login_required
def api_bulk_delete_faqs():
    try:
        # Delete all FAQs
        deleted_count = FAQ.query.delete()
        db.session.commit()
        
        return jsonify({
            "type": "success", 
            "message": f"Successfully deleted all {deleted_count} FAQs"
        })
    except Exception as e:
        db.session.rollback()
        return jsonify({
            "type": "error", 
            "message": f"Failed to delete FAQs: {str(e)}"
        }), 500


@main.route("/api/file/<int:id>", methods=["DELETE"])
@login_required
def api_delete_file(id):
    try:
        file = File.query.get_or_404(id)

        # Delete associated files
        delete_associated_files(file.file_identifier)

        # Delete from database
        db.session.delete(file)
        db.session.commit()

        return jsonify(
            {
                "message": "File and associated data deleted successfully",
                "type": "success",
            }
        )
    except Exception as e:
        return jsonify({"message": str(e), "type": "error"}), 500


@main.route("/api/file/bulk-delete", methods=["DELETE"])
@login_required
def api_bulk_delete_files():
    try:
        # Get all files to delete associated data
        files = File.query.all()
        for file in files:
            delete_associated_files(file.file_identifier)
        
        # Delete all files
        deleted_count = File.query.delete()
        db.session.commit()
        
        return jsonify({
            "type": "success", 
            "message": f"Successfully deleted all {deleted_count} files and associated data"
        })
    except Exception as e:
        db.session.rollback()
        return jsonify({
            "type": "error", 
            "message": f"Failed to delete files: {str(e)}"
        }), 500


@main.route("/api/query/<int:id>", methods=["DELETE"])
@login_required
def api_delete_query(id):
    try:
        query = Query.query.get_or_404(id)
        db.session.delete(query)
        db.session.commit()
        return jsonify({"message": "Query deleted successfully", "type": "success"})
    except Exception as e:
        return jsonify({"message": str(e), "type": "error"}), 500


@main.route("/api/queries/bulk-delete", methods=["DELETE"])
@login_required
def api_bulk_delete_queries():
    try:
        # Delete all queries
        deleted_count = Query.query.delete()
        db.session.commit()
        
        return jsonify({
            "type": "success", 
            "message": f"Successfully deleted all {deleted_count} queries"
        })
    except Exception as e:
        db.session.rollback()
        return jsonify({
            "type": "error", 
            "message": f"Failed to delete queries: {str(e)}"
        }), 500


@main.route("/api/new-question/<int:id>", methods=["DELETE"])
@login_required
def api_delete_new_question(id):
    try:
        new_question = NewQuestion.query.get_or_404(id)

        # Delete associated files
        file_identifier = f"new_question_{new_question.id}"
        delete_associated_files(file_identifier)

        # Delete from database
        db.session.delete(new_question)
        db.session.commit()

        return jsonify(
            {
                "message": "Question and associated files deleted successfully",
                "type": "success",
            }
        )
    except Exception as e:
        return jsonify({"message": str(e), "type": "error"}), 500


@main.route("/api/new-questions/bulk-delete", methods=["DELETE"])
@login_required
def api_bulk_delete_new_questions():
    try:
        # Get all new questions to delete associated files
        questions = NewQuestion.query.all()
        for question in questions:
            file_identifier = f"new_question_{question.id}"
            delete_associated_files(file_identifier)
        
        # Delete all new questions
        deleted_count = NewQuestion.query.delete()
        db.session.commit()
        
        return jsonify({
            "type": "success", 
            "message": f"Successfully deleted all {deleted_count} questions and associated files"
        })
    except Exception as e:
        db.session.rollback()
        return jsonify({
            "type": "error", 
            "message": f"Failed to delete questions: {str(e)}"
        }), 500


@main.route("/api/queries/delete-all", methods=["DELETE"])
@login_required
def api_delete_all_queries():
    try:
        # Delete all queries
        Query.query.delete()
        db.session.commit()
        return jsonify({"message": "All queries deleted successfully"})
    except Exception as e:
        db.session.rollback()
        # print(f"Error deleting all queries: {e}")
        return jsonify({"error": str(e)}), 500


@main.route("/api/search-faqs", methods=["POST"])
def search_faqs():
    """Search FAQs based on text matching for real-time suggestions."""
    try:
        data = request.get_json()
        query = data.get("query", "").strip()
        
        if not query or len(query) < 3:
            return jsonify({"suggestions": []})
        
        # Get all FAQs
        faqs = FAQ.query.all()
        if not faqs:
            return jsonify({"suggestions": []})
        
        # Text matching algorithm
        query_words = set(query.lower().split())
        scored_faqs = []
        
        for faq in faqs:
            faq_words = set(faq.question.lower().split())
            
            # Calculate word matches
            matching_words = query_words.intersection(faq_words)
            match_count = len(matching_words)
            total_query_words = len(query_words)
            
            if match_count == 0:
                continue
                
            # Calculate match percentage
            match_percentage = (match_count / total_query_words) * 100
            
            # Bonus for exact phrase match
            exact_match_bonus = 0
            if query.lower() in faq.question.lower():
                exact_match_bonus = 20
            
            # Final score
            score = match_percentage + exact_match_bonus
            
            scored_faqs.append({
                "id": faq.id,
                "question": faq.question,
                "score": score
            })
        
        # Sort by score (highest first) and return top 5
        scored_faqs.sort(key=lambda x: x["score"], reverse=True)
        suggestions = scored_faqs[:5]
        
        return jsonify({"suggestions": suggestions})
        
    except Exception as e:
        print(f"Error in search_faqs: {str(e)}")
        return jsonify({"suggestions": []}), 500


@main.route("/api/admin/files/paginated", methods=["GET"])
@login_required
def api_files_paginated():
    """Get paginated files data for admin panel."""
    try:
        page = request.args.get('page', 1, type=int)
        per_page = request.args.get('per_page', 10, type=int)
        search = request.args.get('search', '', type=str)
        order_by = request.args.get('order_by', 'id', type=str)
        order_dir = request.args.get('order_dir', 'desc', type=str)
        
        # Validate per_page to prevent abuse
        per_page = min(per_page, 100)
        
        # Build query with search filter
        query = File.query.join(User, File.user_id == User.id)
        
        if search:
            query = query.filter(
                File.original_filename.ilike(f'%{search}%') |
                File.text.ilike(f'%{search}%')
            )
        
        # Apply ordering
        if hasattr(File, order_by):
            if order_dir == 'desc':
                query = query.order_by(getattr(File, order_by).desc())
            else:
                query = query.order_by(getattr(File, order_by).asc())
        else:
            # Default ordering
            query = query.order_by(File.id.desc())
        
        # Apply pagination
        pagination = query.paginate(
            page=page,
            per_page=per_page,
            error_out=False
        )
        
        # Prepare data with truncated text
        files_data = []
        for file in pagination.items:
            files_data.append({
                'id': file.id,
                'original_filename': file.original_filename,
                'text': file.text[:100] + '...' if len(file.text) > 100 else file.text,
                'user_name': file.user.name if file.user else 'Unknown',
                'created_at': file.created_at.strftime('%Y-%m-%d %H:%M'),
                'file_identifier': file.file_identifier,
                'source': file.source or 'upload',
            })
        
        return jsonify({
            'data': files_data,
            'total': pagination.total,
            'pages': pagination.pages,
            'current_page': pagination.page,
            'per_page': pagination.per_page,
            'has_prev': pagination.has_prev,
            'has_next': pagination.has_next
        })
        
    except Exception as e:
        print(f"Error in api_files_paginated: {str(e)}")
        return jsonify({"error": str(e)}), 500


@main.route("/api/admin/queries/paginated", methods=["GET"])
@login_required
def api_queries_paginated():
    """Get paginated queries data for admin panel."""
    try:
        page = request.args.get('page', 1, type=int)
        per_page = request.args.get('per_page', 10, type=int)
        search = request.args.get('search', '', type=str)
        order_by = request.args.get('order_by', 'id', type=str)
        order_dir = request.args.get('order_dir', 'desc', type=str)
        
        # Validate per_page to prevent abuse
        per_page = min(per_page, 100)
        
        # Build query with search filter
        query = Query.query
        
        if search:
            query = query.filter(
                Query.question.ilike(f'%{search}%') |
                Query.answer.ilike(f'%{search}%')
            )
        
        # Apply ordering
        if hasattr(Query, order_by):
            if order_dir == 'desc':
                query = query.order_by(getattr(Query, order_by).desc())
            else:
                query = query.order_by(getattr(Query, order_by).asc())
        else:
            # Default ordering
            query = query.order_by(Query.id.desc())
        
        # Apply pagination
        pagination = query.paginate(
            page=page,
            per_page=per_page,
            error_out=False
        )
        
        # Prepare data
        queries_data = []
        for query in pagination.items:
            queries_data.append({
                'id': query.id,
                'question': query.question[:100] + '...' if len(query.question) > 100 else query.question,
                'answer': query.answer[:100] + '...' if query.answer and len(query.answer) > 100 else query.answer,
                'answer_found': query.answer_found,
                'happy': query.happy,
                'language': query.language,
                'created_at': query.created_at.strftime('%Y-%m-%d %H:%M')
            })
        
        return jsonify({
            'data': queries_data,
            'total': pagination.total,
            'pages': pagination.pages,
            'current_page': pagination.page,
            'per_page': pagination.per_page,
            'has_prev': pagination.has_prev,
            'has_next': pagination.has_next
        })
        
    except Exception as e:
        print(f"Error in api_queries_paginated: {str(e)}")
        return jsonify({"error": str(e)}), 500


@main.route("/api/chat/<int:query_id>/feedback", methods=["POST"])
def chat_feedback(query_id):
    try:
        data = request.get_json()
        feedback = data.get("feedback")
        message = data.get("message")
        language = data.get("language", "en")

        if feedback not in ["like", "dislike"]:
            return jsonify({"error": "Invalid feedback type"}), 400

        query = Query.query.get_or_404(query_id)
        query.happy = feedback == "like"
        # if feedback == "dislike":
        #     query.answer = message
        db.session.commit()

        # Return success message in the selected language
        if language == "fi":
            message = "Kiitos palautteestasi!"
        else:
            message = "Thank you for your feedback!"

        return jsonify({"message": message})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


def regenerate_missing_embeddings():
    """Regenerate embeddings for records that are missing them."""
    try:
        # Check Files
        files = File.query.filter_by(file_identifier=None).all()
        for file in files:
            file_identifier = generate_and_store_embeddings(file.text)
            if file_identifier:
                file.file_identifier = file_identifier
                # print(f"Regenerated embedding for file {file.id}")

        # Check FAQs
        faqs = FAQ.query.filter_by(embedding=None).all()
        for faq in faqs:
            combined_text = f"Question: {faq.question}\nAnswer: {faq.answer}"
            file_identifier = generate_and_store_embeddings(combined_text)
            if file_identifier:
                faq.embedding = json.dumps(file_identifier)
                # print(f"Regenerated embedding for FAQ {faq.id}")

        db.session.commit()
        return True
    except Exception as e:
        print(f"Error regenerating embeddings: {e}")
        db.session.rollback()
        return False


@main.route("/debug/content")
@login_required
def debug_content():
    """Debug route to check content in the database."""
    try:
        files = File.query.all()
        faqs = FAQ.query.all()

        file_info = [
            {
                "id": f.id,
                "filename": f.original_filename,
                "has_identifier": bool(f.file_identifier),
                "identifier": f.file_identifier,
            }
            for f in files
        ]

        faq_info = [
            {
                "id": f.id,
                "question": (
                    f.question[:50] + "..." if len(f.question) > 50 else f.question
                ),
                "has_embedding": bool(f.embedding),
                "embedding": f.embedding,
            }
            for f in faqs
        ]

        return jsonify(
            {
                "files": file_info,
                "faqs": faq_info,
                "file_count": len(files),
                "faq_count": len(faqs),
            }
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@main.route("/api/query/<int:id>", methods=["GET"])
@login_required
def api_get_query(id):
    try:
        query = Query.query.get_or_404(id)

        return jsonify(
            {
                "type": "success",
                "query": {
                    "id": query.id,
                    "question": query.question,
                    "answer": query.answer or "",  # Use stored answer
                    "answer_found": query.answer_found,
                    "happy": query.happy,
                    "created_at": query.created_at.strftime("%Y-%m-%d %H:%M"),
                },
            }
        )
    except Exception as e:
        return jsonify({"type": "error", "message": str(e)}), 500

def extract_webpage_text(url, headless=True, wait_time=0, enable_javascript=True):
    """
    Extract clean text from webpage using Selenium WebDriver.
    
    Args:
        url (str): The URL to extract text from
        headless (bool): Run browser in headless mode (default: True)
        wait_time (int): Time to wait for dynamic content to load (default: 5 seconds)
        enable_javascript (bool): Enable JavaScript for dynamic content (default: True)
    
    Returns:
        str: Extracted clean text content, or None if extraction fails
    """
    try:
        # Validate URL
        if not url.startswith(('http://', 'https://')):
            url = 'https://' + url
        
        # Check if Selenium is available
        if not SELENIUM_AVAILABLE:
            print("Selenium not available. Install with: pip install selenium")
            return None
        
        # Extract content using Selenium
        result = extract_with_selenium(url, headless, wait_time, enable_javascript)
        
        if result:
            return result
        else:
            print(f"No content could be extracted from: {url}")
            return None
            
    except Exception as e:
        print(f"Error extracting webpage text: {str(e)}")
        return None

def extract_with_selenium(url, headless=True, wait_time=0, enable_javascript=True):
    """
    Extract content using Selenium WebDriver with enhanced features.
    
    Args:
        url (str): The URL to extract content from
        headless (bool): Run browser in headless mode
        wait_time (int): Time to wait for dynamic content
        enable_javascript (bool): Enable JavaScript execution (recommended for modern sites)
    
    Returns:
        str: Clean extracted text content
    """
    if not SELENIUM_AVAILABLE:
        return None
        
    driver = None
    try:
        # Setup Chrome driver with optimized options
        chrome_options = Options()
        
        # Headless mode
        if headless:
            chrome_options.add_argument("--headless")
        
        # Performance optimizations
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--disable-extensions")
        chrome_options.add_argument("--disable-images")  # Faster loading
        
        # JavaScript control - disabled only if explicitly requested
        if not enable_javascript:
            chrome_options.add_argument("--disable-javascript")
        
        chrome_options.add_argument("--disable-plugins")
        chrome_options.add_argument("--disable-web-security")
        chrome_options.add_argument("--disable-features=VizDisplayCompositor")
        
        # User agent
        chrome_options.add_argument("--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36")
        
        # Window size
        chrome_options.add_argument("--window-size=1920,1080")
        
        # Initialize driver (assumes ChromeDriver is in PATH)
        driver = webdriver.Chrome(options=chrome_options)
        driver.set_page_load_timeout(30)
        
        # Navigate to page
        print(f"🌐 Loading page: {url}")
        driver.get(url)
        
        # Wait for page to load
        print("⏳ Waiting for page to load...")
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.TAG_NAME, "body"))
        )
        
        # Wait for dynamic content to load
        print(f"⏳ Waiting {wait_time} seconds for dynamic content...")
        time.sleep(wait_time)
        
        # Scroll to trigger lazy-loaded content
        if enable_javascript:
            try:
                print("📜 Scrolling page to load all content...")
                # Scroll down gradually to trigger lazy loading
                driver.execute_script("window.scrollTo(0, document.body.scrollHeight/2);")
                time.sleep(0.5)
                driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                time.sleep(0.5)
                driver.execute_script("window.scrollTo(0, 0);")  # Scroll back to top
                time.sleep(0.5)
            except Exception as e:
                print(f"⚠️ Scrolling failed (non-critical): {e}")
        
        # Get page source after JavaScript execution
        print("📄 Extracting page content...")
        html = driver.page_source
        
        # Clean and extract text
        print("🧹 Cleaning and processing text...")
        return clean_and_extract_text(html)
        
    except TimeoutException:
        print("⏰ Page load timeout - trying to extract available content")
        if driver:
            try:
                html = driver.page_source
                return clean_and_extract_text(html)
            except:
                return None
    except Exception as e:
        print(f"❌ Selenium extraction failed: {e}")
        return None
    finally:
        if driver:
            driver.quit()


def clean_and_extract_text(html):
    """Clean HTML and extract meaningful text content as plain text"""
    soup = BeautifulSoup(html, "html.parser")
    
    # Remove unwanted elements
    unwanted_tags = [
        'script', 'style', 'nav', 'header', 'footer', 'aside',
        'advertisement', 'ads', 'ad', 'popup', 'modal',
        'cookie-notice', 'cookie-banner', 'newsletter',
        'social-media', 'share-buttons', 'comment-section',
        'related-articles', 'sidebar', 'menu', 'navigation'
    ]
    
    for tag in unwanted_tags:
        for element in soup.find_all(tag):
            element.decompose()
    
    # Remove elements with specific classes/IDs that are likely unwanted
    # Note: Be careful with wildcards - avoid matching body/html elements
    unwanted_selectors = [
        '[class*="ad-"]', '[class*="advertisement"]', '[class*="popup"]',
        '[class*="modal"]', '[class*="cookie"]', '[class*="newsletter"]',
        'div[class*="social"]', 'div[class*="share"]', 'div[class*="comment"]',
        'aside[class*="sidebar"]', 'div.sidebar',  # More specific sidebar targeting
        '[id*="ad-"]', '[id*="advertisement"]', '[id*="popup"]',
        '[id*="modal"]', '[id*="cookie"]', '[id*="newsletter"]',
        'div[id*="social"]', 'div[id*="share"]', 'div[id*="comment"]',
        'aside[id*="sidebar"]', 'div#sidebar'
    ]
    
    for selector in unwanted_selectors:
        for element in soup.select(selector):
            element.decompose()
    
    # Find main content area
    main_content = find_main_content(soup)
    
    if main_content:
        text = main_content.get_text(separator=' ', strip=True)
    else:
        text = soup.get_text(separator=' ', strip=True)
    
    # Clean up text
    return post_process_text(text)

def find_main_content(soup):
    """Find the main content area of the page"""
    # Common selectors for main content
    main_selectors = [
        'main', 'article', '[role="main"]',
        '.content', '.main-content', '.article-content',
        '.post-content', '.entry-content', '.story-content',
        '#content', '#main', '#article', '#post'
    ]
    
    for selector in main_selectors:
        element = soup.select_one(selector)
        if element and len(element.get_text(strip=True)) > 100:
            return element
    
    # Fallback: find the element with most text content
    all_elements = soup.find_all(['div', 'section', 'article'])
    max_length = 0
    main_element = None
    
    for element in all_elements:
        text_length = len(element.get_text(strip=True))
        if text_length > max_length and text_length > 100:
            max_length = text_length
            main_element = element
    
    return main_element

def post_process_text(text):
    """Clean and format extracted text as plain text"""
    # Remove excessive whitespace first
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    
    # Remove common unwanted patterns
    unwanted_patterns = [
        r'Cookie\s+Policy[^.]*\.',
        r'Privacy\s+Policy[^.]*\.',
        r'Terms\s+of\s+Service[^.]*\.',
        r'Subscribe\s+to\s+our\s+newsletter[^.]*\.',
        r'Follow\s+us\s+on[^.]*\.',
        r'Share\s+this\s+article[^.]*\.',
        r'Advertisement[^.]*\.',
        r'Sponsored\s+Content[^.]*\.',
        r'We use cookies.*?(?=\.|\n)',
        r'By clicking.*?consent.*?(?=\.|\n)',
        r'Accept All.*?Reject All.*?(?=\.|\n)',
    ]
    
    for pattern in unwanted_patterns:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE)
    
    # Remove excessive whitespace again after pattern removal
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    
    # Only filter if text is very short (likely just title/navigation)
    # Otherwise return the cleaned text as-is
    if len(text) > 100:
        return text
    else:
        return text if len(text) > 20 else ""

    """Clean and format extracted text as plain text"""
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove common unwanted patterns
    unwanted_patterns = [
        r'Cookie\s+Policy.*?(?=\n|$)',
        r'Privacy\s+Policy.*?(?=\n|$)',
        r'Terms\s+of\s+Service.*?(?=\n|$)',
        r'Subscribe\s+to\s+our\s+newsletter.*?(?=\n|$)',
        r'Follow\s+us\s+on.*?(?=\n|$)',
        r'Share\s+this\s+article.*?(?=\n|$)',
        r'Advertisement.*?(?=\n|$)',
        r'Sponsored\s+Content.*?(?=\n|$)',
    ]
    
    for pattern in unwanted_patterns:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE)
    
    # Remove lines that are too short (likely navigation or metadata)
    lines = text.split('\n')
    cleaned_lines = []
    for line in lines:
        line = line.strip()
        if len(line) > 20:  # Keep lines with substantial content
            cleaned_lines.append(line)
    
    return '\n'.join(cleaned_lines)