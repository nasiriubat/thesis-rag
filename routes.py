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
from urllib.parse import urlparse

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
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api.formatters import TextFormatter
from pathlib import Path
from werkzeug.utils import secure_filename

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
        if not openai_key or not openai_key.value:
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

        results = search_across_indices(question, file_ids, top_k=5)
        if not results:
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

        # 5. Generate answer via OpenAI
        system_prompt = {
            "en": "You are a helpful assistant. Use the following context to answer the question in English. If the context doesn't contain enough information to answer the question, say so:",
            "fi": "Olet avulias assistentti. Käytä seuraavaa kontekstia vastataksesi kysymykseen suomeksi. Jos kontekstissa ei ole tarpeeksi tietoa vastataksesi kysymykseen, kerro niin:",
        }

        client = OpenAI(api_key=openai_key.value)
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {
                    "role": "system",
                    "content": system_prompt.get(language, system_prompt["en"]),
                },
                {
                    "role": "user",
                    "content": f"Context: {context}\n\nQuestion: {question}",
                },
            ],
            temperature=0.8,
            max_tokens=500,
        )

        answer = response.choices[0].message.content

        # 6. Save query with the generated answer
        query = save_query(answer=answer)

        return jsonify({"answer": answer, "type": "success", "query_id": query.id})

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
    files = File.query.all()
    settings = {s.key: s.value for s in Settings.query.all()}
    return render_template("admin/files.html", files=files, settings=settings)


@main.route("/admin/queries")
@login_required
def admin_queries():
    queries = Query.query.all()
    settings = {s.key: s.value for s in Settings.query.all()}
    return render_template("admin/queries.html", queries=queries, settings=settings)


@main.route("/admin/settings", methods=["GET", "POST"])
@login_required
def admin_settings():
    if request.method == "POST":
        try:
            # Handle file uploads
            upload_dir = os.path.join(current_app.static_folder, 'uploads')
            os.makedirs(upload_dir, exist_ok=True)

            settings_to_update = {
                "logo": request.form.get("logo"),
                "openai_key": request.form.get("openai_key"),
                "copyright": request.form.get("copyright"),
                "about": request.form.get("about"),
                "contact": request.form.get("contact"),
            }

            # Handle logo file upload
            logo_file = request.files.get('logo_file')
            if logo_file and logo_file.filename:
                # Delete old logo if exists
                old_logo = Settings.query.filter_by(key='logo_file').first()
                if old_logo and old_logo.value:
                    old_logo_path = os.path.join(upload_dir, old_logo.value)
                    if os.path.exists(old_logo_path):
                        os.remove(old_logo_path)
                
                # Save new logo
                filename = secure_filename(logo_file.filename)
                logo_file.save(os.path.join(upload_dir, filename))
                settings_to_update['logo_file'] = filename

            # Handle favicon file upload
            favicon_file = request.files.get('favicon_file')
            if favicon_file and favicon_file.filename:
                # Delete old favicon if exists
                old_favicon = Settings.query.filter_by(key='favicon_file').first()
                if old_favicon and old_favicon.value:
                    old_favicon_path = os.path.join(upload_dir, old_favicon.value)
                    if os.path.exists(old_favicon_path):
                        os.remove(old_favicon_path)
                
                # Save new favicon
                filename = secure_filename(favicon_file.filename)
                favicon_file.save(os.path.join(upload_dir, filename))
                settings_to_update['favicon_file'] = filename

            for key, value in settings_to_update.items():
                setting = Settings.query.filter_by(key=key).first()
                if setting:
                    setting.value = value
                else:
                    setting = Settings(key=key, value=value)
                    db.session.add(setting)

            db.session.commit()
            return jsonify(
                {"message": "Settings updated successfully", "type": "success"}
            )
        except Exception as e:
            return (
                jsonify(
                    {"message": f"Error updating settings: {str(e)}", "type": "error"}
                ),
                500,
            )

    settings = {s.key: s.value for s in Settings.query.all()}
    return render_template("admin/settings.html", settings=settings)


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

        return jsonify({"message": "Question answered and added to FAQs", "type": "success"})
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
                return jsonify({"type": "error", "message": "Question and answer are required"}), 400

            # Create new FAQ
            faq = FAQ(
                question=data["question"],
                answer=data["answer"]
            )
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
                return jsonify({"message": "Question and answer are required", "type": "error"}), 400

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
                    return extract_youtube_text(file_or_url)
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


def extract_youtube_text(url):
    """Extract text from YouTube video using transcript."""
    try:
        # Extract video ID from URL
        video_id = None

        # Handle different YouTube URL formats
        if "youtube.com" in url:
            # Handle standard YouTube URLs
            if "v=" in url:
                video_id = re.search(r"v=([^&]+)", url).group(1)
            # Handle YouTube Shorts
            elif "/shorts/" in url:
                video_id = url.split("/shorts/")[1].split("?")[0]
            # Handle YouTube channel URLs
            elif "/channel/" in url or "/user/" in url:
                return None
        elif "youtu.be" in url:
            # Handle shortened YouTube URLs
            video_id = url.split("/")[-1].split("?")[0]

        if not video_id:
            print(f"Could not extract video ID from URL: {url}")
            return None

        try:
            # Try to get transcript
            transcript = YouTubeTranscriptApi.get_transcript(video_id)
            formatter = TextFormatter()
            return formatter.format_transcript(transcript)
        except Exception as e:
            print(f"Error getting transcript for video {video_id}: {str(e)}")
            # Try to get transcript in a different language if available
            try:
                transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
                # Try to get transcript in English or any available language
                transcript = transcript_list.find_transcript(["en"]).fetch()
                formatter = TextFormatter()
                return formatter.format_transcript(transcript)
            except Exception as e2:
                print(f"Error getting alternative transcript: {str(e2)}")
                return None

    except Exception as e:
        print(f"Error extracting YouTube transcript: {str(e)}")
        return None


def extract_webpage_text(url):
    """Extract text from webpage."""
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")

        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()

        # Get text
        text = soup.get_text()

        # Break into lines and remove leading and trailing space on each
        lines = (line.strip() for line in text.splitlines())
        # Break multi-headlines into a line each
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        # Drop blank lines
        text = "\n".join(chunk for chunk in chunks if chunk)

        return text
    except Exception as e:
        print(f"Error extracting webpage text: {str(e)}")
        return None


def generate_filename(source, content=None):
    """Generate a unique filename based on the source type and content."""
    try:
        # Get current time for uniqueness
        current_time = datetime.now().strftime("%H%M")

        if isinstance(source, str):  # URL or raw text
            if source.startswith(("http://", "https://")):
                # For URLs, use the URL itself as the base name
                parsed_url = urlparse(source)
                # Remove protocol and www
                domain = parsed_url.netloc.replace("www.", "")
                # Get path and remove leading/trailing slashes
                path = parsed_url.path.strip("/")
                # Combine domain and path, replace slashes with underscores
                url_name = f"{domain}_{path}".replace("/", "_")
                # Make it safe for filenames
                safe_name = re.sub(r"[^\w\s-]", "", url_name)
                safe_name = re.sub(r"[-\s]+", "_", safe_name)
                return f"url_{safe_name}_{current_time}"
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
        print("Upload Type:", upload_type)
        print("Request form data:", request.form)
        print("Request files:", request.files)

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

            print(f"Processing URL: {url}")
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
    try:
        file_paths = get_file_paths(file_identifier)
        for file_path in file_paths.values():
            if os.path.exists(file_path):
                os.remove(file_path)
                print(f"Deleted associated file: {file_path}")
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
                print(f"Error generating embeddings: {e}")
                return jsonify({"error": str(e)}), 500

    except Exception as e:
        db.session.rollback()
        print(f"Error managing FAQ: {e}")
        return jsonify({"error": str(e)}), 500


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
        print(f"Error deleting all queries: {e}")
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
                print(f"Regenerated embedding for file {file.id}")

        # Check FAQs
        faqs = FAQ.query.filter_by(embedding=None).all()
        for faq in faqs:
            combined_text = f"Question: {faq.question}\nAnswer: {faq.answer}"
            file_identifier = generate_and_store_embeddings(combined_text)
            if file_identifier:
                faq.embedding = json.dumps(file_identifier)
                print(f"Regenerated embedding for FAQ {faq.id}")

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
        
        return jsonify({
            "type": "success",
            "query": {
                "id": query.id,
                "question": query.question,
                "answer": query.answer or "",  # Use stored answer
                "answer_found": query.answer_found,
                "happy": query.happy,
                "created_at": query.created_at.strftime('%Y-%m-%d %H:%M')
            }
        })
    except Exception as e:
        return jsonify({"type": "error", "message": str(e)}), 500
