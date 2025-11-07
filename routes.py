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
from models import db, User, Role, FAQ, File, Query, Settings, NewQuestion, KnowledgeFact
from urllib.parse import urlparse,parse_qs

from werkzeug.security import generate_password_hash
from datetime import datetime
import os
import sys
from embed_and_search import (
    generate_and_store_embeddings,
    search_across_indices,
    split_into_chunks,
    create_embedding,
    count_tokens,
    truncate_context,
)
from query_preprocessing import preprocess_query
from query_intent import infer_query_intent, normalize_relations
from knowledge_graph import (
    process_file_to_graph,
    search_knowledge_graph,
    get_graph_stats,
    clear_all_graphs,
    delete_document_graph,
)
from openai import OpenAI
from openai import APIConnectionError, APIError, AuthenticationError, RateLimitError
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
from langdetect import detect, DetectorFactory, LangDetectException
from collections import defaultdict
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
        # Preprocess query
        preprocessed = preprocess_query(question)
        processed_question = preprocessed.get('normalized', question)
        query_type = preprocessed.get('query_type', 'factual')
        
        # Log query type for debugging
        print(f"Query type detected: {query_type}, keywords: {preprocessed.get('keywords', [])}")

        # Infer higher-level query intent for structured aggregation scenarios
        intent = infer_query_intent(question)
        print(
            "Intent detection:",
            {
                "entity_type": intent.get("entity_type"),
                "relations": intent.get("relations"),
                "filters": intent.get("filters"),
                "confidence": intent.get("confidence"),
                "source": intent.get("source"),
            },
        )
        structured_results = []
        structured_context = ""

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
        # Get retrieval method (Vector or KAG)
        retrieval_method_setting = Settings.query.filter_by(key="retrieval_method").first()
        if retrieval_method_setting and retrieval_method_setting.value:
            retrieval_method = retrieval_method_setting.value.strip()
            print(f"DEBUG: Found retrieval_method in DB = '{retrieval_method}'")
        else:
            retrieval_method = "Vector"
            print(f"DEBUG: No retrieval_method in DB, using default = '{retrieval_method}'")
        
        # Ensure it's either Vector or KAG
        if retrieval_method not in ["Vector", "KAG", "SQL"]:
            print(f"DEBUG: Invalid retrieval_method '{retrieval_method}', defaulting to 'Vector'")
            retrieval_method = "Vector"
        
        print(f"DEBUG: Final retrieval_method being used = '{retrieval_method}'")

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

        # Initialize OpenAI client with timeout settings
        client = OpenAI(
            api_key=openai_key.value,
            timeout=30.0,  # 30 second timeout
            max_retries=2  # Retry up to 2 times
        )

        if intent.get("confidence", 0.0) < 0.5:
            llm_intent = infer_query_intent(question, client)
            if llm_intent.get("confidence", 0.0) > intent.get("confidence", 0.0):
                intent = llm_intent
                print("Intent refined via LLM fallback:", intent)

        structured_results, structured_context, structured_source_files = fetch_structured_facts(
            intent
        )
        if structured_results:
            print(
                f"Structured aggregation available: {len(structured_results)} entities, "
                f"context length {len(structured_context.splitlines())} lines"
            )

        sql_mode = retrieval_method == "SQL"

        if sql_mode and structured_results:
            answer_text = format_structured_results_text(structured_results, intent.get("filters"))
            sources_used, used_file_names = build_sources_html_from_structured(structured_source_files)
            query = save_query(answer=answer_text)
            return jsonify({
                "answer": answer_text,
                "sources_used": sources_used,
                "used_file_names": used_file_names,
                "structured_results": [],
                "type": "success",
                "query_id": query.id,
            })

        # 3. Search across file embeddings or knowledge graph
        files = File.query.all()
        file_ids = [f.file_identifier for f in files if f.file_identifier]
        file_id_to_title = {
            f.file_identifier: f.original_filename for f in files if f.file_identifier
        }

        # get chunk number from settings
        chunk_number = Settings.query.filter_by(key="chunk_number").first()
        chunk_number = chunk_number.value if chunk_number and chunk_number.value else 3
        chunk_number = int(chunk_number)

        # Helper function for KAG retrieval (two-phase)
        def get_kag_results(query, processed_query):
            """Two-phase KAG retrieval: Phase 1 - Get all matching triples, Phase 2 - Extract unique entities"""
            # Phase 1: Find all matching triples (no limit - chunk_number only applies to Vector)
            kg_results = search_knowledge_graph(processed_query, top_k=None)
            
            if not kg_results:
                return []
            
            # Phase 2: Extract unique entities (subjects and objects)
            unique_entities = set()
            all_triples = []
            
            for triple in kg_results:
                subject = triple.get('subject', '').strip()
                obj = triple.get('object', '').strip()
                
                if subject:
                    unique_entities.add(subject)
                if obj:
                    unique_entities.add(obj)
                
                all_triples.append(triple)
            
            # Phase 3: Format for context
            # Build entities list and triples context in a more natural format
            entities_list = sorted(list(unique_entities))
            
            # Format triples in a more readable way
            triples_context = "\n".join([
                f"- {t.get('subject', '')} {t.get('relation', '')} {t.get('object', '')}"
                for t in all_triples
            ])
            
            # Combine into single context chunk with better formatting
            combined_context = f"Relevant information from knowledge graph:\n{triples_context}"
            
            # Return as results format (single result with all entities and triples)
            # Use first triple's source_id for file reference
            source_id = all_triples[0].get('source_id') if all_triples else None
            file_id = None
            if source_id:
                file_obj = File.query.filter_by(id=source_id).first()
                if file_obj and file_obj.file_identifier:
                    file_id = file_obj.file_identifier
            
            return [{
                'file_id': file_id or 'unknown',
                'chunk': combined_context,
                'score': 1.0,  # High score for KAG results
                'distance': 0.0,
                'is_kag': True,  # Flag to identify KAG results
                'entities': entities_list,
                'triples': all_triples
            }]
        
        # Helper function for Vector retrieval
        def get_vector_results(query, processed_query, file_ids_list, chunk_num):
            """Vector search retrieval"""
            if not file_ids_list:
                return []
            return search_across_indices(processed_query, file_ids_list, top_k=chunk_num)
        
        results = []
        use_structured_context = bool(structured_results) and sql_mode

        active_method = retrieval_method

        if not use_structured_context:
            if sql_mode:
                active_method = "Vector"
                print("DEBUG: SQL mode without structured results, falling back to Vector retrieval")

            if active_method == "KAG":
                results = get_kag_results(question, processed_question)
            else:
                if not file_ids:
                    return handle_no_content(question, language)
                results = get_vector_results(question, processed_question, file_ids, chunk_number)

        primary_method = active_method

        # Get system prompt (use default with {language} placeholder)
        model_name = openai_model.value if openai_model and openai_model.value else "gpt-4"
        # Default prompt with {language} placeholder support
        default_prompt = (
            "You are a helpful assistant. Use the following context to answer the user's question in {language}. "
            "Provide a natural, human-like answer that synthesizes the information from the context. "
            "For questions about people, places, or entities, create a coherent narrative that combines relevant facts. "
            "For aggregate queries (e.g., 'list all companies'), organize the information clearly but naturally. "
            "Do not simply list numbered items - instead, write in complete sentences and paragraphs. "
            "If the context doesn't contain enough information to fully answer the question, say so clearly. "
            "For casual greetings and chats, reply naturally and ask how you can assist."
        )
        language_name = "English" if language == "en" else "Finnish"
        system_prompt_text = default_prompt.replace("{language}", language_name)
        
        # Build temporary context to check token count for fallback
        temp_context_segments = []
        if structured_context:
            temp_context_segments.append(structured_context)
        temp_context_segments.extend([
            f"Relevant content from {file_id_to_title.get(r.get('file_id', 'unknown'), 'unknown')}:\n{r.get('chunk', '')}"
            if not r.get('is_kag', False) else r.get('chunk', '')
            for r in results
            if r.get("file_id") in file_id_to_title or r.get('is_kag', False)
        ])
        temp_context = "\n\n".join(temp_context_segments)
        
        # Check if we have enough context (>= 100 tokens including system prompt and query)
        temp_total_tokens = (
            count_tokens(system_prompt_text, model_name) +
            count_tokens(question, model_name) +
            count_tokens(temp_context, model_name)
        )
        
        # Fallback system: if insufficient context, try alternative method
        if temp_total_tokens < 100 and results:
            print(f"DEBUG: Insufficient context ({temp_total_tokens} tokens), trying fallback method")
            sys.stdout.flush()
            
            fallback_results = []
            if primary_method == "KAG":
                # Try Vector search as fallback
                if file_ids:
                    fallback_results = get_vector_results(question, processed_question, file_ids, chunk_number)
                    if fallback_results:
                        # Check if fallback has enough tokens
                        fallback_context = "\n\n".join([
                            r.get('chunk', '')
                            for r in fallback_results
                        ])
                        fallback_tokens = (
                            count_tokens(system_prompt_text, model_name) +
                            count_tokens(question, model_name) +
                            count_tokens(fallback_context, model_name)
                        )
                        if fallback_tokens >= 100:
                            results = fallback_results
                            print(f"DEBUG: Using KAG fallback ({fallback_tokens} tokens)")
                            sys.stdout.flush()
            elif primary_method == "Vector":
                # Try KAG search as fallback
                fallback_results = get_kag_results(question, processed_question)
                if fallback_results:
                    # Check if fallback has enough tokens
                    fallback_context = "\n\n".join([
                        r.get('chunk', '')
                        for r in fallback_results
                    ])
                    fallback_tokens = (
                        count_tokens(system_prompt_text, model_name) +
                        count_tokens(question, model_name) +
                        count_tokens(fallback_context, model_name)
                    )
                    if fallback_tokens >= 100:
                        results = fallback_results
                        print(f"DEBUG: Using KAG fallback ({fallback_tokens} tokens)")
                        sys.stdout.flush()
            else:
                print("DEBUG: Skipping fallback - SQL mode without alternate retrieval")
        
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

        # 4. Truncate context if needed based on token limits
        
        # Truncate results if they exceed token limits
        truncated_results = truncate_context(
            results=results,
            system_prompt=system_prompt_text,
            user_query=question,
            history_text=history_text,
            model=model_name,
            reserve_tokens=600  # Reserve for response (max_tokens=600)
        )
        
        # Build context from structured aggregation + truncated results
        context_parts = []
        used_files = []

        if structured_context:
            source_label = f"Structured facts ({len(structured_results)} entities)"
            derived_from = ", ".join(
                sorted({src["name"] for src in structured_source_files if src.get("name")})
            )
            chunk_text = structured_context
            if derived_from:
                chunk_text += f"\n\nDerived from: {derived_from}"
            context_parts.append("Structured facts:\n" + structured_context)
            used_files.append(
                {
                    "name": source_label,
                    "chunk": chunk_text,
                }
            )

        for r in truncated_results:
            if r.get('is_kag', False):
                # KAG results already formatted with entities and triples
                context_parts.append(r.get('chunk', ''))
            elif r.get("file_id") in file_id_to_title:
                # Vector results - standard format
                context_parts.append(f"Relevant content from {file_id_to_title[r['file_id']]}:\n{r['chunk']}")
            elif r.get("file_id"):
                # Vector results with file_id not in mapping (fallback)
                file_id = r.get("file_id", "Unknown source")
                context_parts.append(f"Relevant content from {file_id}:\n{r.get('chunk', '')}")

            if r.get('is_kag', False):
                used_files.append({
                    "name": "Knowledge Graph Results",
                    "chunk": r.get("chunk", ""),
                })
                print(f"DEBUG: Added KAG result to sources")
            elif r.get("file_id") in file_id_to_title:
                used_files.append({
                    "name": file_id_to_title[r["file_id"]],
                    "chunk": r.get("chunk", ""),
                })
                print(f"DEBUG: Added Vector result to sources: {file_id_to_title[r['file_id']]}")
            elif r.get("file_id"):
                file_id = r.get("file_id", "Unknown source")
                used_files.append({
                    "name": file_id,
                    "chunk": r.get("chunk", ""),
                })
                print(f"DEBUG: Added Vector result to sources (fallback): {file_id}")
            else:
                print(f"DEBUG: WARNING - Result missing file_id and is_kag flag, skipping from sources")
                print(f"DEBUG: Result keys: {list(r.keys())}")

        context = "\n\n".join(context_parts)

        # Log token counts for debugging
        context_tokens = count_tokens(context, model_name)
        system_tokens = count_tokens(system_prompt_text, model_name)
        query_tokens = count_tokens(question, model_name)
        history_tokens = count_tokens(history_text, model_name) if history_text else 0
        total_tokens = system_tokens + query_tokens + history_tokens + context_tokens
        print(f"Token counts - System: {system_tokens}, Query: {query_tokens}, History: {history_tokens}, Context: {context_tokens}, Total: {total_tokens}")

        # 5. Generate answer via OpenAI
        # Structure the prompt more clearly
        if history_text:
            prompt_content = f"""Based on the conversation history and the retrieved context below, please answer the user's question in a natural, human-like way.

Conversation history:
{history_text}

User's question: {question}

Retrieved context:
{context}

Please provide a comprehensive answer that synthesizes the information from the context. Write in complete sentences and paragraphs, not as a numbered list."""
        else:
            prompt_content = f"""Based on the retrieved context below, please answer the user's question in a natural, human-like way.

User's question: {question}

Retrieved context:
{context}

Please provide a comprehensive answer that synthesizes the information from the context. Write in complete sentences and paragraphs, not as a numbered list."""
        
        try:
            response = client.chat.completions.create(
                model=(
                    openai_model.value if openai_model and openai_model.value else "gpt-4"
                ),
                messages=[
                    {
                        "role": "system",
                        "content": system_prompt_text,
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
        except APIConnectionError as e:
            error_msg = {
                "en": "Unable to connect to OpenAI API. Please check your internet connection and try again.",
                "fi": "Yhteyttä OpenAI API:in ei voitu muodostaa. Tarkista internetyhteytesi ja yritä uudelleen."
            }.get(language, "en")
            print(f"OpenAI Connection Error: {str(e)}")
            return jsonify({"answer": error_msg, "type": "error"}), 500
        except AuthenticationError as e:
            error_msg = {
                "en": "Invalid OpenAI API key. Please check your API key in Settings.",
                "fi": "Virheellinen OpenAI API-avain. Tarkista API-avaimesi Asetukset-sivulla."
            }.get(language, "en")
            print(f"OpenAI Authentication Error: {str(e)}")
            return jsonify({"answer": error_msg, "type": "error"}), 500
        except RateLimitError as e:
            error_msg = {
                "en": "OpenAI API rate limit exceeded. Please wait a moment and try again.",
                "fi": "OpenAI API:n nopeusrajoitus ylitetty. Odota hetki ja yritä uudelleen."
            }.get(language, "en")
            print(f"OpenAI Rate Limit Error: {str(e)}")
            return jsonify({"answer": error_msg, "type": "error"}), 429
        except APIError as e:
            error_msg = {
                "en": f"OpenAI API error: {str(e)}. Please try again later.",
                "fi": f"OpenAI API -virhe: {str(e)}. Yritä myöhemmin uudelleen."
            }.get(language, "en")
            print(f"OpenAI API Error: {str(e)}")
            return jsonify({"answer": error_msg, "type": "error"}), 500
        
        # Initialize sources_used and used_file_names
        sources_used = ""  # Initialize to empty string
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
        return jsonify({
            "answer": answer,
            "sources_used": sources_used,
            "used_file_names": used_file_names,
            "structured_results": structured_results,
            "type": "success",
            "query_id": query.id,
        })

    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"Error in chat route: {str(e)}")
        print(f"Traceback: {error_trace}")
        
        # Provide user-friendly error message
        error_msg = {
            "en": "An error occurred while processing your request. Please try again later.",
            "fi": "Pyyntösi käsittelyssä tapahtui virhe. Yritä myöhemmin uudelleen."
        }.get(language, "en")
        
        return jsonify({"answer": error_msg, "type": "error"}), 500


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


@main.route("/admin/settings", methods=["GET", "POST"])
@login_required
def admin_settings():
    if request.method == "POST":
        print("DEBUG: admin_settings POST route called")
        sys.stdout.flush()
        print(f"DEBUG: Form data keys: {list(request.form.keys())}")
        sys.stdout.flush()
        print(f"DEBUG: retrieval_method in form: {'retrieval_method' in request.form}")
        sys.stdout.flush()
        print(f"DEBUG: retrieval_method value: {request.form.get('retrieval_method', 'NOT FOUND')}")
        sys.stdout.flush()
        try:
            # Handle file uploads
            upload_dir = os.path.join(current_app.static_folder, "uploads")
            os.makedirs(upload_dir, exist_ok=True)
            
            # Only process fields that are actually in the form
            # This way each form only updates its own fields
            settings_to_update = {}
            
            # RAG Settings form fields
            if "retrieval_method" in request.form:
                settings_to_update["retrieval_method"] = request.form.get("retrieval_method")
                print(f"DEBUG: Adding retrieval_method: '{settings_to_update['retrieval_method']}'")
            if "chunk_number" in request.form:
                settings_to_update["chunk_number"] = request.form.get("chunk_number")
            if "message_history" in request.form:
                settings_to_update["message_history"] = request.form.get("message_history")
            if "show_matched_text" in request.form:
                settings_to_update["show_matched_text"] = request.form.get("show_matched_text", "yes")
            if "faq_search_enabled" in request.form:
                settings_to_update["faq_search_enabled"] = request.form.get("faq_search_enabled", "yes")
            if "zip_size_limit" in request.form:
                zip_size_limit = request.form.get("zip_size_limit")
                if zip_size_limit:
                    try:
                        # Validate and store as integer
                        zip_size_limit = int(zip_size_limit)
                        if 10 <= zip_size_limit <= 5000:
                            settings_to_update["zip_size_limit"] = zip_size_limit
                        else:
                            print(f"DEBUG: zip_size_limit out of range: {zip_size_limit}")
                    except ValueError:
                        print(f"DEBUG: Invalid zip_size_limit value: {zip_size_limit}")
            
            # Model Settings form fields
            if "openai_key" in request.form:
                settings_to_update["openai_key"] = request.form.get("openai_key")
            if "openai_model" in request.form:
                model_name = request.form.get("openai_model")
                settings_to_update["openai_model"] = model_name
                # Save context limit for this specific model
                if "model_context_limit" in request.form:
                    context_limit = request.form.get("model_context_limit")
                    if context_limit and context_limit.strip():
                        # Store as model-specific setting: model_context_limit_{model_name}
                        settings_to_update[f"model_context_limit_{model_name}"] = context_limit.strip()
                        print(f"DEBUG: Saving context limit for {model_name}: {context_limit} tokens")
                        sys.stdout.flush()
            # General Settings form fields
            if "logo" in request.form:
                settings_to_update["logo"] = request.form.get("logo")
            if "copyright" in request.form:
                settings_to_update["copyright"] = request.form.get("copyright")
            
            # About & Contact form fields
            if "about" in request.form:
                settings_to_update["about"] = request.form.get("about")
            if "contact" in request.form:
                settings_to_update["contact"] = request.form.get("contact")
            
            # Theme settings (from theme form)
            if "theme_primary_color" in request.form:
                settings_to_update["theme_primary_color"] = request.form.get("theme_primary_color", "#5b1fa6")
            if "theme_secondary_color" in request.form:
                settings_to_update["theme_secondary_color"] = request.form.get("theme_secondary_color", "#d1aff3")
            if "theme_bot_message_color" in request.form:
                settings_to_update["theme_bot_message_color"] = request.form.get("theme_bot_message_color", "#f3f0fa")
            if "theme_background_start" in request.form:
                settings_to_update["theme_background_start"] = request.form.get("theme_background_start", "#f5f0ff")
            if "theme_background_end" in request.form:
                settings_to_update["theme_background_end"] = request.form.get("theme_background_end", "#ede3f7")
            if "theme_accent_color" in request.form:
                settings_to_update["theme_accent_color"] = request.form.get("theme_accent_color", "#5b1fa6")
            
            print(f"DEBUG: Processing {len(settings_to_update)} settings from form")
            sys.stdout.flush()

            # Handle logo file upload
            logo_file = request.files.get("logo_file")
            if logo_file and logo_file.filename:
                # Delete old logo if exists
                old_logo = Settings.query.filter_by(key="logo_file").first()
                if old_logo and old_logo.value:
                    old_logo_path = os.path.join(upload_dir, old_logo.value)
                    if os.path.exists(old_logo_path):
                        os.remove(old_logo_path)

                # Save new logo
                filename = secure_filename(logo_file.filename)
                logo_file.save(os.path.join(upload_dir, filename))
                settings_to_update["logo_file"] = filename

            # Handle favicon file upload
            favicon_file = request.files.get("favicon_file")
            if favicon_file and favicon_file.filename:
                # Delete old favicon if exists
                old_favicon = Settings.query.filter_by(key="favicon_file").first()
                if old_favicon and old_favicon.value:
                    old_favicon_path = os.path.join(upload_dir, old_favicon.value)
                    if os.path.exists(old_favicon_path):
                        os.remove(old_favicon_path)

                # Save new favicon
                filename = secure_filename(favicon_file.filename)
                favicon_file.save(os.path.join(upload_dir, filename))
                settings_to_update["favicon_file"] = filename

            for key, value in settings_to_update.items():
                # Skip None values to avoid constraint errors
                if value is None:
                    continue
                
                # Strip whitespace for retrieval_method to avoid issues
                if key == "retrieval_method":
                    value = value.strip() if isinstance(value, str) else value
                    print(f"DEBUG: Saving retrieval_method = '{value}' (type: {type(value)})")
                    sys.stdout.flush()
                    
                setting = Settings.query.filter_by(key=key).first()
                if setting:
                    old_value = setting.value
                    setting.value = str(value).strip() if isinstance(value, str) else str(value)
                    if key == "retrieval_method":
                        print(f"DEBUG: Updated retrieval_method from '{old_value}' to '{setting.value}'")
                        sys.stdout.flush()
                else:
                    setting = Settings(key=key, value=str(value).strip() if isinstance(value, str) else str(value))
                    db.session.add(setting)
                    if key == "retrieval_method":
                        print(f"DEBUG: Created new retrieval_method setting with value '{setting.value}'")
                        sys.stdout.flush()

            db.session.commit()
            print("DEBUG: Settings saved successfully, committing to DB")
            sys.stdout.flush()
            return jsonify(
                {"message": "Settings updated successfully", "type": "success"}
            )
        except Exception as e:
            import traceback
            error_trace = traceback.format_exc()
            print(f"DEBUG: Error in admin_settings POST: {str(e)}")
            print(f"DEBUG: Traceback: {error_trace}")
            sys.stdout.flush()
            return (
                jsonify(
                    {"message": f"Error updating settings: {str(e)}", "type": "error"}
                ),
                500,
            )

    settings = {s.key: s.value for s in Settings.query.all()}
    models = []

    if "openai_key" in settings and settings["openai_key"]:
        try:
            client = OpenAI(
                api_key=settings["openai_key"],
                timeout=30.0,  # 30 second timeout
                max_retries=2  # Retry up to 2 times
            )
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
        except APIConnectionError as e:
            print(f"⚠️ Connection error fetching models: {e}")
            models = ["gpt-4"]
        except AuthenticationError as e:
            print(f"⚠️ Authentication error fetching models: {e}")
            models = ["gpt-4"]
        except RateLimitError as e:
            print(f"⚠️ Rate limit error fetching models: {e}")
            models = ["gpt-4"]
        except APIError as e:
            print(f"⚠️ API error fetching models: {e}")
            models = ["gpt-4"]
        except Exception as e:
            print(f"⚠️ Error fetching models: {e}")
            models = ["gpt-4"]
    else:
        models = ["gpt-4"]

    return render_template("admin/settings.html", settings=settings, models=models)


@main.route("/api/models", methods=["GET"])
@login_required
def api_get_models():
    """API endpoint to fetch OpenAI models for the current API key."""
    try:
        openai_key = Settings.query.filter_by(key="openai_key").first()
        
        if not openai_key or not openai_key.value:
            return jsonify({"error": "OpenAI API key is not configured", "models": ["gpt-4"]}), 200
        
        try:
            client = OpenAI(
                api_key=openai_key.value,
                timeout=30.0,  # 30 second timeout
                max_retries=2  # Retry up to 2 times
            )
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
            
            return jsonify({"models": models, "error": None})
        except APIConnectionError as e:
            error_msg = "Unable to connect to OpenAI API. Please check your internet connection."
            print(f"⚠️ Connection error fetching models: {e}")
            return jsonify({"models": ["gpt-4"], "error": error_msg}), 200
        except AuthenticationError as e:
            error_msg = "Invalid OpenAI API key. Please check your API key in Settings."
            print(f"⚠️ Authentication error fetching models: {e}")
            return jsonify({"models": ["gpt-4"], "error": error_msg}), 200
        except RateLimitError as e:
            error_msg = "OpenAI API rate limit exceeded. Please wait a moment and try again."
            print(f"⚠️ Rate limit error fetching models: {e}")
            return jsonify({"models": ["gpt-4"], "error": error_msg}), 200
        except APIError as e:
            error_msg = f"OpenAI API error: {str(e)}"
            print(f"⚠️ API error fetching models: {e}")
            return jsonify({"models": ["gpt-4"], "error": error_msg}), 200
        except Exception as e:
            print(f"⚠️ Error fetching models: {e}")
            return jsonify({"models": ["gpt-4"], "error": str(e)}), 200
    except Exception as e:
        return jsonify({"models": ["gpt-4"], "error": str(e)}), 500


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


def detect_language(text: str) -> str:
    """
    Detect the language of the given text.
    
    Args:
        text: Text to detect language from
    
    Returns:
        Language code ('en' for English, 'fi' for Finnish, or 'en' as default)
    """
    try:
        # Set seed for reproducibility
        DetectorFactory.seed = 0
        
        # Sample text for detection (first 1000 chars for speed)
        sample_text = text[:1000] if len(text) > 1000 else text
        
        # Skip if too short
        if len(sample_text.strip()) < 10:
            return 'en'  # Default to English
        
        detected = detect(sample_text)
        
        # Map detected language to our supported languages
        # Only support English (en) and Finnish (fi), default to English
        if detected == 'fi':
            return 'fi'
        else:
            return 'en'  # Default to English for all other languages
    except (LangDetectException, Exception) as e:
        print(f"Language detection error: {e}, defaulting to English")
        return 'en'  # Default to English on error


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


def delete_file_completely(file_id):
    """
    Delete a file completely: database record, embeddings, and knowledge graph data.
    
    Args:
        file_id: ID of the file to delete
    
    Returns:
        True if successful, False otherwise
    """
    try:
        file_obj = File.query.filter_by(id=file_id).first()
        if not file_obj:
            return False
        
        file_identifier = file_obj.file_identifier
        
        # Delete embeddings files
        if file_identifier:
            from pathlib import Path
            embeddings_folder = Path("dataembedding")
            embeddings_file = embeddings_folder / f"{file_identifier}_embeddings.npy"
            chunks_file = embeddings_folder / f"{file_identifier}_chunks.json"
            
            if embeddings_file.exists():
                embeddings_file.unlink()
            if chunks_file.exists():
                chunks_file.unlink()
        
        # Delete knowledge graph triples
        from knowledge_graph import KnowledgeGraph
        KnowledgeGraph.query.filter_by(source_id=file_id).delete()
        
        # Delete file record
        db.session.delete(file_obj)
        db.session.flush()
        
        return True
    except Exception as e:
        print(f"Error deleting file {file_id}: {str(e)}")
        db.session.rollback()
        return False


def extract_zip_files(zip_file):
    """
    Extract files from ZIP archive, preserving folder structure.
    
    Args:
        zip_file: Uploaded ZIP file object
    
    Returns:
        List of tuples: [(file_path, relative_path_in_zip), ...]
        Only includes supported file types (.txt, .pdf, .docx)
    """
    import zipfile
    import tempfile
    import shutil
    
    supported_extensions = {'.txt', '.pdf', '.docx'}
    extracted_files = []
    temp_dir = None
    
    try:
        # Create temporary directory for extraction
        temp_dir = tempfile.mkdtemp()
        
        # Extract ZIP file
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            # Get all file names in ZIP
            all_files = zip_ref.namelist()
            
            # Filter and extract supported files
            for file_path in all_files:
                # Skip directories
                if file_path.endswith('/'):
                    continue
                
                # Check if file has supported extension
                file_ext = os.path.splitext(file_path)[1].lower()
                if file_ext not in supported_extensions:
                    continue
                
                # Extract file to temp directory
                extracted_path = os.path.join(temp_dir, file_path)
                os.makedirs(os.path.dirname(extracted_path), exist_ok=True)
                
                try:
                    zip_ref.extract(file_path, temp_dir)
                    extracted_files.append((extracted_path, file_path))
                except Exception as e:
                    print(f"Error extracting {file_path} from ZIP: {str(e)}")
                    continue
        
        return extracted_files, temp_dir
    
    except zipfile.BadZipFile:
        print("Invalid ZIP file")
        if temp_dir and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        return [], None
    except Exception as e:
        print(f"Error extracting ZIP file: {str(e)}")
        if temp_dir and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        return [], None


def generate_failed_files_csv(failed_files):
    """
    Generate CSV content for failed files.
    
    Args:
        failed_files: List of dicts with 'filename', 'error', 'path_in_zip'
    
    Returns:
        CSV content as string
    """
    import csv
    import io
    
    output = io.StringIO()
    writer = csv.writer(output)
    
    # Write header
    writer.writerow(['Filename', 'Error Reason', 'Path in ZIP'])
    
    # Write data rows
    for failed_file in failed_files:
        writer.writerow([
            failed_file.get('filename', ''),
            failed_file.get('error', ''),
            failed_file.get('path_in_zip', '')
        ])
    
    return output.getvalue()


def generate_filename_from_zip_path(zip_path, content=None):
    """
    Generate filename from ZIP path, preserving folder structure.
    
    Args:
        zip_path: Relative path in ZIP (e.g., "folder1/subfolder/file.pdf")
        content: Optional text content for fallback
    
    Returns:
        Safe filename preserving folder structure
    """
    try:
        # Get current time for uniqueness
        current_time = datetime.now().strftime("%H%M")
        
        # Preserve folder structure by replacing / with /
        # But ensure it's safe for database
        # Keep the folder structure as-is
        safe_path = zip_path.replace('\\', '/')  # Normalize path separators
        
        # If path is just filename, use existing logic
        if '/' not in safe_path:
            original_name = os.path.splitext(safe_path)[0]
            extension = os.path.splitext(safe_path)[1]
            safe_name = re.sub(r"[^\w\s-]", "", original_name)
            safe_name = re.sub(r"[-\s]+", "_", safe_name)
            return f"{safe_name}_{current_time}{extension}"
        
        # Preserve folder structure
        # Replace special characters in path components but keep /
        path_parts = safe_path.split('/')
        safe_parts = []
        for part in path_parts:
            if part:
                # Clean each part
                safe_part = re.sub(r"[^\w\s-]", "", part)
                safe_part = re.sub(r"[-\s]+", "_", safe_part)
                safe_parts.append(safe_part)
        
        # Reconstruct path
        safe_path = '/'.join(safe_parts)
        
        # Add timestamp to filename (last part)
        if safe_parts:
            last_part = safe_parts[-1]
            name, ext = os.path.splitext(last_part)
            safe_parts[-1] = f"{name}_{current_time}{ext}"
            safe_path = '/'.join(safe_parts)
        
        return safe_path
    
    except Exception as e:
        print(f"Error generating filename from ZIP path: {str(e)}")
        return f"file_{current_time}"


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

                    # Detect language from text
                    detected_language = detect_language(text)
                    print(f"Detected language for {file.filename}: {detected_language}")

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
                        language=detected_language,
                    )
                    db.session.add(db_file)
                    db.session.flush()  # Flush to get the file.id
                    
                    # Extract and store knowledge graph triples with language awareness
                    try:
                        openai_key = Settings.query.filter_by(key="openai_key").first()
                        if openai_key and openai_key.value:
                            client = OpenAI(
                                api_key=openai_key.value,
                                timeout=30.0,
                                max_retries=2
                            )
                            process_file_to_graph(db_file.id, text, client, detected_language)
                    except Exception as kg_error:
                        print(f"Error processing knowledge graph for file {db_file.id}: {kg_error}")
                        # Don't fail the upload if KG extraction fails
                    
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

            # Detect language from text
            detected_language = detect_language(text)
            print(f"Detected language for URL {url}: {detected_language}")

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
                language=detected_language,
            )
            db.session.add(db_file)
            db.session.flush()  # Flush to get the file.id
            
            # Extract and store knowledge graph triples with language awareness
            try:
                openai_key = Settings.query.filter_by(key="openai_key").first()
                if openai_key and openai_key.value:
                    client = OpenAI(
                        api_key=openai_key.value,
                        timeout=30.0,
                        max_retries=2
                    )
                    process_file_to_graph(db_file.id, text, client, detected_language)
            except Exception as kg_error:
                print(f"Error processing knowledge graph for URL file {db_file.id}: {kg_error}")
                # Don't fail the upload if KG extraction fails
            
            db.session.commit()

            return jsonify(
                {"message": "URL content processed successfully", "type": "success"}
            )

        elif upload_type == "text":
            text = request.form.get("text", "").strip()
            if not text:
                return jsonify({"message": "No text provided", "type": "error"}), 400

            # Detect language from text
            detected_language = detect_language(text)
            print(f"Detected language for text input: {detected_language}")

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
                language=detected_language,
            )
            db.session.add(db_file)
            db.session.flush()  # Flush to get the file.id
            
            # Extract and store knowledge graph triples with language awareness
            try:
                openai_key = Settings.query.filter_by(key="openai_key").first()
                if openai_key and openai_key.value:
                    client = OpenAI(
                        api_key=openai_key.value,
                        timeout=30.0,
                        max_retries=2
                    )
                    process_file_to_graph(db_file.id, text, client, detected_language)
            except Exception as kg_error:
                print(f"Error processing knowledge graph for text file {db_file.id}: {kg_error}")
                # Don't fail the upload if KG extraction fails
            
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


@main.route("/api/file/zip", methods=["POST"])
@login_required
def api_upload_zip():
    """Handle ZIP file upload, extract files, and process them sequentially."""
    import zipfile
    import shutil
    import base64
    import tempfile
    
    temp_dir = None
    try:
        # Get ZIP file
        zip_file = request.files.get("zip")
        if not zip_file or zip_file.filename == "":
            return jsonify({"message": "No ZIP file selected", "type": "error"}), 400
        
        # Check file extension
        if not zip_file.filename.lower().endswith('.zip'):
            return jsonify({"message": "Invalid file type. Please upload a ZIP file.", "type": "error"}), 400
        
        # Get ZIP size limit from settings
        zip_size_limit_setting = Settings.query.filter_by(key="zip_size_limit").first()
        zip_size_limit_mb = int(zip_size_limit_setting.value) if zip_size_limit_setting and zip_size_limit_setting.value else 200
        zip_size_limit_bytes = zip_size_limit_mb * 1024 * 1024  # Convert MB to bytes
        
        # Check ZIP file size
        zip_file.seek(0, os.SEEK_END)
        zip_file_size = zip_file.tell()
        zip_file.seek(0)
        
        if zip_file_size > zip_size_limit_bytes:
            return jsonify({
                "message": f"ZIP file size ({zip_file_size / 1024 / 1024:.2f} MB) exceeds limit ({zip_size_limit_mb} MB)",
                "type": "error"
            }), 400
        
        # Validate ZIP file
        try:
            with zipfile.ZipFile(zip_file, 'r') as test_zip:
                test_zip.testzip()
        except zipfile.BadZipFile:
            return jsonify({"message": "Invalid or corrupted ZIP file", "type": "error"}), 400
        
        # Extract ZIP files
        zip_file.seek(0)  # Reset file pointer
        extracted_files, temp_dir = extract_zip_files(zip_file)
        
        if not extracted_files:
            return jsonify({
                "message": "ZIP file contains no supported files (PDF, DOCX, TXT)",
                "type": "error"
            }), 400
        
        # Process files sequentially
        success_count = 0
        failed_files = []
        
        # Get OpenAI client for KG processing
        openai_key = Settings.query.filter_by(key="openai_key").first()
        openai_client = None
        if openai_key and openai_key.value:
            try:
                openai_client = OpenAI(
                    api_key=openai_key.value,
                    timeout=30.0,
                    max_retries=2
                )
            except Exception as e:
                print(f"Error initializing OpenAI client: {str(e)}")
        
        for extracted_path, relative_path in extracted_files:
            try:
                # Generate base filename from ZIP path (preserve folder structure)
                # First, create a base filename without timestamp for duplicate checking
                safe_path = relative_path.replace('\\', '/')
                path_parts = safe_path.split('/')
                safe_parts = []
                for part in path_parts:
                    if part:
                        safe_part = re.sub(r"[^\w\s-]", "", part)
                        safe_part = re.sub(r"[-\s]+", "_", safe_part)
                        safe_parts.append(safe_part)
                base_path = '/'.join(safe_parts)
                
                # Check for duplicate by matching base path (without timestamp)
                # Look for files where original_filename starts with base_path
                existing_file = File.query.filter(
                    File.original_filename.like(f"{base_path}%")
                ).first()
                
                if existing_file:
                    # Delete old file completely
                    if not delete_file_completely(existing_file.id):
                        print(f"Warning: Could not delete old file {existing_file.id}")
                
                # Generate final filename with timestamp
                filename = generate_filename_from_zip_path(relative_path)
                
                # Open file for processing
                # Create a file-like object for extract_text_from_file
                class FileWrapper:
                    def __init__(self, file_path, filename):
                        self.file_path = file_path
                        self.filename = filename
                        self._file = None
                    
                    def read(self):
                        if not self._file:
                            self._file = open(self.file_path, 'rb')
                        # Reset to beginning if needed
                        self._file.seek(0)
                        return self._file.read()
                    
                    def seek(self, pos):
                        if not self._file:
                            self._file = open(self.file_path, 'rb')
                        self._file.seek(pos)
                    
                    def close(self):
                        if self._file:
                            self._file.close()
                            self._file = None
                
                file_wrapper = FileWrapper(extracted_path, relative_path)
                
                try:
                    # Extract text
                    text = extract_text_from_file(file_wrapper)
                finally:
                    file_wrapper.close()
                
                if not text:
                    failed_files.append({
                        'filename': os.path.basename(relative_path),
                        'error': 'Could not extract text from file',
                        'path_in_zip': relative_path
                    })
                    continue
                
                # Detect language from text
                detected_language = detect_language(text)
                print(f"Detected language for {relative_path}: {detected_language}")
                
                # Generate embeddings
                file_identifier = generate_and_store_embeddings(text)
                if not file_identifier:
                    failed_files.append({
                        'filename': os.path.basename(relative_path),
                        'error': 'Failed to generate embeddings',
                        'path_in_zip': relative_path
                    })
                    continue
                
                # Create database record
                db_file = File(
                    text=text,
                    user_id=current_user.id,
                    file_identifier=file_identifier,
                    original_filename=filename,
                    language=detected_language,
                )
                db.session.add(db_file)
                db.session.flush()  # Flush to get the file.id
                
                # Extract and store knowledge graph triples with language awareness
                if openai_client:
                    try:
                        process_file_to_graph(db_file.id, text, openai_client, detected_language)
                    except Exception as kg_error:
                        print(f"Error processing knowledge graph for file {db_file.id}: {kg_error}")
                        # Don't fail the upload if KG extraction fails
                
                success_count += 1
                    
            except Exception as e:
                failed_files.append({
                    'filename': os.path.basename(relative_path),
                    'error': f'Error processing file: {str(e)}',
                    'path_in_zip': relative_path
                })
                print(f"Error processing {relative_path}: {str(e)}")
                continue
        
        # Commit all successful files
        if success_count > 0:
            db.session.commit()
        
        # Generate CSV for failed files if any
        csv_data = None
        if failed_files:
            csv_content = generate_failed_files_csv(failed_files)
            csv_data = base64.b64encode(csv_content.encode('utf-8')).decode('utf-8')
        
        # Clean up temp directory
        if temp_dir and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        
        # Prepare response
        message = f"Successfully processed {success_count} file(s)"
        if failed_files:
            message += f". {len(failed_files)} file(s) failed to process"
        
        response = {
            "type": "success",
            "message": message,
            "success_count": success_count,
            "failed_count": len(failed_files)
        }
        
        if csv_data:
            response["csv_data"] = csv_data
        
        return jsonify(response)
        
    except Exception as e:
        # Clean up temp directory on error
        if temp_dir and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        
        print(f"Error in api_upload_zip: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"message": f"Error processing ZIP file: {str(e)}", "type": "error"}), 500


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
                # print(f"Deleted associated file: {file_path}")
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
                'file_identifier': file.file_identifier
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

def fetch_structured_facts(intent: dict, max_entities: int = 50, max_facts: int = 300):
    """Retrieve structured facts matching the inferred query intent."""

    if not intent:
        return [], "", []

    relations = intent.get("relations") or []
    filters = intent.get("filters") or {}
    entity_type = intent.get("entity_type")

    if not relations and not filters:
        # Not enough structure to run an aggregated query
        return [], "", []

    query = KnowledgeFact.query

    if relations:
        query = query.filter(KnowledgeFact.normalized_relation.in_(relations))

    if entity_type:
        query = query.filter(
            (KnowledgeFact.entity_type == entity_type)
            | (KnowledgeFact.entity_type.is_(None))
        )

    location_filter = filters.get("location")
    if location_filter:
        query = query.filter(KnowledgeFact.object.ilike(f"%{location_filter}%"))

    facts = (
        query.order_by(KnowledgeFact.confidence.desc(), KnowledgeFact.created_at.desc())
        .limit(max_facts)
        .all()
    )

    if not facts:
        return [], "", []

    aggregated = {}
    unique_files = {}

    for fact in facts:
        subject = fact.subject
        entry = aggregated.setdefault(
            subject,
            {
                "subject": subject,
                "entity_type": fact.entity_type,
                "facts": [],
                "fact_keys": set(),
                "sources": set(),
            },
        )

        if fact.entity_type and not entry.get("entity_type"):
            entry["entity_type"] = fact.entity_type

        fact_key = (fact.normalized_relation, fact.object)
        if fact_key not in entry["fact_keys"]:
            entry["facts"].append(
                {
                    "relation": fact.normalized_relation,
                    "value": fact.object,
                    "raw_relation": fact.relation,
                }
            )
            entry["fact_keys"].add(fact_key)

        if fact.file:
            entry["sources"].add(
                (
                    fact.file.file_identifier,
                    fact.file.original_filename,
                )
            )
            unique_files.setdefault(
                fact.file.file_identifier,
                {
                    "name": fact.file.original_filename,
                    "file_identifier": fact.file.file_identifier,
                },
            )

        if len(aggregated) >= max_entities:
            break

    structured_results = []
    context_lines = []

    for entry in aggregated.values():
        structured_results.append(
            {
                "subject": entry["subject"],
                "entity_type": entry.get("entity_type"),
                "facts": entry["facts"],
                "sources": [
                    {
                        "file_identifier": fid,
                        "file_name": fname,
                    }
                    for fid, fname in entry["sources"]
                ],
            }
        )

        context_lines.append(f"{entry['subject']}:")
        for fact in entry["facts"]:
            context_lines.append(f"  - {fact['relation']}: {fact['value']}")

    structured_context = "\n".join(context_lines)
    structured_sources = list(unique_files.values())

    return structured_results, structured_context, structured_sources

def format_structured_results_text(structured_results, filters=None):
    """Build a deterministic plain-text representation of structured SQL results."""

    lines = []
    filters = filters or {}

    if filters:
        filter_parts = [f"{key}: {value}" for key, value in filters.items() if value]
        if filter_parts:
            lines.append("Filters: " + ", ".join(filter_parts))
            lines.append("")

    for idx, result in enumerate(structured_results, start=1):
        subject = result.get("subject") or f"Entity {idx}"
        lines.append(f"{idx}. {subject}")
        for fact in result.get("facts", []):
            relation = fact.get("relation", "detail").replace("_", " ").title()
            value = fact.get("value", "")
            lines.append(f"   - {relation}: {value}")
        lines.append("")

    return "\n".join(lines).strip()

def build_sources_html_from_structured(structured_source_files):
    if not structured_source_files:
        return "", []

    seen = set()
    entries = []
    used_file_names = []

    for src in structured_source_files:
        name = src.get("name") or src.get("file_identifier") or "Unknown source"
        if name in seen:
            continue
        seen.add(name)
        entries.append(f"- {name}")
        used_file_names.append(name)

    html = "<br><br><b>Sources used:</b><br>" + "<br>".join(entries)
    return html, used_file_names