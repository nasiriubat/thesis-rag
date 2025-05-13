from flask import render_template, request, jsonify, redirect, url_for, flash, session
from flask_login import login_user, login_required, logout_user, current_user
from app import app, db
from models import User, Role, FAQ, File, Query, Settings, NewQuestion
from werkzeug.security import generate_password_hash
from datetime import datetime
import os
from embed_and_search import generate_and_store_embeddings, search_across_indices
import openai
import json

# Public routes
@app.route('/')
def index():
    faqs = FAQ.query.all()
    settings = {s.key: s.value for s in Settings.query.all()}
    return render_template('public/index.html', faqs=faqs, settings=settings)

@app.route('/about')
def about():
    settings = {s.key: s.value for s in Settings.query.all()}
    return render_template('public/about.html', settings=settings)

@app.route('/contact')
def contact():
    settings = {s.key: s.value for s in Settings.query.all()}
    return render_template('public/contact.html', settings=settings)

@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.json
    question = data.get('question')
    
    if not question:
        return jsonify({'error': 'No question provided', 'type': 'error'}), 400
    
    try:
        # Check if OpenAI key is configured
        openai_key = Settings.query.filter_by(key='openai_key').first()
        if not openai_key or not openai_key.value:
            return jsonify({
                'answer': "This project is not configured properly. Please contact the responsible person.",
                'type': 'error'
            }), 400

        openai.api_key = openai_key.value
        
        # Get all file IDs and FAQ IDs from the database
        file_ids = [str(f.id) for f in File.query.all()]
        faq_ids = [f"faq_{f.id}" for f in FAQ.query.all()]
        
        # Search for relevant content
        results = search_across_indices(question, file_ids + faq_ids)
        
        # Create context from results
        context = "\n".join([r['chunk'] for r in results])
        
        if not results:
            # If no results found, add to new questions
            new_question = NewQuestion(question=question)
            db.session.add(new_question)
            db.session.commit()
            
            # Generate embedding for the new question
            embedding = generate_and_store_embeddings(question, f"new_question_{new_question.id}")
            new_question.embedding = json.dumps(embedding)
            db.session.commit()
            
            return jsonify({
                'answer': "I'm sorry, I couldn't find a specific answer to your question. I've noted it down and our team will review it.",
                'sources': [],
                'type': 'info'
            })
        
        # Generate response using OpenAI
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant. Use the following context to answer the question:"},
                {"role": "user", "content": f"Context: {context}\n\nQuestion: {question}"}
            ]
        )
        answer = response.choices[0].message.content
        
        # Save query
        query = Query(
            question=question,
            answer_found=bool(results),
            happy=False  # Default value, can be updated by user feedback
        )
        db.session.add(query)
        db.session.commit()
        
        return jsonify({
            'answer': answer,
            'sources': [{'text': r['chunk'][:200] + '...', 'score': r['score']} for r in results],
            'type': 'success'
        })
    except Exception as e:
        return jsonify({'error': str(e), 'type': 'error'}), 500

# Admin routes
@app.route('/admin/login', methods=['GET', 'POST'])
def admin_login():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        user = User.query.filter_by(email=email).first()
        
        if user and user.check_password(password):
            login_user(user)
            flash('Welcome back!', 'success')
            return redirect(url_for('admin_dashboard'))
        flash('Invalid email or password', 'error')
    return render_template('admin/login.html')

@app.route('/admin/logout')
@login_required
def admin_logout():
    logout_user()
    flash('You have been logged out successfully', 'success')
    return redirect(url_for('admin_login'))

@app.route('/admin')
@login_required
def admin_dashboard():
    users = User.query.all()
    faqs = FAQ.query.all()
    files = File.query.all()
    queries = Query.query.all()
    settings = {s.key: s.value for s in Settings.query.all()}
    return render_template('admin/dashboard.html', users=users, faqs=faqs, files=files, queries=queries, settings=settings)

# CRUD routes for admin
@app.route('/admin/profile', methods=['GET', 'POST'])
@login_required
def admin_profile():
    if request.method == 'POST':
        try:
            current_user.name = request.form.get('name')
            current_user.phone = request.form.get('phone')
            if request.form.get('password'):
                current_user.set_password(request.form.get('password'))
            db.session.commit()
            flash('Profile updated successfully', 'success')
        except Exception as e:
            flash(f'Error updating profile: {str(e)}', 'error')
    return render_template('admin/profile.html')

@app.route('/admin/roles')
@login_required
def admin_roles():
    roles = Role.query.all()
    settings = {s.key: s.value for s in Settings.query.all()}
    return render_template('admin/roles.html', roles=roles, settings=settings)

@app.route('/admin/users')
@login_required
def admin_users():
    users = User.query.all()
    roles = Role.query.all()
    settings = {s.key: s.value for s in Settings.query.all()}
    return render_template('admin/users.html', users=users, roles=roles, settings=settings)

@app.route('/admin/faqs')
@login_required
def admin_faqs():
    faqs = FAQ.query.all()
    settings = {s.key: s.value for s in Settings.query.all()}
    return render_template('admin/faqs.html', faqs=faqs, settings=settings)

@app.route('/admin/files')
@login_required
def admin_files():
    files = File.query.all()
    settings = {s.key: s.value for s in Settings.query.all()}
    return render_template('admin/files.html', files=files, settings=settings)

@app.route('/admin/queries')
@login_required
def admin_queries():
    queries = Query.query.all()
    settings = {s.key: s.value for s in Settings.query.all()}
    return render_template('admin/queries.html', queries=queries, settings=settings)

@app.route('/admin/settings', methods=['GET', 'POST'])
@login_required
def admin_settings():
    if request.method == 'POST':
        try:
            settings_to_update = {
                'logo': request.form.get('logo'),
                'openai_key': request.form.get('openai_key'),
                'copyright': request.form.get('copyright'),
                'about': request.form.get('about'),
                'contact': request.form.get('contact')
            }
            
            for key, value in settings_to_update.items():
                setting = Settings.query.filter_by(key=key).first()
                if setting:
                    setting.value = value
                else:
                    setting = Settings(key=key, value=value)
                    db.session.add(setting)
            
            db.session.commit()
            return jsonify({'message': 'Settings updated successfully', 'type': 'success'})
        except Exception as e:
            return jsonify({'message': f'Error updating settings: {str(e)}', 'type': 'error'}), 500
    
    settings = {s.key: s.value for s in Settings.query.all()}
    return render_template('admin/settings.html', settings=settings)

@app.route('/admin/new-questions')
@login_required
def admin_new_questions():
    new_questions = NewQuestion.query.order_by(NewQuestion.created_at.desc()).all()
    settings = {s.key: s.value for s in Settings.query.all()}
    return render_template('admin/new_questions.html', new_questions=new_questions, settings=settings)

@app.route('/api/new-question/<int:id>/answer', methods=['POST'])
@login_required
def api_answer_new_question(id):
    try:
        new_question = NewQuestion.query.get_or_404(id)
        answer = request.form.get('answer')
        ai = request.form.get('ai') == 'on'
        
        if not answer:
            return jsonify({'message': 'Answer is required', 'type': 'error'}), 400
        
        # Create new FAQ
        faq = FAQ(
            question=new_question.question,
            answer=answer,
            ai=ai
        )
        db.session.add(faq)
        
        # Generate embedding for the FAQ
        embedding = generate_and_store_embeddings(new_question.question, f"faq_{faq.id}")
        faq.embedding = json.dumps(embedding)
        
        # Delete the new question
        db.session.delete(new_question)
        db.session.commit()
        
        return jsonify({'message': 'Question answered and added to FAQs', 'type': 'success'})
    except Exception as e:
        return jsonify({'message': str(e), 'type': 'error'}), 500

# API routes for CRUD operations
@app.route('/api/role', methods=['POST'])
@login_required
def api_create_role():
    try:
        name = request.form.get('name')
        if not name:
            return jsonify({'message': 'Name is required', 'type': 'error'}), 400
        
        role = Role(name=name)
        db.session.add(role)
        db.session.commit()
        return jsonify({'message': 'Role created successfully', 'type': 'success'})
    except Exception as e:
        return jsonify({'message': str(e), 'type': 'error'}), 500

@app.route('/api/role/<int:id>', methods=['PUT', 'DELETE'])
@login_required
def api_manage_role(id):
    try:
        role = Role.query.get_or_404(id)
        
        if request.method == 'DELETE':
            db.session.delete(role)
            db.session.commit()
            return jsonify({'message': 'Role deleted successfully', 'type': 'success'})
        
        name = request.form.get('name')
        if not name:
            return jsonify({'message': 'Name is required', 'type': 'error'}), 400
        
        role.name = name
        db.session.commit()
        return jsonify({'message': 'Role updated successfully', 'type': 'success'})
    except Exception as e:
        return jsonify({'message': str(e), 'type': 'error'}), 500

@app.route('/api/user', methods=['POST'])
@login_required
def api_create_user():
    try:
        name = request.form.get('name')
        email = request.form.get('email')
        password = request.form.get('password')
        phone = request.form.get('phone')
        role_id = request.form.get('role_id')
        
        if not all([name, email, password]):
            return jsonify({'message': 'Name, email, and password are required', 'type': 'error'}), 400
        
        user = User(
            name=name,
            email=email,
            phone=phone,
            role_id=role_id if role_id else None
        )
        user.set_password(password)
        
        db.session.add(user)
        db.session.commit()
        return jsonify({'message': 'User created successfully', 'type': 'success'})
    except Exception as e:
        return jsonify({'message': str(e), 'type': 'error'}), 500

@app.route('/api/user/<int:id>', methods=['PUT', 'DELETE'])
@login_required
def api_manage_user(id):
    try:
        user = User.query.get_or_404(id)
        
        if request.method == 'DELETE':
            db.session.delete(user)
            db.session.commit()
            return jsonify({'message': 'User deleted successfully', 'type': 'success'})
        
        user.name = request.form.get('name')
        user.email = request.form.get('email')
        user.phone = request.form.get('phone')
        user.role_id = request.form.get('role_id')
        
        if request.form.get('password'):
            user.set_password(request.form.get('password'))
        
        db.session.commit()
        return jsonify({'message': 'User updated successfully', 'type': 'success'})
    except Exception as e:
        return jsonify({'message': str(e), 'type': 'error'}), 500

@app.route('/api/faq', methods=['POST'])
@login_required
def api_create_faq():
    try:
        question = request.form.get('question')
        answer = request.form.get('answer')
        ai = request.form.get('ai') == 'on'
        
        if not all([question, answer]):
            return jsonify({'message': 'Question and answer are required', 'type': 'error'}), 400
        
        faq = FAQ(
            question=question,
            answer=answer,
            ai=ai
        )
        db.session.add(faq)
        db.session.commit()
        return jsonify({'message': 'FAQ created successfully', 'type': 'success'})
    except Exception as e:
        return jsonify({'message': str(e), 'type': 'error'}), 500

@app.route('/api/faq/<int:id>', methods=['PUT', 'DELETE'])
@login_required
def api_manage_faq(id):
    try:
        faq = FAQ.query.get_or_404(id)
        
        if request.method == 'DELETE':
            db.session.delete(faq)
            db.session.commit()
            return jsonify({'message': 'FAQ deleted successfully', 'type': 'success'})
        
        faq.question = request.form.get('question')
        faq.answer = request.form.get('answer')
        faq.ai = request.form.get('ai') == 'on'
        
        db.session.commit()
        return jsonify({'message': 'FAQ updated successfully', 'type': 'success'})
    except Exception as e:
        return jsonify({'message': str(e), 'type': 'error'}), 500

@app.route('/api/file', methods=['POST'])
@login_required
def api_upload_file():
    try:
        if 'file' not in request.files:
            return jsonify({'message': 'No file provided', 'type': 'error'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'message': 'No file selected', 'type': 'error'}), 400
        
        # Extract text based on file type
        text = extract_text_from_file(file)
        
        # Generate embeddings
        file_id = generate_and_store_embeddings(text)
        
        if file_id:
            # Save to database
            db_file = File(
                text=text,
                user_id=current_user.id
            )
            db.session.add(db_file)
            db.session.commit()
            return jsonify({'message': 'File uploaded successfully', 'type': 'success'})
        
        return jsonify({'message': 'Failed to process file', 'type': 'error'}), 500
    except Exception as e:
        return jsonify({'message': str(e), 'type': 'error'}), 500

@app.route('/api/file/<int:id>', methods=['DELETE'])
@login_required
def api_delete_file(id):
    try:
        file = File.query.get_or_404(id)
        db.session.delete(file)
        db.session.commit()
        return jsonify({'message': 'File deleted successfully', 'type': 'success'})
    except Exception as e:
        return jsonify({'message': str(e), 'type': 'error'}), 500

@app.route('/api/query/<int:id>', methods=['DELETE'])
@login_required
def api_delete_query(id):
    try:
        query = Query.query.get_or_404(id)
        db.session.delete(query)
        db.session.commit()
        return jsonify({'message': 'Query deleted successfully', 'type': 'success'})
    except Exception as e:
        return jsonify({'message': str(e), 'type': 'error'}), 500

def extract_text_from_file(file):
    # Implement text extraction based on file type
    # This is a placeholder - implement actual extraction logic
    return file.read().decode('utf-8') 