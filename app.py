"""
Care Companion - Healthcare Assistant like August AI
Multimodal & Multilingual with Flask, SQLite, ChromaDB, Gemini
+ RAG for knowledge_base directory (PDFs, TXT, MD in subdirectories)
"""
import os
import sqlite3
import json
import uuid
from datetime import datetime, timedelta
from functools import wraps

from flask import Flask, render_template, request, jsonify, session, redirect, url_for, g, flash
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename

# Load env
try:
    from dotenv import load_dotenv
    load_dotenv()
except:
    pass

from utils.guardrails import SafetyGuardrails
from utils.chroma_manager import get_kb
from utils.llm_handler import get_llm_handler, LANGUAGES
from utils.prescription_parser import extract_text_from_pdf, is_image_file, is_pdf_file
from utils.email_notifier import send_medication_email, send_schedule_confirmation_email
from utils.rag_engine import get_rag_loader

# Flask setup
app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "care-companion-super-secret-2025-dev")
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(__file__), 'uploads')
app.config['KNOWLEDGE_BASE_DIR'] = os.path.join(os.path.dirname(__file__), 'knowledge_base')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['KNOWLEDGE_BASE_DIR'], exist_ok=True)

DATABASE = os.path.join(os.path.dirname(__file__), 'care_companion.db')

# Scheduler
try:
    from apscheduler.schedulers.background import BackgroundScheduler
    scheduler = BackgroundScheduler()
    scheduler.start()
    SCHEDULER_AVAILABLE = True
except Exception as e:
    print(f"Scheduler not available: {e} - using mock")
    SCHEDULER_AVAILABLE = False
    class MockScheduler:
        def add_job(self, *args, **kwargs):
            pass
    scheduler = MockScheduler()

def get_db():
    db = getattr(g, '_database', None)
    if db is None:
        db = g._database = sqlite3.connect(DATABASE)
        db.row_factory = sqlite3.Row
    return db

def init_db():
    with app.app_context():
        db = get_db()
        cur = db.cursor()
        cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            has_seen_disclaimer BOOLEAN DEFAULT 0,
            has_completed_profile BOOLEAN DEFAULT 0
        )
        """)
        cur.execute("""
        CREATE TABLE IF NOT EXISTS profiles (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER UNIQUE NOT NULL,
            full_name TEXT,
            age INTEGER,
            weight REAL,
            height REAL,
            email TEXT,
            contact TEXT,
            emergency_contact TEXT,
            preferred_language TEXT DEFAULT 'en',
            health_conditions TEXT,
            allergies TEXT,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(user_id) REFERENCES users(id)
        )
        """)
        cur.execute("""
        CREATE TABLE IF NOT EXISTS medication_schedule (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            medicine_name TEXT NOT NULL,
            dosage TEXT,
            frequency TEXT,
            duration TEXT,
            instructions TEXT,
            precautions TEXT,
            times TEXT,
            start_date TEXT,
            end_date TEXT,
            is_active BOOLEAN DEFAULT 1,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(user_id) REFERENCES users(id)
        )
        """)
        cur.execute("""
        CREATE TABLE IF NOT EXISTS chat_threads (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            title TEXT DEFAULT 'New Chat',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(user_id) REFERENCES users(id)
        )
        """)
        cur.execute("""
        CREATE TABLE IF NOT EXISTS chat_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            role TEXT NOT NULL,
            message TEXT NOT NULL,
            message_type TEXT DEFAULT 'text',
            metadata TEXT,
            thread_id INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(user_id) REFERENCES users(id),
            FOREIGN KEY(thread_id) REFERENCES chat_threads(id)
        )
        """)
        # Try add thread_id column if table existed before without it
        try:
            cur.execute("ALTER TABLE chat_history ADD COLUMN thread_id INTEGER")
        except:
            pass
        db.commit()

@app.teardown_appcontext
def close_connection(exception):
    db = getattr(g, '_database', None)
    if db is not None:
        db.close()

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

def get_user_profile(user_id):
    db = get_db()
    cur = db.cursor()
    cur.execute("SELECT * FROM profiles WHERE user_id = ?", (user_id,))
    row = cur.fetchone()
    if row:
        return dict(row)
    return None

# Initialize DB + KB + RAG
init_db()
kb = get_kb()
kb.initialize_if_empty()

# RAG Ingestion from directory
rag_loader = get_rag_loader(kb_dir=app.config['KNOWLEDGE_BASE_DIR'])
print(f"🔍 Scanning knowledge base directory: {app.config['KNOWLEDGE_BASE_DIR']}")
print(f"   Supported: PDFs, TXT, MD — recursive subdirectories")
try:
    # If no Chroma persistent collection, force reingest every startup to populate in-memory fallback
    force_on_start = kb.collection is None
    if force_on_start:
        print("   Chroma not persistent, forcing re-ingestion to in-memory fallback")
    ingestion_report = rag_loader.ingest_directory(kb, force_reingest=force_on_start)
    print(f"📚 RAG Ingestion report: {ingestion_report['discovered_files']} files found, {ingestion_report['processed_files']} processed, {ingestion_report['total_chunks']} chunks added, {ingestion_report['skipped_files']} skipped (log prevents duplicate when Chroma persistent)")
    if ingestion_report['files']:
        for f in ingestion_report['files'][:5]:
            print(f"  - {f['file']}: {f['chunks']} chunks")
    if ingestion_report['discovered_files'] == 0:
        print(f"   ⚠️  No files in {app.config['KNOWLEDGE_BASE_DIR']} — Add your PDFs/TXT/MD here (subdirs supported) and restart or call POST /api/kb/ingest")
except Exception as e:
    print(f"RAG ingestion failed: {e}")
    ingestion_report = {"discovered_files":0,"processed_files":0,"total_chunks":0,"skipped_files":0,"files":[],"errors":[str(e)]}

llm_handler = get_llm_handler()

# Scheduler job
def check_medication_reminders():
    with app.app_context():
        db = get_db()
        cur = db.cursor()
        now = datetime.now()
        current_time_str = now.strftime("%H:%M")
        today_str = now.strftime("%Y-%m-%d")
        cur.execute("SELECT ms.*, u.email as user_email, p.full_name, p.email as profile_email FROM medication_schedule ms JOIN users u ON ms.user_id=u.id LEFT JOIN profiles p ON p.user_id=ms.user_id WHERE ms.is_active=1")
        schedules = cur.fetchall()
        for sched in schedules:
            try:
                times_json = json.loads(sched['times']) if sched['times'] else []
                for t in times_json:
                    if t == current_time_str:
                        start = sched['start_date']
                        end = sched['end_date']
                        if start and today_str < start:
                            continue
                        if end and today_str > end:
                            continue
                        to_email = sched['profile_email'] or sched['user_email']
                        medicine_card = {
                            "name": sched['medicine_name'],
                            "dosage": sched['dosage'],
                            "frequency": sched['frequency'],
                            "duration": sched['duration'],
                            "instructions": sched['instructions'],
                            "precautions": sched['precautions']
                        }
                        send_medication_email(to_email, medicine_card, sched['full_name'] or "User")
            except Exception as e:
                print(f"Reminder check error: {e}")

if SCHEDULER_AVAILABLE and hasattr(scheduler, 'add_job'):
    try:
        scheduler.add_job(func=check_medication_reminders, trigger="interval", minutes=1, id="med_reminder", replace_existing=True)
    except:
        pass

# Routes
@app.route('/')
def landing():
    if 'user_id' in session:
        db = get_db()
        cur = db.cursor()
        cur.execute("SELECT has_seen_disclaimer, has_completed_profile FROM users WHERE id=?", (session['user_id'],))
        user = cur.fetchone()
        if user:
            if not user['has_seen_disclaimer']:
                return redirect(url_for('disclaimer'))
            if not user['has_completed_profile']:
                return redirect(url_for('setup_profile'))
    return render_template('landing.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')
        db = get_db()
        cur = db.cursor()
        cur.execute("SELECT * FROM users WHERE username=? OR email=?", (username, username))
        user = cur.fetchone()
        if user and check_password_hash(user['password_hash'], password):
            session['user_id'] = user['id']
            session['username'] = user['username']
            if not user['has_seen_disclaimer']:
                return redirect(url_for('disclaimer'))
            if not user['has_completed_profile']:
                return redirect(url_for('setup_profile'))
            return redirect(url_for('chat'))
        else:
            flash("Invalid credentials")
            return render_template('login.html', error="Invalid credentials")
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username','').strip()
        email = request.form.get('email','').strip()
        password = request.form.get('password','')
        if not username or not email or not password:
            return render_template('register.html', error="All fields required")
        db = get_db()
        cur = db.cursor()
        try:
            cur.execute("INSERT INTO users (username, email, password_hash) VALUES (?,?,?)",
                        (username, email, generate_password_hash(password)))
            db.commit()
            user_id = cur.lastrowid
            session['user_id'] = user_id
            session['username'] = username
            return redirect(url_for('disclaimer'))
        except sqlite3.IntegrityError:
            return render_template('register.html', error="Username or email already exists")
    return render_template('register.html')

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('landing'))

@app.route('/disclaimer', methods=['GET', 'POST'])
@login_required
def disclaimer():
    db = get_db()
    cur = db.cursor()
    if request.method == 'POST':
        accepted = request.form.get('accept_terms') == 'on'
        accepted_disclaimer = request.form.get('accept_disclaimer') == 'on'
        if accepted and accepted_disclaimer:
            cur.execute("UPDATE users SET has_seen_disclaimer=1 WHERE id=?", (session['user_id'],))
            db.commit()
            return redirect(url_for('setup_profile'))
        else:
            return render_template('disclaimer.html', error="You must accept both to continue")
    cur.execute("SELECT has_seen_disclaimer FROM users WHERE id=?", (session['user_id'],))
    row = cur.fetchone()
    if row and row['has_seen_disclaimer']:
        return redirect(url_for('chat'))
    return render_template('disclaimer.html')

@app.route('/setup-profile', methods=['GET','POST'])
@login_required
def setup_profile():
    db = get_db()
    cur = db.cursor()
    cur.execute("SELECT has_completed_profile FROM users WHERE id=?", (session['user_id'],))
    user_row = cur.fetchone()
    profile = get_user_profile(session['user_id'])
    if request.method == 'POST':
        full_name = request.form.get('full_name','').strip()
        age = request.form.get('age')
        weight = request.form.get('weight')
        height = request.form.get('height')
        email = request.form.get('email','').strip()
        contact = request.form.get('contact','').strip()
        emergency_contact = request.form.get('emergency_contact','').strip()
        preferred_language = request.form.get('preferred_language','en')
        health_conditions = request.form.get('health_conditions','')
        allergies = request.form.get('allergies','')

        if profile:
            cur.execute("""
            UPDATE profiles SET full_name=?, age=?, weight=?, height=?, email=?, contact=?, 
            emergency_contact=?, preferred_language=?, health_conditions=?, allergies=?, updated_at=CURRENT_TIMESTAMP
            WHERE user_id=?
            """, (full_name, age or None, weight or None, height or None, email, contact, emergency_contact, preferred_language, health_conditions, allergies, session['user_id']))
        else:
            cur.execute("""
            INSERT INTO profiles (user_id, full_name, age, weight, height, email, contact, emergency_contact, preferred_language, health_conditions, allergies)
            VALUES (?,?,?,?,?,?,?,?,?,?,?)
            """, (session['user_id'], full_name, age or None, weight or None, height or None, email, contact, emergency_contact, preferred_language, health_conditions, allergies))
        cur.execute("UPDATE users SET has_completed_profile=1 WHERE id=?", (session['user_id'],))
        db.commit()
        return redirect(url_for('chat'))

    return render_template('setup_profile.html', profile=profile, languages=LANGUAGES, is_setup=True)

@app.route('/profile', methods=['GET','POST'])
@login_required
def profile_page():
    profile = get_user_profile(session['user_id'])
    if request.method == 'POST':
        db = get_db()
        cur = db.cursor()
        full_name = request.form.get('full_name','').strip()
        age = request.form.get('age')
        weight = request.form.get('weight')
        height = request.form.get('height')
        email = request.form.get('email','').strip()
        contact = request.form.get('contact','').strip()
        emergency_contact = request.form.get('emergency_contact','').strip()
        preferred_language = request.form.get('preferred_language','en')
        health_conditions = request.form.get('health_conditions','')
        allergies = request.form.get('allergies','')
        if profile:
            cur.execute("""
            UPDATE profiles SET full_name=?, age=?, weight=?, height=?, email=?, contact=?, 
            emergency_contact=?, preferred_language=?, health_conditions=?, allergies=?, updated_at=CURRENT_TIMESTAMP
            WHERE user_id=?
            """, (full_name, age or None, weight or None, height or None, email, contact, emergency_contact, preferred_language, health_conditions, allergies, session['user_id']))
        else:
            cur.execute("""
            INSERT INTO profiles (user_id, full_name, age, weight, height, email, contact, emergency_contact, preferred_language, health_conditions, allergies)
            VALUES (?,?,?,?,?,?,?,?,?,?,?)
            """, (session['user_id'], full_name, age or None, weight or None, height or None, email, contact, emergency_contact, preferred_language, health_conditions, allergies))
        db.commit()
        profile = get_user_profile(session['user_id'])
        return render_template('setup_profile.html', profile=profile, languages=LANGUAGES, is_setup=False, success="Profile updated successfully!")

    return render_template('setup_profile.html', profile=profile, languages=LANGUAGES, is_setup=False)

@app.route('/chat')
@login_required
def chat():
    db = get_db()
    cur = db.cursor()
    cur.execute("SELECT has_seen_disclaimer, has_completed_profile FROM users WHERE id=?", (session['user_id'],))
    user = cur.fetchone()
    if not user['has_seen_disclaimer']:
        return redirect(url_for('disclaimer'))
    if not user['has_completed_profile']:
        return redirect(url_for('setup_profile'))

    profile = get_user_profile(session['user_id'])

    # Thread handling
    thread_id = request.args.get('thread_id', type=int)
    cur.execute("SELECT * FROM chat_threads WHERE user_id=? ORDER BY updated_at DESC", (session['user_id'],))
    all_threads = cur.fetchall()
    all_threads_list = [dict(t) for t in all_threads]

    if not all_threads_list:
        cur.execute("INSERT INTO chat_threads (user_id, title) VALUES (?, ?)", (session['user_id'], "New Chat"))
        db.commit()
        thread_id = cur.lastrowid
        cur.execute("SELECT * FROM chat_threads WHERE user_id=? ORDER BY updated_at DESC", (session['user_id'],))
        all_threads = cur.fetchall()
        all_threads_list = [dict(t) for t in all_threads]
    else:
        if not thread_id or not any(t['id']==thread_id for t in all_threads_list):
            thread_id = all_threads_list[0]['id']

    # Migrate old history without thread_id
    try:
        cur.execute("SELECT COUNT(*) as cnt FROM chat_history WHERE user_id=? AND (thread_id IS NULL OR thread_id=0)", (session['user_id'],))
        cnt_row = cur.fetchone()
        if cnt_row and cnt_row['cnt'] > 0:
            cur.execute("UPDATE chat_history SET thread_id=? WHERE user_id=? AND (thread_id IS NULL OR thread_id=0)", (thread_id, session['user_id']))
            db.commit()
    except:
        pass

    cur.execute("SELECT id, role, message, message_type, metadata, created_at, thread_id FROM chat_history WHERE user_id=? AND thread_id=? ORDER BY created_at ASC LIMIT 200", (session['user_id'], thread_id))
    history_rows = cur.fetchall()

    history_list = []
    for row in history_rows:
        try:
            history_list.append({
                "id": row["id"],
                "role": row["role"],
                "message": row["message"],
                "message_type": row["message_type"] or "text",
                "metadata": row["metadata"],
                "created_at": row["created_at"],
                "thread_id": row["thread_id"]
            })
        except Exception as e:
            print(f"History ser error: {e}")
            continue

    kb_stats = rag_loader.get_stats()
    rag_count = kb.count
    current_thread = next((t for t in all_threads_list if t['id']==thread_id), all_threads_list[0] if all_threads_list else None)

    return render_template('chat.html', profile=profile, history=history_rows, history_json=json.dumps(history_list, ensure_ascii=False), threads=all_threads_list, current_thread_id=thread_id, current_thread=current_thread, languages=LANGUAGES, kb_stats=kb_stats, rag_count=rag_count)

@app.route('/medications')
@login_required
def medications():
    db = get_db()
    cur = db.cursor()
    cur.execute("SELECT has_seen_disclaimer, has_completed_profile FROM users WHERE id=?", (session['user_id'],))
    user = cur.fetchone()
    if not user['has_seen_disclaimer']:
        return redirect(url_for('disclaimer'))
    if not user['has_completed_profile']:
        return redirect(url_for('setup_profile'))

    profile = get_user_profile(session['user_id'])
    cur.execute("SELECT * FROM medication_schedule WHERE user_id=? AND is_active=1 ORDER BY created_at DESC", (session['user_id'],))
    schedules = cur.fetchall()

    # Parse times for display in template (ensure list)
    parsed_schedules = []
    for s in schedules:
        d = dict(s)
        try:
            times_raw = d.get('times')
            if isinstance(times_raw, str):
                d['times'] = json.loads(times_raw)
            if not isinstance(d['times'], list):
                d['times'] = [d['times']] if d['times'] else []
        except:
            d['times'] = []
        parsed_schedules.append(d)

    return render_template('medications.html', profile=profile, schedules=parsed_schedules, languages=LANGUAGES)

# API Endpoints

@app.route('/api/chat', methods=['POST'])
@login_required
def api_chat():
    data = request.get_json()
    query = data.get('query','').strip()
    thread_id = data.get('thread_id')
    if not query:
        return jsonify({"error": "Query empty"}), 400
    profile = get_user_profile(session['user_id'])
    preferred_lang = profile['preferred_language'] if profile else 'en'

    db = get_db()
    cur = db.cursor()

    # Ensure thread_id valid, else use latest
    if not thread_id:
        cur.execute("SELECT id FROM chat_threads WHERE user_id=? ORDER BY updated_at DESC LIMIT 1", (session['user_id'],))
        row = cur.fetchone()
        if row:
            thread_id = row['id']
        else:
            cur.execute("INSERT INTO chat_threads (user_id, title) VALUES (?, ?)", (session['user_id'], query[:40] or "New Chat"))
            db.commit()
            thread_id = cur.lastrowid
    else:
        # Verify thread belongs to user
        cur.execute("SELECT id FROM chat_threads WHERE id=? AND user_id=?", (thread_id, session['user_id']))
        if not cur.fetchone():
            return jsonify({"error": "Invalid thread"}), 403

    # Update thread title if first message and title is New Chat
    cur.execute("SELECT title, (SELECT COUNT(*) FROM chat_history WHERE thread_id=?) as cnt FROM chat_threads WHERE id=?", (thread_id, thread_id))
    trow = cur.fetchone()
    if trow and trow['cnt'] == 0 and trow['title'] == 'New Chat':
        new_title = query[:50]
        cur.execute("UPDATE chat_threads SET title=?, updated_at=CURRENT_TIMESTAMP WHERE id=?", (new_title, thread_id))
    else:
        cur.execute("UPDATE chat_threads SET updated_at=CURRENT_TIMESTAMP WHERE id=?", (thread_id,))

    cur.execute("INSERT INTO chat_history (user_id, role, message, message_type, thread_id) VALUES (?,?,?,?,?)",
                (session['user_id'], 'user', query, 'text', thread_id))
    db.commit()

    result = llm_handler.generate_text_response(query, preferred_lang, profile)

    cur.execute("INSERT INTO chat_history (user_id, role, message, message_type, metadata, thread_id) VALUES (?,?,?,?,?,?)",
                (session['user_id'], 'assistant', result['response'], 'text', json.dumps({"source": result.get('source',''), "rag_sources": result.get('rag_sources',[])}, ensure_ascii=False), thread_id))
    db.commit()

    return jsonify({**result, "thread_id": thread_id})

@app.route('/api/upload-prescription', methods=['POST'])
@login_required
def api_upload_prescription():
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "Empty filename"}), 400
    filename = secure_filename(file.filename)
    ext = filename.rsplit('.',1)[-1].lower()
    saved_name = f"{uuid.uuid4().hex}.{ext}"
    saved_path = os.path.join(app.config['UPLOAD_FOLDER'], saved_name)
    file.save(saved_path)

    # Get thread_id
    thread_id = request.form.get('thread_id', type=int)
    db = get_db()
    cur = db.cursor()
    if not thread_id:
        cur.execute("SELECT id FROM chat_threads WHERE user_id=? ORDER BY updated_at DESC LIMIT 1", (session['user_id'],))
        row = cur.fetchone()
        if row:
            thread_id = row['id']
        else:
            cur.execute("INSERT INTO chat_threads (user_id, title) VALUES (?, ?)", (session['user_id'], f"Prescription: {filename}"))
            db.commit()
            thread_id = cur.lastrowid
    cur.execute("UPDATE chat_threads SET updated_at=CURRENT_TIMESTAMP WHERE id=?", (thread_id,))

    profile = get_user_profile(session['user_id'])
    preferred_lang = profile['preferred_language'] if profile else 'en'

    ocr_text = ""
    if is_pdf_file(filename):
        ocr_text = extract_text_from_pdf(saved_path)

    result = llm_handler.analyze_prescription_image(saved_path, ocr_text, preferred_lang)

    cur.execute("INSERT INTO chat_history (user_id, role, message, message_type, metadata, thread_id) VALUES (?,?,?,?,?,?)",
                (session['user_id'], 'user', f"Uploaded prescription: {filename}", 'prescription_image', json.dumps({"file_path": saved_name}, ensure_ascii=False), thread_id))
    if 'medicines' in result and result['medicines']:
        cur.execute("INSERT INTO chat_history (user_id, role, message, message_type, metadata, thread_id) VALUES (?,?,?,?,?,?)",
                    (session['user_id'], 'assistant', json.dumps(result, ensure_ascii=False), 'medicine_cards', json.dumps(result, ensure_ascii=False), thread_id))
    else:
        raw = result.get('raw_response', 'Unable to parse prescription clearly. Please upload clearer image.')
        cur.execute("INSERT INTO chat_history (user_id, role, message, message_type, metadata, thread_id) VALUES (?,?,?,?,?,?)",
                    (session['user_id'], 'assistant', raw, 'text', json.dumps(result, ensure_ascii=False), thread_id))
    db.commit()

    result['thread_id'] = thread_id
    return jsonify(result)

@app.route('/api/upload-skin-image', methods=['POST'])
@login_required
def api_upload_skin_image():
    if 'file' not in request.files:
        return jsonify({"error": "No file"}), 400
    file = request.files['file']
    query = request.form.get('query','')
    thread_id = request.form.get('thread_id', type=int)
    if file.filename == '':
        return jsonify({"error": "Empty"}), 400
    filename = secure_filename(file.filename)
    ext = filename.rsplit('.',1)[-1].lower()
    saved_name = f"{uuid.uuid4().hex}.{ext}"
    saved_path = os.path.join(app.config['UPLOAD_FOLDER'], saved_name)
    file.save(saved_path)

    db = get_db()
    cur = db.cursor()
    if not thread_id:
        cur.execute("SELECT id FROM chat_threads WHERE user_id=? ORDER BY updated_at DESC LIMIT 1", (session['user_id'],))
        row = cur.fetchone()
        if row:
            thread_id = row['id']
        else:
            cur.execute("INSERT INTO chat_threads (user_id, title) VALUES (?, ?)", (session['user_id'], f"Skin: {filename}"))
            db.commit()
            thread_id = cur.lastrowid
    cur.execute("UPDATE chat_threads SET updated_at=CURRENT_TIMESTAMP WHERE id=?", (thread_id,))

    profile = get_user_profile(session['user_id'])
    preferred_lang = profile['preferred_language'] if profile else 'en'

    result = llm_handler.analyze_skin_image(saved_path, query, preferred_lang, profile)

    cur.execute("INSERT INTO chat_history (user_id, role, message, message_type, metadata, thread_id) VALUES (?,?,?,?,?,?)",
                (session['user_id'], 'user', f"Uploaded skin image: {filename} Query: {query}", 'skin_image', json.dumps({"file": saved_name}, ensure_ascii=False), thread_id))
    cur.execute("INSERT INTO chat_history (user_id, role, message, message_type, metadata, thread_id) VALUES (?,?,?,?,?,?)",
                (session['user_id'], 'assistant', result['response'], 'text', json.dumps({"source": result.get('source'), "rag_sources": result.get('rag_sources',[])}, ensure_ascii=False), thread_id))
    db.commit()

    result['thread_id'] = thread_id
    return jsonify(result)

@app.route('/api/voice-to-text', methods=['POST'])
@login_required
def api_voice_to_text():
    if 'audio' not in request.files:
        return jsonify({"error": "No audio file"}), 400
    audio_file = request.files['audio']
    filename = secure_filename(audio_file.filename or f"{uuid.uuid4().hex}.webm")
    saved_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    audio_file.save(saved_path)

    thread_id = request.form.get('thread_id', type=int)
    db = get_db()
    cur = db.cursor()
    if not thread_id:
        cur.execute("SELECT id FROM chat_threads WHERE user_id=? ORDER BY updated_at DESC LIMIT 1", (session['user_id'],))
        row = cur.fetchone()
        if row:
            thread_id = row['id']
        else:
            cur.execute("INSERT INTO chat_threads (user_id, title) VALUES (?, ?)", (session['user_id'], "Voice Chat"))
            db.commit()
            thread_id = cur.lastrowid
    cur.execute("UPDATE chat_threads SET updated_at=CURRENT_TIMESTAMP WHERE id=?", (thread_id,))

    profile = get_user_profile(session['user_id'])
    preferred_lang = profile['preferred_language'] if profile else 'en'

    transcribed_text = ""
    try:
        from faster_whisper import WhisperModel
        model = WhisperModel("tiny", device="cpu", compute_type="int8")
        segments, info = model.transcribe(saved_path, beam_size=5)
        transcribed_text = " ".join([seg.text for seg in segments])
        print(f"Transcribed: {transcribed_text} language {info.language}")
    except Exception as e:
        print(f"Whisper transcription failed: {e}")
        transcribed_text = request.form.get('mock_text', '')
        if not transcribed_text:
            return jsonify({"error": f"Voice model not loaded: {e}. Please ensure faster-whisper installed or provide text input.", "transcribed_text": ""}), 500

    if not transcribed_text:
        return jsonify({"error": "Could not transcribe"}), 400

    result = llm_handler.generate_text_response(transcribed_text, preferred_lang, profile)

    cur.execute("INSERT INTO chat_history (user_id, role, message, message_type, thread_id) VALUES (?,?,?,?,?)",
                (session['user_id'], 'user', f"🎤 {transcribed_text}", 'voice', thread_id))
    cur.execute("INSERT INTO chat_history (user_id, role, message, message_type, metadata, thread_id) VALUES (?,?,?,?,?,?)",
                (session['user_id'], 'assistant', result['response'], 'text', json.dumps({"source": result.get('source'), "transcribed": transcribed_text, "rag_sources": result.get('rag_sources',[])}, ensure_ascii=False), thread_id))
    db.commit()

    return jsonify({"transcribed_text": transcribed_text, "response": result['response'], "source": result.get('source'), "language": preferred_lang, "rag_sources": result.get('rag_sources',[]), "thread_id": thread_id})

@app.route('/api/scheduler', methods=['GET','POST','DELETE'])
@login_required
def api_scheduler():
    db = get_db()
    cur = db.cursor()
    if request.method == 'GET':
        cur.execute("SELECT * FROM medication_schedule WHERE user_id=? ORDER BY created_at DESC", (session['user_id'],))
        rows = cur.fetchall()
        result = []
        for r in rows:
            d = dict(r)
            try:
                d['times'] = json.loads(d['times']) if d['times'] else []
            except:
                d['times'] = []
            result.append(d)
        return jsonify(result)

    if request.method == 'POST':
        data = request.get_json()
        medicine_name = data.get('medicine_name','').strip()
        dosage = data.get('dosage','')
        frequency = data.get('frequency','')
        duration = data.get('duration','')
        instructions = data.get('instructions','')
        precautions = data.get('precautions','')
        times = data.get('times', [])
        start_date = data.get('start_date','')
        end_date = data.get('end_date','')

        if not medicine_name:
            return jsonify({"error": "Medicine name required"}), 400

        # Save to DB
        cur.execute("""
        INSERT INTO medication_schedule (user_id, medicine_name, dosage, frequency, duration, instructions, precautions, times, start_date, end_date)
        VALUES (?,?,?,?,?,?,?,?,?,?)
        """, (session['user_id'], medicine_name, dosage, frequency, duration, instructions, precautions, json.dumps(times, ensure_ascii=False), start_date, end_date))
        db.commit()
        new_id = cur.lastrowid

        # Immediately email medication card to profile email via SMTP
        email_result = None
        to_email = None
        try:
            profile = get_user_profile(session['user_id'])
            if profile and profile.get('email'):
                to_email = profile['email']
                user_name = profile.get('full_name') or session.get('username','User')
            else:
                cur.execute("SELECT email, username FROM users WHERE id=?", (session['user_id'],))
                urow = cur.fetchone()
                to_email = urow['email'] if urow else None
                user_name = profile['full_name'] if profile and profile.get('full_name') else (urow['username'] if urow else 'User')

            if to_email:
                medicine_card = {
                    "name": medicine_name,
                    "dosage": dosage,
                    "frequency": frequency,
                    "duration": duration,
                    "instructions": instructions,
                    "precautions": precautions
                }
                schedule_info = {
                    "times": times,
                    "start_date": start_date,
                    "end_date": end_date
                }
                email_result = send_schedule_confirmation_email(to_email, medicine_card, schedule_info, user_name)
                print(f"📧 Schedule email result to {to_email}: {email_result}")
            else:
                print("⚠️ No email found in profile, cannot send schedule confirmation")
                email_result = {"success": False, "is_mock": False, "message": "No profile email found. Add email in profile first."}
        except Exception as e:
            print(f"Failed to send immediate schedule email: {e}")
            email_result = {"success": False, "is_mock": False, "message": f"Exception: {e}", "error": str(e)}

        return jsonify({
            "success": True,
            "id": new_id,
            "email_sent_to": to_email,
            "email_result": email_result,
            "is_mock": email_result.get('is_mock') if email_result else False,
            "real_sent": email_result.get('real_sent') if email_result else False
        })

    if request.method == 'DELETE':
        try:
            data = request.get_json(silent=True) or {}
            sched_id = data.get('id')
            if not sched_id:
                sched_id = request.args.get('id') or request.form.get('id')
            if not sched_id:
                return jsonify({"error": "id required"}), 400
            cur.execute("UPDATE medication_schedule SET is_active=0 WHERE id=? AND user_id=?", (sched_id, session['user_id']))
            db.commit()
            if cur.rowcount == 0:
                return jsonify({"error": "Not found or not yours"}), 404
            print(f"🗑️ Deleted schedule {sched_id} for user {session['user_id']}")
            return jsonify({"success": True, "deleted_id": sched_id})
        except Exception as e:
            print(f"Delete error: {e}")
            return jsonify({"error": str(e)}), 500

@app.route('/api/smtp-status', methods=['GET'])
@login_required
def api_smtp_status():
    from utils.email_notifier import get_smtp_status
    status = get_smtp_status()
    profile = get_user_profile(session['user_id'])
    status['profile_email'] = profile.get('email') if profile else None
    status['profile_full_name'] = profile.get('full_name') if profile else None
    # Check if email_log exists
    log_path = os.path.join(os.path.dirname(__file__), "email_log.txt")
    if os.path.exists(log_path):
        try:
            with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                # Get last 3 emails
                status['last_log_entries'] = content[-3000:]
                status['log_exists'] = True
                status['log_path'] = log_path
        except:
            status['log_exists'] = False
    else:
        status['log_exists'] = False
    return jsonify(status)

@app.route('/api/test-email', methods=['POST'])
@login_required
def api_test_email():
    from utils.email_notifier import send_schedule_confirmation_email, get_smtp_status
    db = get_db()
    cur = db.cursor()
    profile = get_user_profile(session['user_id'])
    if not profile or not profile.get('email'):
        cur.execute("SELECT email FROM users WHERE id=?", (session['user_id'],))
        row = cur.fetchone()
        to_email = row['email'] if row else None
        user_name = session.get('username','User')
    else:
        to_email = profile['email']
        user_name = profile.get('full_name') or session.get('username','User')

    if not to_email:
        return jsonify({"error": "No profile email found. Please add email in profile."}), 400

    # Send test email with dummy medicine card
    test_card = {
        "name": "Test Medicine - SMTP Check",
        "dosage": "500mg",
        "frequency": "Once daily",
        "duration": "Test",
        "instructions": "This is a test email to verify SMTP is working",
        "precautions": "If you received this, SMTP is configured correctly!"
    }
    test_schedule = {
        "times": ["08:00"],
        "start_date": datetime.now().strftime("%Y-%m-%d"),
        "end_date": ""
    }

    result = send_schedule_confirmation_email(to_email, test_card, test_schedule, user_name)
    smtp_status = get_smtp_status()

    return jsonify({
        "test_sent_to": to_email,
        "email_result": result,
        "smtp_status": smtp_status,
        "is_mock": result.get('is_mock', False),
        "success": result.get('success', False),
        "real_sent": result.get('real_sent', False),
        "message": result.get('message','')
    })

@app.route('/api/clear-history', methods=['POST'])
@login_required
def clear_history():
    db = get_db()
    cur = db.cursor()
    # Check if thread_id provided to clear only that thread, else clear all for user
    data = request.get_json(silent=True) or {}
    thread_id = data.get('thread_id')
    if thread_id:
        cur.execute("DELETE FROM chat_history WHERE user_id=? AND thread_id=?", (session['user_id'], thread_id))
    else:
        cur.execute("DELETE FROM chat_history WHERE user_id=?", (session['user_id'],))
    db.commit()
    return jsonify({"success": True})

@app.route('/api/threads', methods=['GET', 'POST'])
@login_required
def api_threads():
    db = get_db()
    cur = db.cursor()
    if request.method == 'GET':
        cur.execute("SELECT id, title, created_at, updated_at, (SELECT COUNT(*) FROM chat_history WHERE thread_id=chat_threads.id) as msg_count FROM chat_threads WHERE user_id=? ORDER BY updated_at DESC", (session['user_id'],))
        rows = cur.fetchall()
        threads = []
        for r in rows:
            d = dict(r)
            # Truncate title for display
            if not d['title'] or d['title'].strip() == '':
                d['title'] = 'New Chat'
            threads.append(d)
        return jsonify(threads)
    elif request.method == 'POST':
        data = request.get_json(silent=True) or {}
        title = data.get('title', 'New Chat').strip()[:80] or 'New Chat'
        cur.execute("INSERT INTO chat_threads (user_id, title) VALUES (?, ?)", (session['user_id'], title))
        db.commit()
        new_id = cur.lastrowid
        cur.execute("SELECT id, title, created_at, updated_at FROM chat_threads WHERE id=?", (new_id,))
        row = cur.fetchone()
        return jsonify(dict(row) if row else {"id": new_id, "title": title})

@app.route('/api/threads/<int:thread_id>', methods=['DELETE', 'GET'])
@login_required
def api_thread_detail(thread_id):
    db = get_db()
    cur = db.cursor()
    cur.execute("SELECT * FROM chat_threads WHERE id=? AND user_id=?", (thread_id, session['user_id']))
    thread = cur.fetchone()
    if not thread:
        return jsonify({"error": "Thread not found"}), 404

    if request.method == 'GET':
        cur.execute("SELECT id, role, message, message_type, created_at FROM chat_history WHERE thread_id=? AND user_id=? ORDER BY created_at ASC", (thread_id, session['user_id']))
        messages = [dict(r) for r in cur.fetchall()]
        return jsonify({"thread": dict(thread), "messages": messages})
    elif request.method == 'DELETE':
        # Delete messages first, then thread
        cur.execute("DELETE FROM chat_history WHERE thread_id=? AND user_id=?", (thread_id, session['user_id']))
        cur.execute("DELETE FROM chat_threads WHERE id=? AND user_id=?", (thread_id, session['user_id']))
        db.commit()
        # Ensure user still has at least one thread
        cur.execute("SELECT COUNT(*) as cnt FROM chat_threads WHERE user_id=?", (session['user_id'],))
        cnt = cur.fetchone()['cnt']
        if cnt == 0:
            cur.execute("INSERT INTO chat_threads (user_id, title) VALUES (?, ?)", (session['user_id'], "New Chat"))
            db.commit()
        return jsonify({"success": True, "deleted_id": thread_id})

@app.route('/api/profile', methods=['GET'])
@login_required
def api_get_profile():
    profile = get_user_profile(session['user_id'])
    return jsonify(profile or {})

# --- RAG Knowledge Base Management Endpoints ---

@app.route('/api/kb/status', methods=['GET'])
@login_required
def api_kb_status():
    try:
        stats = rag_loader.get_stats()
        report = {
            "kb_dir": stats["kb_dir"],
            "total_files_on_disk": stats["total_files"],
            "total_size_bytes": stats["total_size_bytes"],
            "supported_exts": stats["supported_exts"],
            "files_on_disk": stats["files_list"],
            "chroma_count": kb.count,
            "ingestion_log_entries": stats["ingestion_log_entries"],
            "discovered_files": rag_loader.discover_files()
        }
        # Add sample retrieval test
        sample_query = request.args.get('q', 'fever management')
        rag_test = kb.rag_retrieve(sample_query, n_results=3)
        report["sample_retrieval"] = {
            "query": sample_query,
            "has_relevant": rag_test["has_relevant"],
            "num_chunks_found": len(rag_test["all_chunks"]),
            "sources": rag_test["sources"][:3]
        }
        return jsonify(report)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/kb/ingest', methods=['POST'])
@login_required
def api_kb_ingest():
    try:
        data = request.get_json() or {}
        force = data.get('force', False)
        report = rag_loader.ingest_directory(kb, force_reingest=force)
        return jsonify(report)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/kb/upload', methods=['POST'])
@login_required
def api_kb_upload():
    """Upload PDF/TXT/MD to knowledge_base directory and auto-ingest"""
    if 'file' not in request.files:
        return jsonify({"error": "No file"}), 400
    file = request.files['file']
    subfolder = request.form.get('subfolder', 'general').strip()
    # Sanitize subfolder
    subfolder = "".join(c for c in subfolder if c.isalnum() or c in ('_', '-', '/')).strip('/') or 'general'

    if file.filename == '':
        return jsonify({"error": "Empty filename"}), 400

    filename = secure_filename(file.filename)
    if not filename.lower().endswith(('.pdf','.txt','.md')):
        return jsonify({"error": "Only .pdf, .txt, .md allowed for KB"}), 400

    target_dir = os.path.join(app.config['KNOWLEDGE_BASE_DIR'], subfolder)
    os.makedirs(target_dir, exist_ok=True)
    save_path = os.path.join(target_dir, filename)
    file.save(save_path)

    # Auto ingest this file
    try:
        rel_path = os.path.relpath(save_path, app.config['KNOWLEDGE_BASE_DIR'])
        try:
            kb.delete_by_source(rel_path)
        except:
            pass

        chunks = rag_loader.prepare_chunks_for_ingestion(save_path)
        if chunks:
            if kb.collection:
                batch_size = 50
                for i in range(0, len(chunks), batch_size):
                    batch = chunks[i:i+batch_size]
                    ids = [c["id"] for c in batch]
                    docs = [c["content"] for c in batch]
                    metas = [c["metadata"] for c in batch]
                    try:
                        kb.collection.upsert(ids=ids, documents=docs, metadatas=metas)
                    except:
                        try:
                            kb.collection.add(ids=ids, documents=docs, metadatas=metas)
                        except Exception as e:
                            print(f"KB upload add error: {e}")
            else:
                # Fallback in-memory
                for c in chunks:
                    kb.fallback_docs.append({"id": c["id"], "content": c["content"], "metadata": c["metadata"]})

            fhash = rag_loader._file_hash(save_path)
            rag_loader.log[rel_path] = {"hash": fhash, "chunks": len(chunks), "last_ingested": str(os.path.getmtime(save_path))}
            rag_loader._save_log()

        return jsonify({"success": True, "file": rel_path, "chunks": len(chunks), "full_path": save_path})
    except Exception as e:
        return jsonify({"error": str(e), "file_saved": save_path}), 500

@app.route('/api/kb/search', methods=['POST'])
@login_required
def api_kb_search():
    """Direct RAG search for debugging"""
    data = request.get_json() or {}
    query = data.get('query','').strip()
    if not query:
        return jsonify({"error": "query required"}), 400
    n_results = data.get('n_results', 5)
    result = kb.rag_retrieve(query, n_results=n_results)
    return jsonify(result)

@app.route('/api/health')
def health_check():
    try:
        count = kb.count
    except:
        count = 0
    try:
        kb_stats = rag_loader.get_stats()
    except:
        kb_stats = {}
    return jsonify({
        "status": "ok",
        "gemini": llm_handler.model is not None,
        "kb_count": count,
        "kb_files_on_disk": kb_stats.get("total_files", 0),
        "rag_ready": True
    })

if __name__ == '__main__':
    print("Starting Care Companion on http://127.0.0.1:5000")
    print(f"Knowledge base dir: {app.config['KNOWLEDGE_BASE_DIR']} (add PDFs/TXT/MD, subdirs supported)")
    app.run(host='0.0.0.0', port=5000, debug=True)
