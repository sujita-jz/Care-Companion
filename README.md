# Care Companion

> A multilingual, multimodal, knowledge-base-grounded healthcare information chatbot built with Flask, ChromaDB, Gemini 3.6 Flash, and Faster-Whisper.

Care Companion lets users ask healthcare-information questions through **text**, **prescription images**, or **voice**. It provides structured, simple-language responses grounded in approved local documents, persists conversations, supports medication reminders, and includes optional WhatsApp notifications.

> **Medical safety notice:** Care Companion is for general healthcare information only. It does not diagnose conditions, prescribe treatment, replace a clinician, or provide emergency care. Users must verify prescription information with a licensed clinician or pharmacist.

---

## Features

### User and safety flow

- Landing page with healthcare-focused animated UI
- Secure registration and login
- Password hashing with Werkzeug
- User profile with preferred language, WhatsApp number, and emergency contact
- Required terms/disclaimer consent before chat access

### Multimodal chat

- **Text chat:** ChromaDB RAG retrieval + Gemini-generated structured response
- **Prescription image chat:** Gemini image understanding returns extraction text and medicine cards
- **Voice chat:** Faster-Whisper transcription, language detection, translation, RAG, and translated text response
- Structured responses with Summary, relevant actions, and safety guidance
- No text-to-speech feature; voice is input only

### Knowledge base and RAG

- Recursive ingestion of `.pdf`, `.md`, and `.txt` files from `knowledge_base/`
- PDF text extraction with PyPDF
- Overlapping document chunking
- Persistent ChromaDB vector store
- Semantic similarity search using ChromaDB’s HNSW index
- Gemini instructed to use retrieved knowledge-base content only

### Saved conversations

- Persistent SQLite chat history
- Saved chats shown in the taskbar/sidebar
- Restore chat after refresh or later login
- Delete saved chats
- Read-only share links with privacy confirmation
- Saved prescription card restoration

### Medication scheduling

- Add a medicine from a prescription card to the scheduler
- Edit dosage, frequency, duration, start date, and reminder times
- Medication calendar
- APScheduler reminder dispatcher
- Optional Twilio WhatsApp schedule alerts

### Multilingual interface

- Supported profile languages:
  - English
  - Hindi
  - Marathi
  - Gujarati
  - Punjabi
  - Bengali
  - Tamil
  - Telugu
  - Kannada
  - Malayalam
  - Odia
  - Assamese
- Profile language controls AI response language
- Runtime static UI localisation through Gemini translation batches

---

## Tech stack

| Layer | Technology |
|---|---|
| Backend | Flask |
| Templates | Jinja2 |
| Frontend | HTML, CSS, Vanilla JavaScript |
| Authentication | Flask-Login + Werkzeug hashing |
| Relational database | SQLite + SQLAlchemy |
| Vector database | ChromaDB |
| PDF extraction | PyPDF |
| LLM / multimodal model | Gemini 3.6 Flash |
| Gemini interface | Interactions API / Google Generative Language API |
| Voice recognition | Faster-Whisper `large-v3` |
| Scheduling | APScheduler |
| WhatsApp alerts | Twilio |

---

## Project structure

```text
carecompanion/
├── app.py                         # Flask application and API routes
├── run.py                         # Local launcher (port 5050)
├── config.py                      # Runtime configuration
├── models.py                      # SQLAlchemy models
├── extensions.py                  # Flask extensions
├── requirements.txt               # Python dependencies
├── .env.example                   # Environment-variable template
├── knowledge_base/                # Approved PDF / MD / TXT documents
├── services/
│   ├── rag.py                     # ChromaDB ingestion and retrieval
│   ├── gemini.py                  # Gemini 3.6 integration
│   ├── language.py                # Detection and translation helpers
│   ├── voice.py                   # Faster-Whisper transcription
│   ├── notifications.py            # Twilio WhatsApp integration
│   └── scheduler.py                # Reminder dispatcher
├── templates/                     # Jinja2 pages
├── static/
│   ├── css/style.css              # Healthcare UI theme and animations
│   ├── js/app.js                  # Global/taskbar interactions
│   ├── js/chat.js                 # Text, image, and voice chat UI
│   └── js/i18n.js                 # UI localisation
├── ARCHITECTURE.md                # Earlier architecture reference
└── PROJECT_ARCHITECTURE.md        # Detailed current architecture
```

---

## Prerequisites

- Python 3.10 or later
- A Gemini API key from Google AI Studio
- FFmpeg for browser-recorded WebM voice files
- Optional: NVIDIA CUDA GPU for faster Faster-Whisper inference
- Optional: Twilio account with WhatsApp configured

### Install FFmpeg

**Ubuntu / Debian**

```bash
sudo apt update
sudo apt install ffmpeg
```

**macOS**

```bash
brew install ffmpeg
```

**Windows**

```powershell
winget install Gyan.FFmpeg
```

---

## Installation

### 1. Open the project directory

```bash
cd /home/user/carecompanion
```

### 2. Create and activate a virtual environment

```bash
python -m venv .venv
```

**macOS / Linux**

```bash
source .venv/bin/activate
```

**Windows PowerShell**

```powershell
.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Create the environment file

```bash
cp .env.example .env
```

On Windows, create a `.env` file manually or run:

```powershell
Copy-Item .env.example .env
```

### 5. Configure `.env`

```env
# Flask
SECRET_KEY=replace_with_a_long_random_secret

# Gemini
GEMINI_API_KEY=your_google_ai_studio_api_key
GEMINI_MODEL=gemini-3.6-flash

# Faster-Whisper: CPU-safe configuration
WHISPER_MODEL=large-v3
WHISPER_DEVICE=cpu
WHISPER_COMPUTE_TYPE=int8

# Optional Twilio WhatsApp configuration
TWILIO_ACCOUNT_SID=
TWILIO_AUTH_TOKEN=
TWILIO_WHATSAPP_FROM=whatsapp:+14155238886
```

### CUDA configuration

If CUDA and an NVIDIA GPU are correctly installed, use:

```env
WHISPER_MODEL=large-v3
WHISPER_DEVICE=cuda
WHISPER_COMPUTE_TYPE=float16
```

### 6. Run the project

```bash
python run.py
```

Open:

```text
http://127.0.0.1:5050
```

### 7. Verify the active build

Open:

```text
http://127.0.0.1:5050/api/health
```

A healthy instance returns JSON similar to:

```json
{
  "build": "carecompanion-rag-fallback-20260725",
  "indexed_chunks": 2,
  "vector_store": "ready"
}
```

---

## Add healthcare knowledge-base documents

Copy approved documents into:

```text
carecompanion/knowledge_base/
```

Example:

```text
knowledge_base/
├── health_basics.md
├── diabetes_guidance.pdf
├── medication_safety.pdf
└── speciality/
    └── respiratory_care.txt
```

Then restart the application:

```bash
python run.py
```

The application detects content changes, extracts readable text, chunks content, and rebuilds the ChromaDB collection.

> **Important:** scanned/image-only PDFs need OCR preprocessing before they can provide text to the RAG pipeline.

---

## RAG flow

```text
User question
    ↓
ChromaDB semantic retrieval over approved documents
    ↓
Relevant context chunks
    ↓
Gemini 3.6 Flash grounded-answer prompt
    ↓
Structured healthcare information response
```

Gemini is instructed to use only retrieved context. If the information is not available in the knowledge base, the assistant should state that it lacks enough approved information.

---

## Prescription-image flow

```text
Prescription image upload
    ↓
Gemini 3.6 Flash image understanding
    ↓
Extracted text + medicine name + dosage + frequency + duration + use case
    ↓
Verification advice and optional medication scheduling
```

Never rely on image extraction alone for medication use. Verify every detail with the original prescription and a clinician or pharmacist.

---

## Voice flow

```text
Browser microphone recording
    ↓
WebM/audio upload
    ↓
Faster-Whisper large-v3 transcription
    ↓
Language detection
    ↓
Translation to English for RAG
    ↓
ChromaDB retrieval + Gemini response
    ↓
Translation back to detected user language
    ↓
Text response saved to chat history
```

---

## Medication reminders

1. Upload a prescription image.
2. Verify the medicine card.
3. Select **Add to scheduler**.
4. Confirm/edit the medicine plan.
5. Save the schedule.
6. The reminder worker checks schedules every minute.
7. If Twilio is configured, WhatsApp alerts are sent to the configured profile number.

### Twilio note

Twilio WhatsApp delivery often requires an approved sender, recipient opt-in, and approved templates depending on the account and destination.

---

## Useful API endpoints

| Endpoint | Method | Purpose |
|---|---|---|
| `/api/health` | GET | Health/vector-store diagnostic |
| `/api/chat/text` | POST | Send text RAG question |
| `/api/chat/image` | POST | Upload prescription image |
| `/api/chat/voice` | POST | Upload recorded voice question |
| `/api/chats/<id>` | GET | Load saved chat history |
| `/api/chats/<id>` | DELETE | Delete saved chat |
| `/api/chats/<id>/share` | POST | Create/reuse read-only share URL |
| `/api/medications` | POST | Save medication schedule |
| `/api/ui/translate` | POST | Translate static UI text to profile language |

---

## Troubleshooting

### Gemini response is unavailable

Check `.env`:

```env
GEMINI_API_KEY=your_key
GEMINI_MODEL=gemini-3.6-flash
```

Restart the app after editing `.env`.

The application provides a concise knowledge-base fallback if the model is unavailable.

### Voice transcription fails

Confirm:

```bash
ffmpeg -version
```

Then verify Faster-Whisper settings in `.env`. For systems without CUDA, use:

```env
WHISPER_DEVICE=cpu
WHISPER_COMPUTE_TYPE=int8
```

### Knowledge base does not answer a question

- Confirm the document has readable text.
- Confirm the document is located under `knowledge_base/`.
- Restart the app after adding or changing files.
- Ensure the question is covered by the approved source material.

### UI language does not update

- Save the selected language in **My Profile**.
- Refresh the page with `Ctrl + Shift + R`.
- Confirm Gemini API credentials are configured; static UI falls back to English when translation is unavailable.

---

## Production recommendations

Before real-world deployment with healthcare users:

- Use HTTPS and secure session cookies.
- Replace SQLite with PostgreSQL or another managed database.
- Add CSRF protection and rate limiting.
- Add virus scanning and strict validation for file uploads.
- Replace APScheduler with a durable production worker/scheduler.
- Encrypt data and backups; redact logs.
- Add share-link expiry/revocation.
- Clinically review and version all knowledge-base content.
- Conduct privacy, security, legal, and medical-safety review for the deployment region.

---

## Documentation

For the full architecture, data model, flow diagrams, and service responsibilities, read:

```text
PROJECT_ARCHITECTURE.md
```
