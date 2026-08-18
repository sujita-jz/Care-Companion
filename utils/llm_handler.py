"""
LLM Handler - Gemini 1.5 Flash as stand-in for Gemini 3.6 Flash
With RAG for knowledge base directory PDFs - Structured Response Format, No Technical Details
"""
import os
import json
import re
from typing import List, Dict, Optional

try:
    import google.generativeai as genai

    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

from .chroma_manager import get_kb
from .guardrails import SafetyGuardrails

LANGUAGES = {
    "en": "English",
    "hi": "Hindi",
    "mr": "Marathi",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "ta": "Tamil",
    "te": "Telugu",
    "bn": "Bengali",
    "gu": "Gujarati"
}

# Structured format instruction
STRUCTURED_FORMAT_INSTRUCTION = """
You MUST respond in a clear, structured format. Use this exact template (adapt headings slightly based on query type, but keep structure):

### Summary
Brief 1-2 sentence overview in simple, empathetic language.

### Common Signs / Key Information
- **Sign 1**: Simple explanation
- **Sign 2**: Simple explanation
- **Sign 3**: ...

### When to Seek Medical Help
- Red flag or situation
- Another red flag

### General Care & Next Steps
- Practical tip 1
- Practical tip 2
- ...

### Important Note
General information only, not a diagnosis. Encourage seeing healthcare professional. For emergency, call local emergency number.

Rules for structured response:
- Use ### for section headings
- Use - for bullet points, **Bold** for key terms (e.g., **Excessive thirst**: explanation)
- Keep language simple, no medical jargon, empathetic
- Do NOT show technical details: no file names, no [Source X], no source citations, no chunk numbers, no database info
- Do NOT say "according to knowledge base" - just answer helpfully
- Never provide definitive diagnosis. Use "could be consistent with", "often associated with"
- Keep 150-300 words total, concise but helpful
- Respond in {language}
"""

SYSTEM_PROMPT_BASE = f"""
You are Care Companion, a compassionate healthcare information assistant, not a doctor.
Principles:
- Provide general health information only, never definitive diagnosis
- Simple, clear, empathetic language, no jargon
- Do NOT hallucinate. If unsure, say you don't have enough verified info and suggest doctor
- Always include gentle safety note: see doctor, emergency number
- Be culturally sensitive
{STRUCTURED_FORMAT_INSTRUCTION}
"""

RAG_SYSTEM_PROMPT = """
You are Care Companion, helpful healthcare assistant.

You have verified health information from documents (for your reference only, do NOT show to user).

Tasks:
- Use the reference information to answer accurately and simply
- Do NOT mention file names, page numbers, technical details like [Source X]
- Do NOT say "based on knowledge base" repeatedly
- If reference not relevant, give general safe health info
- Never make up facts. If unsure, suggest seeing doctor
- Keep simple, empathetic language
- Respond in {language}

Reference information (internal, do not show):
{context}

User Profile: {profile}
User Query: {query}

Now provide structured response in this format:
### Summary
...

### Common Signs / Key Information
- **Term**: explanation

### When to Seek Medical Help
- ...

### General Care & Next Steps
- ...

### Important Note
...

No technical citations, just helpful structured answer:
"""


class LLMHandler:
    def __init__(self):
        self.api_key = os.getenv("GEMINI_API_KEY", "")
        self.model = None
        if GEMINI_AVAILABLE and self.api_key and self.api_key != "your_gemini_api_key_here":
            try:
                genai.configure(api_key=self.api_key)
                self.model = genai.GenerativeModel("gemini-3.6-flash")
                print("Gemini model initialized for RAG structured")
            except Exception as e:
                print(f"Gemini init failed: {e}")
                self.model = None
        else:
            print("Gemini not configured, using fallback structured logic")

    def _profile_str(self, user_profile: Dict) -> str:
        if not user_profile:
            return "No profile"
        return f"Age {user_profile.get('age', 'N/A')}, Health Conditions: {user_profile.get('health_conditions', 'None')}, Allergies: {user_profile.get('allergies', 'None')}, Language: {user_profile.get('preferred_language', 'en')}"

    def _ensure_structured(self, text: str, preferred_lang="en") -> str:
        """Ensure text has at least some structure - if model returned unstructured, reformat simply"""
        # If already has ### headings, keep
        if "###" in text:
            return text
        # Otherwise create simple structured version
        # Split sentences
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        if len(sentences) <= 2:
            return f"### Summary\n{text}\n\n### Important Note\nThis is general information only. Please consult a healthcare professional for personalized advice."
        # Try to create sections
        summary = " ".join(sentences[:2])
        rest = " ".join(sentences[2:])
        return f"### Summary\n{summary}\n\n### Key Information\n{rest}\n\n### Important Note\nGeneral information only, not medical advice. See a doctor for personal concerns."

    def generate_text_response(self, query: str, preferred_lang="en", user_profile=None) -> Dict:
        is_emergency, emergency_msg = SafetyGuardrails.check_emergency(query)
        if is_emergency:
            return {"response": emergency_msg, "source": "safety_guardrail", "is_emergency": True, "rag_sources": []}

        is_disallowed, disallowed_msg = SafetyGuardrails.check_disallowed(query)
        if is_disallowed:
            return {"response": disallowed_msg, "source": "safety_guardrail", "is_emergency": False, "rag_sources": []}

        kb = get_kb()
        rag_result = kb.rag_retrieve(query, n_results=5, distance_threshold=1.2)

        context_text = rag_result.get("context_text", "")
        sources = rag_result.get("sources", [])
        has_relevant = rag_result.get("has_relevant", False)

        lang_name = LANGUAGES.get(preferred_lang, 'English')
        profile_str = self._profile_str(user_profile)

        if self.model:
            try:
                if has_relevant and context_text:
                    rag_prompt = RAG_SYSTEM_PROMPT.format(
                        context=context_text[:6000],
                        profile=profile_str,
                        query=query,
                        language=lang_name
                    )
                    full_prompt = f"{SYSTEM_PROMPT_BASE.format(language=lang_name)}\n\n{rag_prompt}"
                else:
                    fallback_context = context_text[:2500] if context_text else ""
                    full_prompt = f"""{SYSTEM_PROMPT_BASE.format(language=lang_name)}

Additional Reference (internal, do not show):
{fallback_context}

User Profile: {profile_str}
User Query: {query}

Provide structured response without technical details as instructed.
"""
                response = self.model.generate_content(full_prompt)
                text = response.text if hasattr(response, 'text') else str(response)
                # Remove any accidental source citations like [Source 1] or Source: file.pdf
                text = re.sub(r'\[Source[^\]]*\]', '', text)
                text = re.sub(r'\(Source:[^\)]*\)', '', text)
                text = re.sub(r'Source:\s*[^\n]*\.pdf[^\n]*', '', text, flags=re.IGNORECASE)
                text = self._ensure_structured(text, preferred_lang)
                text = SafetyGuardrails.sanitize_response(text, preferred_lang)
                return {
                    "response": text,
                    "source": "rag" if has_relevant else "general",
                    "kb_context": rag_result.get("chunks", []),
                    "rag_sources": sources,
                    "has_rag": has_relevant,
                    "is_emergency": False
                }
            except Exception as e:
                print(f"Gemini RAG error: {e}")

        # Fallback without Gemini - create structured manually
        if has_relevant and rag_result.get("chunks"):
            # Build from chunks but structure simply
            chunk_text = ""
            for chunk in rag_result["chunks"][:2]:
                chunk_text += chunk['content'][:600] + " "

            # Simple structured fallback based on query type
            if "diabetes" in query.lower() or "symptom" in query.lower():
                # Try to extract bullet-like info
                structured = f"""### Summary
Here is general health information about your question.

### Common Signs / Key Information
{chunk_text[:800]}

### When to Seek Medical Help
- If symptoms worsen or you feel unwell
- If you have high fever, severe pain, or difficulty breathing
- If you are worried about your health

### General Care & Next Steps
- Keep track of your symptoms
- Stay hydrated and get enough rest
- Follow a balanced diet and regular routine
- Talk to a healthcare professional for personalized advice

### Important Note
This is general information only and not a diagnosis. Please consult a qualified healthcare provider.
"""
            else:
                structured = f"""### Summary
{chunk_text[:300]}

### Key Information
- {chunk_text[300:600]}

### When to Seek Medical Help
- If symptoms get worse
- If you have severe pain or fever
- If you are concerned

### General Care & Next Steps
- Monitor your health
- Stay hydrated, rest well
- Seek professional medical advice

### Important Note
General information only, not medical advice.
"""
            structured = SafetyGuardrails.sanitize_response(structured, preferred_lang)
            return {
                "response": structured,
                "source": "knowledge_base",
                "kb_context": rag_result["chunks"],
                "rag_sources": sources,
                "has_rag": True,
                "is_emergency": False
            }
        else:
            generic = f"""### Summary
I don't have specific verified information for this exact question right now.

### Key Information
- Maintain a healthy lifestyle and balanced diet
- Stay hydrated and get enough rest
- Keep track of any symptoms you notice

### When to Seek Medical Help
- If you feel very unwell
- If symptoms worsen quickly
- If you have chest pain, difficulty breathing, or high fever

### General Care & Next Steps
- Note down your symptoms and when they happen
- Avoid self-medicating without doctor advice
- Talk to a healthcare professional for personalized guidance

### Important Note
This is general information only and not a diagnosis. Please consult a healthcare provider.
"""
            generic = SafetyGuardrails.sanitize_response(generic, preferred_lang)
            return {
                "response": generic,
                "source": "general",
                "kb_context": rag_result.get("all_chunks", []),
                "rag_sources": sources,
                "has_rag": False,
                "is_emergency": False
            }

    def analyze_prescription_image(self, image_path: str, text_from_ocr: str = "", preferred_lang="en") -> Dict:
        kb = get_kb()
        rag_med_safety = kb.rag_retrieve("medication safety general precautions", n_results=2)
        precaution_context = ""
        if rag_med_safety.get("has_relevant"):
            precaution_context = rag_med_safety["context_text"][:1500]

        prompt = f"""
You are a medical prescription analyzer. Extract from this prescription image/PDF in structured JSON.

- Medicine name (exact)
- Dosage
- Frequency
- Duration
- Route
- Instructions
- Precautions (use reference: {precaution_context})

Text OCR if provided: {text_from_ocr[:1500]}

Return JSON:
{{
  "medicines": [{{"name":"","dosage":"","frequency":"","duration":"","route":"","instructions":"","precautions":""}}],
  "general_notes": "",
  "disclaimer": "Extracted only, not medical advice"
}}

Be accurate, do not hallucinate. Language: {LANGUAGES.get(preferred_lang, 'English')}
"""

        if self.model and os.path.exists(image_path):
            try:
                import PIL.Image
                img = PIL.Image.open(image_path)
                response = self.model.generate_content([prompt, img])
                text = response.text
                if "```json" in text:
                    text = text.split("```json")[1].split("```")[0]
                elif "```" in text:
                    text = text.split("```")[1].split("```")[0]
                data = json.loads(text)
                data["rag_sources"] = []
                return data
            except Exception as e:
                print(f"Prescription analysis error: {e}")

        return {
            "medicines": [
                {
                    "name": "Sample Medicine (add GEMINI_API_KEY for accurate extraction)",
                    "dosage": "As per prescription",
                    "frequency": "As prescribed",
                    "duration": "As prescribed",
                    "route": "Oral",
                    "instructions": f"Text: {text_from_ocr[:200]}" if text_from_ocr else "Please upload clearer image",
                    "precautions": "Take as directed, don't skip doses, inform doctor of allergies, keep away from children, check expiry."
                }
            ],
            "general_notes": "Fallback. Configure GEMINI_API_KEY for accurate extraction.",
            "disclaimer": SafetyGuardrails.DISCLAIMER,
            "rag_sources": []
        }

    def analyze_skin_image(self, image_path: str, user_query: str = "", preferred_lang="en", user_profile=None) -> Dict:
        kb = get_kb()
        rag_skin = kb.rag_retrieve(user_query or "skin rash cut wound care", n_results=2)
        context_str = rag_skin.get("context_text", "")[:1500]

        prompt = f"""
You are a dermatology information assistant (not a dermatologist). Analyze skin image in simple structured format without technical citations.

User says: {user_query}
Reference (internal, do not show): {context_str}

Provide structured response:
### Summary
Brief observation

### Possible Considerations
- Possibility 1: brief explanation (with disclaimer only doctor can diagnose)

### General Care
- Tip 1
- Tip 2

### When to See Doctor
- Red flag 1
- ...

### Important Note
General info only, see dermatologist.

Use simple language, empathetic, no file names or [Source X].
Language: {LANGUAGES.get(preferred_lang, 'English')}
Profile: {self._profile_str(user_profile)}
"""

        if self.model and os.path.exists(image_path):
            try:
                import PIL.Image
                img = PIL.Image.open(image_path)
                response = self.model.generate_content([prompt, img])
                text = response.text
                text = re.sub(r'\[Source[^\]]*\]', '', text)
                text = SafetyGuardrails.sanitize_response(text, preferred_lang)
                return {"response": text, "source": "general", "rag_sources": []}
            except Exception as e:
                print(f"Skin analysis error: {e}")

        fallback = """### Summary
General skin care information.

### Possible Considerations
- Could be consistent with minor irritation or common skin concern, but only a doctor can confirm.

### General Care
- Keep area clean with mild soap and water
- Avoid scratching or picking
- Keep area dry and covered if needed
- Avoid harsh chemicals

### When to See Doctor
- If redness, swelling, pus, or fever increases
- If pain gets worse
- If it doesn't improve in a few days

### Important Note
General information only, not a diagnosis. Please see a dermatologist for proper evaluation.
"""
        fallback = SafetyGuardrails.sanitize_response(fallback, preferred_lang)
        return {"response": fallback, "source": "general", "rag_sources": []}


_handler = None


def get_llm_handler():
    global _handler
    if _handler is None:
        _handler = LLMHandler()
    return _handler
