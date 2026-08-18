"""
Safety guardrails inspired by August AI
"""
import re


class SafetyGuardrails:
    EMERGENCY_KEYWORDS = [
        "chest pain", "difficulty breathing", "can't breathe", "shortness of breath",
        "stroke", "heart attack", "suicidal", "suicide", "self harm", "overdose",
        "severe bleeding", "unconscious", "seizure", "anaphylaxis", "choking",
        "severe burn", "head injury", "penis", "porn"
    ]

    DISALLOWED_PATTERNS = [
        r"diagnose me definitively",
        r"give me a prescription",
        r"what exact dose should i take.*without doctor",
    ]

    # Profanity / sexual / non-medical harmful
    NSFW_KEYWORDS = ["sex", "porn", "nude", "erotic"]

    DISCLAIMER = """
🌿 Care Companion Safety Notice:
I am an AI health information assistant, not a medical professional.
Information provided is for educational purposes only and does NOT replace
professional medical advice, diagnosis, or treatment. Always consult a qualified
healthcare provider for personal medical concerns. In an emergency, call your
local emergency number (e.g., 112 in India) or go to nearest ER.
"""

    @classmethod
    def check_emergency(cls, text: str) -> tuple[bool, str]:
        lowered = text.lower()
        for kw in cls.EMERGENCY_KEYWORDS:
            if kw in lowered:
                # Distinguish true emergency vs informational
                if kw in ["suicidal", "suicide", "self harm"]:
                    return True, (
                        "🚨 It sounds like you might be in distress. You are not alone. "
                        "If you are having thoughts of self-harm, please reach out immediately:\n"
                        "- India: Kiran Helpline 1800-599-0019 or AASRA 91-9820466726\n"
                        "- Emergency: 112\n"
                        "- Talk to a trusted friend, family, or mental health professional right now.\n"
                        "If this is informational, I can provide general mental health wellness information."
                    )
                if kw in ["chest pain", "difficulty breathing", "heart attack", "stroke", "severe bleeding",
                          "unconscious"]:
                    return True, (
                        f"🚨 Potential emergency detected ('{kw}'). "
                        "If you or someone else is experiencing this RIGHT NOW, please:\n"
                        "• Call emergency services immediately (112 in India / 911 US)\n"
                        "• Do NOT wait for AI advice\n"
                        "• Go to nearest emergency department\n\n"
                        "If you are asking for general information about this symptom, I can share general educational info after you confirm you are safe and not in active emergency."
                    )
        return False, ""

    @classmethod
    def check_disallowed(cls, text: str) -> tuple[bool, str]:
        lowered = text.lower()
        if any(kw in lowered for kw in ["prescribe me", "give prescription", "exact dosage for me", "diagnose me as"]):
            return True, (
                "I cannot provide a definitive diagnosis or prescribe medication. "
                "I can share general information about conditions and help you understand prescriptions you already have. "
                "Please consult your doctor or pharmacist for personalized prescription and dosage."
            )
        return False, ""

    @classmethod
    def sanitize_response(cls, response: str, language: str = "en") -> str:
        # Ensure response contains disclaimer reminders, no definitive diagnosis language
        # Replace definitive statements
        response = re.sub(r"You have (.*?)\.",
                          r"You MAY have signs consistent with \1, but only a clinician can confirm. This is general information only.",
                          response, flags=re.IGNORECASE)
        # Add safety footer if not present
        if "not a medical professional" not in response.lower():
            response += f"\n\n---\n{cls.DISCLAIMER}\n"
        return response

    @classmethod
    def is_medical_query(cls, text: str) -> bool:
        medical_indicators = ["medicine", "drug", "symptom", "fever", "pain", "skin", "rash", "prescription", "disease",
                              "health", "doctor", "treatment", "allergy", "diabetes", "blood pressure", "headache",
                              "cough", "cold"]
        lowered = text.lower()
        return any(ind in lowered for ind in medical_indicators) or len(text.split()) > 2

    @classmethod
    def get_precaution_banner(cls) -> str:
        return """
⚠️ General Precautions:
• Follow your doctor's advice over general information
• Do not self-medicate based only on AI
• Keep medicines away from children
• Check allergies and expiry
• Seek care if worsening
"""
