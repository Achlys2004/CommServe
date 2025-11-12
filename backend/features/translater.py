from langdetect import detect
from deep_translator import GoogleTranslator
from functools import lru_cache


@lru_cache(maxsize=1000)
def detect_language(text):
    try:
        if not text or not text.strip():
            return "en"
        return detect(text)
    except Exception:
        return "en"


@lru_cache(maxsize=500)
def translate_to_english(text):
    try:
        if not text or not text.strip():
            return text

        lang = detect_language(text)
        if lang == "en":
            return text

        translated = GoogleTranslator(source=lang, target="en").translate(text)
        return translated
    except Exception as e:
        print(f"[Translation Error] {e}")
        return text  # fallback to original text
