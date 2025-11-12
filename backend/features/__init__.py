from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .code_generator import generate_code, generate_sql
    from .emotion_analyser import analyze_sentiment_emotion
    from .summariser import Summariser
    from .translater import detect_language, translate_to_english


def __getattr__(name):
    if name == "generate_code":
        from .code_generator import generate_code

        return generate_code
    elif name == "generate_sql":
        from .code_generator import generate_sql

        return generate_sql
    elif name == "analyze_sentiment_emotion":
        from .emotion_analyser import analyze_sentiment_emotion

        return analyze_sentiment_emotion
    elif name == "Summariser":
        from .summariser import Summariser

        return Summariser
    elif name == "detect_language":
        from .translater import detect_language

        return detect_language
    elif name == "translate_to_english":
        from .translater import translate_to_english

        return translate_to_english
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
