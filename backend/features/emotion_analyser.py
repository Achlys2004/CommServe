from transformers import pipeline
from functools import lru_cache


@lru_cache(maxsize=1)
def get_sentiment_pipeline():
    return pipeline("sentiment-analysis")  # type: ignore


@lru_cache(maxsize=1)
def get_emotion_pipeline():
    return pipeline(  # type: ignore
        "text-classification",
        model="j-hartmann/emotion-english-distilroberta-base",
        return_all_scores=False,
    )


@lru_cache(maxsize=1000)
def analyze_sentiment_emotion(text):
    try:
        truncated_text = text[:1028] if text else ""
        if not truncated_text.strip():
            return {
                "sentiment": "NEUTRAL",
                "sentiment_score": 0.0,
                "emotion": "neutral",
                "emotion_score": 0.0,
            }

        sentiment_analyzer = get_sentiment_pipeline()
        emotion_analyzer = get_emotion_pipeline()

        sentiment_res = sentiment_analyzer(truncated_text)[0]
        emotion_res = emotion_analyzer(truncated_text)[0]

        return {
            "sentiment": sentiment_res.get("label", "UNKNOWN"),
            "sentiment_score": sentiment_res.get("score", 0.0),
            "emotion": emotion_res.get("label", "neutral"),
            "emotion_score": emotion_res.get("score", 0.0),
        }
    except Exception as e:
        return {
            "sentiment": "ERROR",
            "sentiment_score": 0.0,
            "emotion": "neutral",
            "emotion_score": 0.0,
            "error": str(e),
        }
