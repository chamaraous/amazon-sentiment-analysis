"""
inference.py
Load saved model artifacts and run sentiment predictions on new text.
"""

import re
import joblib


MODEL_PATH = '../models/trained_model.pkl'
TFIDF_PATH = '../models/tfidf_vectorizer.pkl'
LABEL_ENCODER_PATH = '../models/label_encoder.pkl'


def load_artifacts(
    model_path: str = MODEL_PATH,
    tfidf_path: str = TFIDF_PATH,
    label_encoder_path: str = LABEL_ENCODER_PATH,
):
    """Load the trained model, TF-IDF vectorizer, and label encoder from disk."""
    model = joblib.load(model_path)
    tfidf = joblib.load(tfidf_path)
    le = joblib.load(label_encoder_path)
    print("Artifacts loaded successfully.")
    return model, tfidf, le


def _preprocess(text: str) -> str:
    """Lightweight cleaning for inference (mirrors training preprocessing)."""
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    return text


def predict_sentiment(text: str, model, tfidf, le) -> str:
    """
    Predict the sentiment of a single review string.

    Parameters
    ----------
    text   : raw review text
    model  : trained sklearn classifier
    tfidf  : fitted TfidfVectorizer
    le     : fitted LabelEncoder

    Returns
    -------
    Predicted sentiment label as a string (e.g. 'Positive', 'Negative').
    """
    clean = _preprocess(text)
    vector = tfidf.transform([clean])
    prediction = model.predict(vector)
    return le.inverse_transform(prediction)[0]


def predict_batch(texts: list, model, tfidf, le) -> list:
    """Run sentiment prediction on a list of review strings."""
    cleaned = [_preprocess(t) for t in texts]
    vectors = tfidf.transform(cleaned)
    predictions = model.predict(vectors)
    return le.inverse_transform(predictions).tolist()


# ── CLI quick-test ─────────────────────────────────────────────────────────────

if __name__ == '__main__':
    model, tfidf, le = load_artifacts()

    samples = [
        "I love this product, it tastes amazing!",
        "This is absolutely terrible, would not buy again.",
        "It was okay, nothing special.",
    ]

    print("\nSample predictions:")
    for text in samples:
        label = predict_sentiment(text, model, tfidf, le)
        print(f"  [{label}] {text}")
