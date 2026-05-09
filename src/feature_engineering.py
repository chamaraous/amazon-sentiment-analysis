"""
feature_engineering.py
Feature extraction for Amazon Food Review sentiment analysis.
Supports Bag-of-Words, TF-IDF, and Word2Vec representations.
"""

import numpy as np
import pandas as pd
import joblib
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from gensim.models import Word2Vec
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


MAX_FEATURES = 5000
W2V_VECTOR_SIZE = 100
W2V_WINDOW = 5
W2V_MIN_COUNT = 2
MAX_SEQUENCE_LENGTH = 100


def build_bow(df: pd.DataFrame, text_col: str = 'clean_review'):
    """Bag-of-Words features using CountVectorizer."""
    bow = CountVectorizer(max_features=MAX_FEATURES)
    X_bow = bow.fit_transform(df[text_col])
    print(f"BoW shape: {X_bow.shape}")
    return X_bow, bow


def build_tfidf(df: pd.DataFrame, text_col: str = 'clean_review'):
    """TF-IDF features."""
    tfidf = TfidfVectorizer(max_features=MAX_FEATURES)
    X_tfidf = tfidf.fit_transform(df[text_col])
    print(f"TF-IDF shape: {X_tfidf.shape}")
    return X_tfidf, tfidf


def build_word2vec(df: pd.DataFrame, text_col: str = 'clean_review'):
    """Train a Word2Vec model and return per-review averaged vectors."""
    tokenized = df[text_col].apply(lambda x: x.split())
    model_w2v = Word2Vec(
        sentences=tokenized,
        vector_size=W2V_VECTOR_SIZE,
        window=W2V_WINDOW,
        min_count=W2V_MIN_COUNT,
    )

    def avg_vector(tokens):
        vecs = [model_w2v.wv[w] for w in tokens if w in model_w2v.wv]
        return np.mean(vecs, axis=0) if vecs else np.zeros(W2V_VECTOR_SIZE)

    X_w2v = np.vstack(tokenized.apply(avg_vector))
    print(f"Word2Vec shape: {X_w2v.shape}")
    return X_w2v, model_w2v


def build_sequences(df: pd.DataFrame, text_col: str = 'clean_review'):
    """
    Tokenize and pad sequences for LSTM input.
    Returns padded sequences and the fitted Keras Tokenizer.
    """
    tokenizer = Tokenizer(num_words=MAX_FEATURES)
    tokenizer.fit_on_texts(df[text_col])
    sequences = tokenizer.texts_to_sequences(df[text_col])
    X_seq = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
    print(f"Sequence shape: {X_seq.shape}")
    return X_seq, tokenizer


def encode_labels(df: pd.DataFrame, label_col: str = 'sentiment'):
    """Encode string sentiment labels to integers."""
    le = LabelEncoder()
    y = le.fit_transform(df[label_col])
    print(f"Classes: {le.classes_}")
    return y, le


def save_feature_artifacts(tfidf, le, output_dir: str = '../models'):
    """Persist vectorizer and label encoder for use in inference."""
    import os
    os.makedirs(output_dir, exist_ok=True)
    joblib.dump(tfidf, f'{output_dir}/tfidf_vectorizer.pkl')
    joblib.dump(le, f'{output_dir}/label_encoder.pkl')
    print(f"Feature artifacts saved to {output_dir}/")
