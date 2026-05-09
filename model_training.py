"""
model_training.py
Training, hyperparameter tuning, and evaluation for sentiment classification.
Covers: Logistic Regression, Naive Bayes, Random Forest, and LSTM.
"""

import os
import json
import joblib
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout

# ── Constants ──────────────────────────────────────────────────────────────────
RANDOM_STATE = 42
TEST_SIZE = 0.2
RESULTS_DIR = '../outputs/evaluation_results'
MODELS_DIR = '../models'


# ── Train / test split ─────────────────────────────────────────────────────────

def split_data(X, y):
    return train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)


# ── Classical ML ──────────────────────────────────────────────────────────────

def train_logistic_regression(X_train, y_train):
    lr = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    lr.fit(X_train, y_train)
    return lr


def train_naive_bayes(X_train, y_train):
    nb = MultinomialNB()
    nb.fit(X_train, y_train)
    return nb


def train_random_forest(X_train, y_train):
    rf = RandomForestClassifier(random_state=RANDOM_STATE)
    rf.fit(X_train, y_train)
    return rf


def evaluate_model(model, X_test, y_test, name: str = 'Model') -> dict:
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    print(f"{name} Accuracy: {acc:.4f}")
    return {'name': name, 'accuracy': acc, 'report': report}


# ── Hyperparameter tuning ──────────────────────────────────────────────────────

def tune_logistic_regression(X_train, y_train):
    param_lr = {'C': [0.1, 1, 10], 'solver': ['lbfgs', 'liblinear']}
    grid = GridSearchCV(LogisticRegression(max_iter=1000), param_lr, cv=5, scoring='accuracy')
    grid.fit(X_train, y_train)
    print("Best LR Params:", grid.best_params_)
    return grid.best_estimator_


def tune_naive_bayes(X_train, y_train):
    param_nb = {'alpha': [0.1, 0.5, 1.0]}
    grid = GridSearchCV(MultinomialNB(), param_nb, cv=5, scoring='accuracy')
    grid.fit(X_train, y_train)
    print("Best NB Params:", grid.best_params_)
    return grid.best_estimator_


def tune_random_forest(X_train, y_train):
    param_rf = {'n_estimators': [100, 200], 'max_depth': [None, 10]}
    grid = GridSearchCV(RandomForestClassifier(), param_rf, cv=5, scoring='accuracy')
    grid.fit(X_train, y_train)
    print("Best RF Params:", grid.best_params_)
    return grid.best_estimator_


# ── LSTM ───────────────────────────────────────────────────────────────────────

def build_baseline_lstm(vocab_size: int = 5000, embed_dim: int = 128, seq_len: int = 100):
    model = Sequential([
        Embedding(input_dim=vocab_size, output_dim=embed_dim, input_length=seq_len),
        LSTM(64),
        Dense(1, activation='sigmoid'),
    ])
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def build_optimized_lstm(vocab_size: int = 5000, embed_dim: int = 128, seq_len: int = 100):
    model = Sequential([
        Embedding(input_dim=vocab_size, output_dim=embed_dim, input_length=seq_len),
        LSTM(128, dropout=0.2),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(1, activation='sigmoid'),
    ])
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def train_lstm(model, X_train, y_train, X_test, y_test,
               epochs: int = 5, batch_size: int = 32):
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_test, y_test),
    )
    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"LSTM Accuracy: {acc:.4f}")
    return history, acc


# ── Persistence ────────────────────────────────────────────────────────────────

def save_model(model, filename: str = 'trained_model.pkl'):
    os.makedirs(MODELS_DIR, exist_ok=True)
    path = os.path.join(MODELS_DIR, filename)
    joblib.dump(model, path)
    print(f"Model saved to {path}")


def save_results(results: list):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, 'evaluation_results.json')
    with open(path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {path}")
