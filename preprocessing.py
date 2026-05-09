"""
preprocessing.py
Text cleaning and preprocessing for Amazon Food Review sentiment analysis.
"""

import re
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from collections import Counter

# Download required NLTK resources (run once)
def download_nltk_resources():
    for resource in ['punkt', 'stopwords', 'wordnet', 'punkt_tab']:
        nltk.download(resource, quiet=True)


def preprocess_text(text: str) -> str:
    """
    Full text preprocessing pipeline:
    - Lowercasing
    - Remove punctuation & numbers
    - Tokenization
    - Stopword removal
    - Lemmatization
    """
    if pd.isna(text):
        return ""

    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)

    tokens = word_tokenize(text)

    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]

    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]

    return ' '.join(tokens)


def handle_negations(tokens: list) -> list:
    """
    Combine negation words with the following token
    (e.g., 'not good' -> 'not_good').
    """
    negation_words = ['not', 'no', 'never', "n't"]
    new_tokens = []
    i = 0
    while i < len(tokens):
        if tokens[i] in negation_words and i + 1 < len(tokens):
            new_tokens.append(tokens[i] + '_' + tokens[i + 1])
            i += 2
        else:
            new_tokens.append(tokens[i])
            i += 1
    return new_tokens


def remove_rare_words(df: pd.DataFrame, column: str, min_count: int = 2) -> pd.DataFrame:
    """
    Remove words that appear fewer than `min_count` times across the corpus.
    """
    all_words = ' '.join(df[column]).split()
    word_freq = Counter(all_words)
    rare_words = {word for word, count in word_freq.items() if count < min_count}

    df = df.copy()
    df[column] = df[column].apply(
        lambda text: ' '.join([w for w in text.split() if w not in rare_words])
    )
    return df


def get_sentiment(score: float) -> str:
    """Map numeric review score to sentiment label."""
    if score >= 4:
        return 'Positive'
    elif score == 3:
        return 'Neutral'
    else:
        return 'Negative'


def run_preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Full preprocessing pipeline applied to a raw dataframe.
    Returns a cleaned dataframe with 'clean_review' and 'sentiment' columns.
    """
    download_nltk_resources()

    df = df.copy()
    df['sentiment'] = df['Score'].apply(get_sentiment)
    df = df[['Text', 'sentiment']]
    df = df.sample(n=min(50000, len(df)), random_state=42)

    print("Applying text preprocessing...")
    df['clean_review'] = df['Text'].apply(preprocess_text)

    print("Removing rare words...")
    df = remove_rare_words(df, 'clean_review', min_count=2)

    return df
