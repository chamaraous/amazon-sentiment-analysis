# Amazon Food Review — Sentiment Analysis

Multi-class sentiment classification (Positive / Neutral / Negative) on the
[Amazon Fine Food Reviews](https://www.kaggle.com/datasets/satyabrat35/amazon-food-review-dataset)
dataset using classical ML models and an LSTM neural network.

---

## Project Structure

```
project/
├── data/                          # Raw and processed data files
├── notebooks/
│   └── eda_and_modeling.ipynb     # Full EDA + modelling walkthrough
├── src/
│   ├── preprocessing.py           # Text cleaning, tokenisation, lemmatisation
│   ├── feature_engineering.py     # BoW, TF-IDF, Word2Vec, sequence padding
│   ├── model_training.py          # Train / tune / evaluate all models
│   └── inference.py               # Load saved model and predict on new text
├── models/
│   └── trained_model.pkl          # Best saved classifier
├── outputs/
│   └── evaluation_results/        # JSON evaluation reports
├── requirements.txt
└── README.md
```

---

## Setup

```bash
pip install -r requirements.txt
```

NLTK resources are downloaded automatically on first run.

---

## Usage

### 1 — Data download (inside the notebook or a script)

```python
import opendatasets as od
od.download('https://www.kaggle.com/datasets/satyabrat35/amazon-food-review-dataset')
```

### 2 — Preprocessing

```python
from src.preprocessing import run_preprocessing
import pandas as pd

raw_df = pd.read_csv('data/finefoods.csv')   # adjust to your parsed source
df = run_preprocessing(raw_df)
```

### 3 — Feature engineering

```python
from src.feature_engineering import build_tfidf, encode_labels, save_feature_artifacts

X, tfidf = build_tfidf(df)
y, le    = encode_labels(df)
save_feature_artifacts(tfidf, le)
```

### 4 — Training & evaluation

```python
from src.model_training import *

X_train, X_test, y_train, y_test = split_data(X, y)

lr      = train_logistic_regression(X_train, y_train)
best_lr = tune_logistic_regression(X_train, y_train)
results = evaluate_model(best_lr, X_test, y_test, name='Logistic Regression (tuned)')

save_model(best_lr)
save_results([results])
```

### 5 — Inference

```python
from src.inference import load_artifacts, predict_sentiment

model, tfidf, le = load_artifacts()
print(predict_sentiment("Absolutely love this product!", model, tfidf, le))
```

Or run directly:

```bash
python src/inference.py
```

---

## Models

| Model               | Feature  | Notes                         |
|---------------------|----------|-------------------------------|
| Logistic Regression | TF-IDF   | Tuned with GridSearchCV       |
| Naive Bayes         | TF-IDF   | Tuned with GridSearchCV       |
| Random Forest       | TF-IDF   | Tuned with GridSearchCV       |
| LSTM (baseline)     | Sequences| 1 LSTM layer, 5 epochs        |
| LSTM (optimised)    | Sequences| Dropout + dense head, 10 epochs|
