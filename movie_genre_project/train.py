import re
import nltk
import joblib
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

from data_loader import load_dataset

nltk.download("stopwords")
from nltk.corpus import stopwords

print("Loading dataset...")
df = load_dataset("data/train_data.txt")

stop_words = set(stopwords.words("english"))

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z]", " ", text)
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return " ".join(words)

df["clean_text"] = df["description"].apply(clean_text)

X_train, X_test, y_train, y_test = train_test_split(
    df["clean_text"], df["genre"],
    test_size=0.2,
    random_state=42
)

tfidf = TfidfVectorizer(
    max_features=15000,
    ngram_range=(1,2),
    min_df=3
)

X_train_vec = tfidf.fit_transform(X_train)
X_test_vec = tfidf.transform(X_test)

models = {
    "Logistic Regression": LogisticRegression(max_iter=3000),
    "Naive Bayes": MultinomialNB(),
    "SVM": LinearSVC()
}

best_model = None
best_acc = 0

print("\nTraining Models...\n")

for name, model in models.items():
    model.fit(X_train_vec, y_train)
    preds = model.predict(X_test_vec)
    acc = accuracy_score(y_test, preds)

    print(f"{name} Accuracy: {acc}")

    if acc > best_acc:
        best_acc = acc
        best_model = model

print("\nBest Model Accuracy:", best_acc)

joblib.dump(best_model, "models/best_model.pkl")
joblib.dump(tfidf, "models/tfidf_vectorizer.pkl")

print("Model Saved Successfully!")