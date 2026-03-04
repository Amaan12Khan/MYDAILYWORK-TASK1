from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
import joblib

class GenreClassifier:

    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            max_features=15000,
            ngram_range=(1,2),
            min_df=3
        )

        self.models = {
    "Logistic Regression": LogisticRegression(max_iter=3000),
    "Naive Bayes": MultinomialNB()
        }

        self.best_model = None

    def train(self, X_train, y_train):
        X_train_vec = self.vectorizer.fit_transform(X_train)

        best_acc = 0

        for name, model in self.models.items():
            model.fit(X_train_vec, y_train)
            acc = model.score(X_train_vec, y_train)

            print(f"{name} Training Accuracy: {acc}")

            if acc > best_acc:
                best_acc = acc
                self.best_model = model

        print("\nBest Model Selected.")

    def evaluate(self, X_test, y_test):
        X_test_vec = self.vectorizer.transform(X_test)
        preds = self.best_model.predict(X_test_vec)
        acc = accuracy_score(y_test, preds)
        print("Test Accuracy:", acc)

    def save(self):
        joblib.dump(self.best_model, "models/best_model.pkl")
        joblib.dump(self.vectorizer, "models/tfidf_vectorizer.pkl")
        print("Model Saved Successfully.")