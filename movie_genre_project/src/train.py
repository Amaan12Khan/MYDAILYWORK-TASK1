from sklearn.model_selection import train_test_split
from src.data_loader import load_dataset
from src.preprocessing import clean_text
from src.model import GenreClassifier

print("Loading dataset...")

df = load_dataset("data/train_data.txt")

df["clean_text"] = df["description"].apply(clean_text)

X_train, X_test, y_train, y_test = train_test_split(
    df["clean_text"],
    df["genre"],
    test_size=0.2,
    random_state=42
)

classifier = GenreClassifier()

print("\nTraining Model...\n")
classifier.train(X_train, y_train)

print("\nEvaluating Model...\n")
classifier.evaluate(X_test, y_test)

classifier.save()