"""Compare baseline classifiers on Seaborn's Titanic dataset."""

import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC


def main():
    # Load and preprocess the same features used in the notebook.
    df = sns.load_dataset("titanic")
    df = df.drop(columns=["deck", "embark_town", "alive", "class", "who", "adult_male"])
    df["age"] = df["age"].fillna(df["age"].mean())
    df = df.dropna(subset=["embarked"]).copy()

    for column in ("sex", "embarked"):
        df[column] = LabelEncoder().fit_transform(df[column])
    df = df.astype(int)

    X = df.drop(columns="survived")
    y = df["survived"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Fit scaling on training data only, then reuse it for test data.
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {
        "Logistic Regression": (LogisticRegression(), X_train, X_test),
        "K-Nearest Neighbors (k=5)": (KNeighborsClassifier(n_neighbors=5), X_train_scaled, X_test_scaled),
        "Gaussian Naive Bayes": (GaussianNB(), X_train, X_test),
        "Decision Tree": (DecisionTreeClassifier(random_state=42), X_train_scaled, X_test_scaled),
        "Support Vector Machine (RBF)": (SVC(kernel="rbf"), X_train_scaled, X_test_scaled),
    }

    results = []
    for name, (model, train_features, test_features) in models.items():
        model.fit(train_features, y_train)
        predictions = model.predict(test_features)
        accuracy = accuracy_score(y_test, predictions)
        results.append({"Model": name, "Accuracy": accuracy})
        print(f"\n{name}: accuracy = {accuracy:.4f}")
        print("Confusion matrix:")
        print(confusion_matrix(y_test, predictions))
        print("Classification report:")
        print(classification_report(y_test, predictions, zero_division=0))

    print("\nAccuracy summary:")
    print(pd.DataFrame(results).sort_values("Accuracy", ascending=False).to_string(index=False))


if __name__ == "__main__":
    main()
