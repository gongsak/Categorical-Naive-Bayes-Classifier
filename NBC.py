import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

# =========================
# Naive Bayes Classifier
# =========================

class NaiveBayesClassifier:
    def fit(self, X, y):
        self.features = list(X.columns)
        self.priors = {}
        self.feature_probabilities = {}

        # Calculate probabilities for each class
        for c in (0, 1):
            X_class = X.loc(y == c)
            n_class = len(X_class)

            if n_class == 0:
                raise ValueError(f"No examples of class {c}")

            # P(y = c)
            self.priors[c] = n_class / len(y)

            self.feature_probabilities[c] = {}
            # Calculating P(feature = value | y = c)
            for feature in self.features:
                counts = X_class[feature].value_counts()

                self.feature_probabilities[c][feature] = {
                    value: count / n_class
                    for value, count in counts.items()
                }

        return self


    def compute_likelihood(self, X):
        likelihoods = {}

        for c in (0,1):
            likelihood = 1.0

            for feature in self.features:
                value = X[feature]

                # Unseen values have probability 0
                probability = self.feature_probabilities[c][feature].get(
                    value, 0.0
                )

                likelihood *= probability
            likelihoods[c] = likelihood

        return likelihoods


    def predict_prob(self, X):
        predictions = []

        for _, row in X.iterrows():
            likelihoods = self.compute_likelihood(row)

            # Prior times likelihood
            score_0 = self.priors[0] * likelihoods[0]
            score_1 = self.priors[1] * likelihoods[1]

            denominator = score_0 + score_1

            if denominator == 0:
                # Both classes assigned zero likelihood.
                # Use prior as a documented fallback.
                p1 = self.priors[1]
            else:
                p1 = score_1 / denominator

            predictions.append(p1)

        return np.array(predictions)
        

    def predict(self, X):
        predictions = self.predict_prob(X)

        # Class 1 when posterior probabilit >= 0.5
        return (predictions >= 0.5).astype(int)


# =========================
# Evaluation Metrics
# =========================

def evaluate(y_true, y_pred, probabilities):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # Confusion Matrix
    TP = np.sum((y_true == 1) & (y_pred == 1))
    TN = np.sum((y_true == 0) & (y_pred == 0))
    FP = np.sum((y_true == 0) & (y_pred == 1))
    FN = np.sum((y_true == 1) & (y_pred == 0))

    def safe_divide(a, b):
        return a/b if b != 0 else 0.0

    # Accuracy
    accuracy = (TP + TN) / len(y_true)

    # Sensitivity (Recall) 
    sensitivity = safe_divide(TP, TP + FN)

    # Specificity
    specificity = safe_divide(TN, TN + FP)

    # Precision
    precision = safe_divide(TP, TP + FP)

    # F1 Score
    f1 = safe_divide(
        2 * precision * sensitivity,
        precision + sensitivity
    )

    # Log loss: clip for numerical stability only
    p = np.clip(probabilities, 1e-15, 1 - 1e-15)

    log_loss = -np.mean(
        y_true * np.log(p) + (1 - y_true) * np.log(1-p)
    )

    return {
        "Accuracy": accuracy,
        "Sensitivity": sensitivity,
        "Specificity": specificity,
        "F1 Score": f1,
        "Log Loss": log_loss,
        "TP": TP,
        "TN": TN,
        "FP": FP,
        "FN": FN,
    }


# =========================
# Data Loading
# =========================
#CSV header columns for the mushroom dataset 
COLUMNS = [
    "class", "cap-shape", "cap-surface", "cap-color", "bruises", "odor",
    "gill-attachment", "gill-spacing", "gill-size", "gill-color",
    "stalk-shape", "stalk-root", "stalk-surface-above-ring",
    "stalk-surface-below-ring", "stalk-color-above-ring",
    "stalk-color-below-ring", "veil-type", "veil-color", "ring-number",
    "ring-type", "spore-print-color", "population", "habitat",
]

def load_data(filename="ORIGINALagaricus-lepiota.data"):
    # Path relative to NBC.py
    # constructs path to dataset
    path = Path(__file__).resolve().parent / "mushroom" / filename
    # Read all columns as strings (Categorical data)
    df = pd.read_csv(path, header=None, dtype=str, keep_default_na=False)


    # assign columns to dataframe
    # USE ORIGINAL or else it may misinterpret the first row as header
    df.columns = COLUMNS

    # no white space removal is performed/needed for OG dataset
    # Target is 'class' if named, otherwise first column
    target = "class" if "class" in df.columns else df.columns[0]

    # Map edible = 0, poisonous = 1
    y = df[target].map({"e": 0, "p": 1})
    if y.isna().any():
        raise ValueError(f"Unexpected class labels: {set(df[target][y.isna()])}")
    y = y.astype(int)

    # All input features must be categorical
    X = df.drop(columns=target)
    #we will treat ? the missing category as its own category.

    return X, y

# =========================
# Main Program
# =========================
    # Load dataset

    # split 80% training / 20% testing

    # Train model

    # Training predictions

    # Evaluate training performance

    # Evaluate testing performance

    # Display training results

    # Display testing results

    # Report undefined posterior cases

