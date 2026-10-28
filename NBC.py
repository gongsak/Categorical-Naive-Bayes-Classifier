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
            X_class = X.loc[y == c]
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
        self.n_undefined = 0

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
                self.n_undefined += 1
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
def main():
    # Load dataset
    # X 8124x22, contains all feature rows and columns
        # each row is one mushroom instance, each col is one of the features
    # y 1D, contains target class, 1 for poisonous, 0 for edible
        # each index is corresponding to each row in X.
    X, y = load_data()


    # split 80% training / 20% testing
    X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

    # Train model
    model = NaiveBayesClassifier().fit(X_train, y_train)

    # Training predictions
    train_prob = model.predict_prob(X_train)
    train_undefined = model.n_undefined
    train_pred = model.predict(X_train)

    # Testing predictions
    test_prob = model.predict_prob(X_test)
    test_undefined = model.n_undefined
    test_pred = model.predict(X_test)

    # Evaluate training performance
    train_results = evaluate(y_train, train_pred, train_prob)

    # Evaluate testing performance
    test_results = evaluate(y_test, test_pred, test_prob)

    print(f"Dataset: {len(X)} examples, {X.shape[1]} features")
    print(f"Training: {len(X_train)} examples, Testing: {len(X_test)} examples")
    print(f"Priors: P(edible) = {model.priors[0]:.4f}, P(poisonous) = {model.priors[1]:.4f}\n")
    
    # Display training and testing results
    print(f"{'Metric':<12}{'Training':>12}{'Testing':>12}")
    for name in train_results:
        cell = "{:>12.4f}" if isinstance(train_results[name], float) else "{:>12}"
        print(f"{name:<12}" + cell.format(train_results[name]) + cell.format(test_results[name]))

    # Report undefined posterior cases
    print(
        f"\nUndefined posteriors (both likelihoods 0, fell back to P(poisonous) = "
        f"{model.priors[1]:.4f}): training {train_undefined}, testing {test_undefined}"
    )


if __name__ == "__main__":
    main()
