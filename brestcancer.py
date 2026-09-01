import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression

data = load_breast_cancer()
X=pd.DataFrame(
    data.data,
    columns=data.feature_names
)

y=pd.Series(
    (data.target == 0).astype(int),
    name="malignant"
)

print(y.value_counts())

print("Feature matrix Shape:", X.shape)
print("Target Shape:", y.shape)
print("Class name:", data.target_names)

class_counts = y.value_counts().sort_index()
class_distribution = pd.DataFrame({
    "Class": data.target_names,
    "Count": class_counts.values,
    "Probability": class_counts.values / len(y)
})

print(class_distribution)
class_distribution.plot(
    x="Class",
    y="Count",
    kind="bar",
    legend=False,
    color=["tomato", "steelblue"]
)

plt.ylabel("Number of observations")
plt.title("Class Distribution")
plt.xticks(rotation=0)
plt.show()

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training size:", len(y_train))
print("Testing size:", len(y_test))

print("\n Training proportions:")
print(y_train.value_counts(normalize=True).sort_index())

print("\n Testing proportions:")
print(y_test.value_counts(normalize=True).sort_index())

model=make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=1000)
)

model.fit(X_train,y_train)

probabilities = model.predict_proba(X_test)

print(probabilities[:5])


results = pd.DataFrame({"Actual_class":y_test.values,
                        "P_malignant":probabilities[:,0],
                        "P_benign":probabilities[:,1]
                        })

results["Actual_label"] = results["Actual_class"].map({0:"malignant", 1:"benign"})

print(results[["Actual_label","P_malignant","P_benign"]].head(10))

threshold = 0.5

results["Predcited_malignant"] = (results["P_malignant"] >= threshold).astype(int)

results["Predicted_label"] = results["Predcited_malignant"].map({1:"malignant", 0:"benign"})

print(results[["Actual_label","Predicted_label","P_malignant","P_benign"]].head(10))

for threshold in [0.3, 0.5, 0.7]:
    predictions = (results["P_malignant"] >= threshold).astype(int)

    print(f"\nThreshold: {threshold}"
          f"Predicted malignant cases: {predictions.sum()}"
    )

actual_malignant = (y_test.values == 0).astype(int)

for threshold in [0.3, 0.5, 0.7]:
    predicted_malignant = (probabilities[:,0] >= threshold).astype(int)

    cm = confusion_matrix(actual_malignant, predicted_malignant)
    print(f"\nThreshold: {threshold}")
    print(f"Confusion Matrix:")
    print(cm)

