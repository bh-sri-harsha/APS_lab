import matplotlib.pyplot as plt
import pandas as pd

def plot_class_distribution(y, target_names):
    class_counts = y.value_counts().sort_index()

    class_distribution = pd.DataFrame({
        "Class": [target_names[i] for i in class_counts.index],
        "Count": class_counts.values
    })

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