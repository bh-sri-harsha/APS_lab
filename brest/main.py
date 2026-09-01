from data_loader import load_breast_cancer_data
from train_test import split_and_train
from plot_distribution import plot_class_distribution
from evaluate import evaluate_thresholds

def main():
    X, y, target_names = load_breast_cancer_data()

    plot_class_distribution(y, target_names)

    X_train, X_test, y_train, y_test, model, probabilities = split_and_train(X, y)

    metrics = evaluate_thresholds(y_test, probabilities, [0.1, 0.3, 0.5, 0.7, 0.9])
    print(metrics.round(3))

    y_pred = model.predict(X_test)
    print("Model accuracy:", (y_pred == y_test).mean())

if __name__ == "__main__":
    main()