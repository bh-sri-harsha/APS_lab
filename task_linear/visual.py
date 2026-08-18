import matplotlib.pyplot as plt

def plot_linear_data(x, y):
    plt.scatter(x[y == 0][:, 0], x[y == 0][:, 1], color='blue', label='Class 0')
    plt.scatter(x[y == 1][:, 0], x[y == 1][:, 1], color='red', label='Class 1')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Linear Data Visualization')
    plt.legend()
    plt.show()