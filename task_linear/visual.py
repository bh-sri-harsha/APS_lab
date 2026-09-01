import matplotlib.pyplot as plt

def plot_linear_data(x, y,title='Linear Data Visualization'):
    plt.scatter(x[:,0], x[:,1], c=y)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Linear Data Visualization')
    plt.legend()
    plt.show()

def plot_non_linear_data(x, y,title='Non-Linear Data Visualization'):
    plt.scatter(x[:,0], x[:,1], c=y)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Non-Linear Data Visualization')
    plt.legend()
    plt.show()