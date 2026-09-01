from generate import generate_linear_data
from visual import plot_linear_data, plot_non_linear_data
from generate import generate_non_linear_data
from visual import plot_non_linear_data

def main():
    x, y = generate_linear_data()
    plot_linear_data(x, y)

def main():
    x,y = generate_non_linear_data()
    plot_non_linear_data(x,y)

if __name__ == "__main__":
    main()