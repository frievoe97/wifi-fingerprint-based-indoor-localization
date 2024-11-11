import seaborn as sns
from matplotlib import pyplot as plt
import matplotlib.ticker as mtick

from viz_utils import read_csv, aggregate_data, aggregate_correct_percent_by_parameters, determine_group_columns

plt.rcParams.update({'font.size': 7})

def plot_correct_percent(data, title, ax, palette, ylabel='Correct Percent', xlabel='Number of measurements per room'):
    """
    Plots the correct percentage by room count with a line plot.

    Parameters:
        data (DataFrame): The data to plot.
        title (str): The title of the plot.
        ax (Axes): The matplotlib Axes to plot on.
        palette (list): The color palette to use.
        ylabel (str): The label for the y-axis.
    """
    sns.lineplot(data=data, x='room_count', y='correct_percent', hue='gamma_value', ax=ax, palette=palette, errorbar=None)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    ax.legend(title='gamma')
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.set_ylim(0, 100)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax.xaxis.set_major_locator(mtick.MultipleLocator(1))


def plot_avg_correct_percent(data, title, ax, palette, ylabel='Average Correct Percent'):
    """
    Plots the average correct percentage by algorithm value with a bar plot.

    Parameters:
        data (DataFrame): The data to plot.
        title (str): The title of the plot.
        ax (Axes): The matplotlib Axes to plot on.
        palette (list): The color palette to use.
        ylabel (str): The label for the y-axis.
    """
    sns.barplot(data=data, x='weights', y='correct_percent', hue='weights', ax=ax, palette=palette, dodge=False)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.set_ylim(0, data['correct_percent'].max() + 10)
    ax.set_xlabel('Algorithm Value')


def main():
    """
    Main function to read the CSV file, process the data, and generate multiple plots
    showing the correct percent by algorithm and room count.
    """
    file_path = "02_svm_gamma_value.csv"
    df = read_csv(file_path)

    if df is not None:
        group_columns = determine_group_columns(df)
        aggregated_data = aggregate_data(df, group_columns)
        group_columns.extend(['room_name', 'room_count'])
        grouped_by_parameters = aggregate_correct_percent_by_parameters(aggregated_data, group_columns)

        # Filter data
        svm_rbf_data = grouped_by_parameters[grouped_by_parameters['algorithm'] == 'svm_rbf']

        fig, axes = plt.subplots(1, 1, figsize=(12, 3))

        # Define the color palette
        palette = sns.color_palette("viridis", 2)

        # Plot for knn_euclidean_data
        plot_correct_percent(svm_rbf_data, 'SVM (RBF-Kernel)', axes, palette)

        plt.tight_layout(rect=(0, 0, 1, 1))

        # Save the plot as an image
        plt.savefig('02_svm_gamma_value_01.png', dpi=300)
        plt.show()

if __name__ == "__main__":
    main()
