import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import matplotlib.ticker as mtick

from viz_utils import read_csv, aggregate_data, remove_constant_columns, \
    aggregate_correct_percent_by_parameters, determine_group_columns

# Set uniform font size
plt.rcParams.update({'font.size': 7 })

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
    sns.lineplot(data=data, x='room_count', y='correct_percent', hue='weights', ax=ax, palette=palette, errorbar=None)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    ax.legend(title='weights ')
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.set_ylim(0, 100)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax.xaxis.set_major_locator(mtick.MultipleLocator(1))


def merge_and_aggregate_dataframes(dataframes, merge_columns, value_column, output_file):
    """
    Merges a list of DataFrames vertically, groups by specified columns, and calculates the mean
    of the specified value column for identical groups.

    Parameters:
        dataframes (list of DataFrame): List of DataFrames to merge.
        merge_columns (list of str): List of column names to group by.
        value_column (str): The name of the column to calculate the mean for.
        output_file (str): The file path to save the resulting CSV file.

    Returns:
        DataFrame: The resulting aggregated DataFrame.
    """
    merged_df = pd.concat(dataframes, ignore_index=True)
    aggregated_df = merged_df.groupby(merge_columns, as_index=False)[value_column].mean()
    aggregated_df.to_csv(output_file, index=False)

def main():
    """
    Main function to read the CSV file, process the data, and generate multiple plots
    showing the correct percent by algorithm and room count. Also saves the relevant data to a CSV file.
    """
    file_path = "01_knn_weights.csv"
    df = read_csv(file_path)

    if df is not None:
        group_columns = determine_group_columns(df)
        aggregated_data = aggregate_data(df, group_columns)
        group_columns.extend(['room_name', 'room_count'])
        grouped_by_parameters = aggregate_correct_percent_by_parameters(aggregated_data, group_columns)
        cleaned_data = remove_constant_columns(grouped_by_parameters)

        # Filter data for each algorithm
        knn_euclidean_data = cleaned_data[cleaned_data['algorithm'] == 'knn_euclidean']
        knn_sorensen_data = cleaned_data[cleaned_data['algorithm'] == 'knn_sorensen']

        fig, axes = plt.subplots(2, 1, figsize=(12, 5))

        # Define the color palettee
        palette = sns.color_palette("viridis", 2)

        # Plot for knn_euclidean_data
        plot_correct_percent(knn_euclidean_data, 'KNN (euclidean)', axes[0], palette)

        # Plot for knn_sorensen_data
        plot_correct_percent(knn_sorensen_data, 'KNN (sørensen)', axes[1], palette)

        plt.tight_layout(rect=(0, 0, 1, 1))

        # Export the plot as image
        plt.savefig('01_knn_weights_01.png', dpi=300)
        plt.show()

        dataframes = [knn_euclidean_data, knn_sorensen_data]
        merge_columns = ['weights', 'algorithm', 'room_count']
        value_column = 'correct_percent'
        output_file = '01_knn_weights_03.csv'
        merge_and_aggregate_dataframes(dataframes, merge_columns, value_column, output_file)


if __name__ == "__main__":
    main()
