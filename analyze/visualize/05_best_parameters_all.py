import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import matplotlib.ticker as mtick

from viz_utils import read_csv, aggregate_data, remove_constant_columns, \
    aggregate_correct_percent_by_parameters, determine_group_columns

# Set font size
plt.rcParams.update({'font.size': 7})

def create_lineplot(ax, data, title, ylabel, hue_order, palette, legendTitle):
    """
    Creates a line plot on the given axes.

    Parameters:
        ax (Axes): The matplotlib Axes to plot on.
        data (DataFrame): The data to plot.
        title (str): The title of the plot.
        ylabel (str): The label for the y-axis.
        hue_order (list): The order of hues.
        palette (list): The color palette to use.
    """
    sns.lineplot(
        data=data,
        x='room_count', y='correct_percent',
        hue='algorithm_value', hue_order=hue_order,
        ax=ax, palette=palette, errorbar=None
    )
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.legend(title=legendTitle, bbox_to_anchor=(1, 1), loc='upper left')
    ax.xaxis.set_major_locator(mtick.MultipleLocator(1))

def configure_axes(axes):
    """
    Configures the axes with grid, limits, and format.

    Parameters:
        axes (array): An array of matplotlib Axes.
    """
    for ax in axes.flat:
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        ax.set_ylim(0, 100)
        ax.yaxis.set_major_formatter(mtick.PercentFormatter())


def add_bar_labels(ax):
    """
    Adds labels on top of the bars in a bar plot.

    Parameters:
        ax (Axes): The matplotlib Axes to plot on.
    """
    for p in ax.patches:
        value = format(p.get_height(), '.1f')
        if value != "0.0":
            ax.annotate(value,
                        (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center', va='center',
                        xytext=(0, 9),
                        textcoords='offset points')

def plot_algorithm(data, algorithm, ax, title, xAxisLabel):
    """
    Plots the average and weighted average correct percent for a given algorithm.

    Parameters:
        data (DataFrame): The data to plot.
        algorithm (str): The algorithm to filter the data.
        ax (Axes): The matplotlib Axes to plot on.
        title (str): The title of the plot.
    """
    palette = sns.color_palette("viridis", len(data[data['algorithm'] == algorithm]['algorithm_value'].unique()))
    data_to_plot = data[data['algorithm'] == algorithm]

    bar_width = 0.4
    x = range(len(data_to_plot))

    # Plot for average
    ax.bar(x, data_to_plot['correct_percent'], width=bar_width, label='Average', color=palette)
    # Plot for weighted average
    ax.bar([p + bar_width for p in x], data_to_plot['weighted_correct_percent'],
           width=bar_width, label='Weighted Average', color=palette, alpha=0.6)

    ax.set_title(title)
    ax.set_ylabel('Average Correct Percent')
    ax.set_xlabel(xAxisLabel)
    ax.set_xticks([p + bar_width / 2 for p in x])
    ax.set_xticklabels(data_to_plot['algorithm_value'])

    add_bar_labels(ax)

def main():
    """
    Main function to read the CSV file, process the data, and generate multiple plots
    showing the correct percent by algorithm and room count.
    """
    file_path = "05_best_parameters_all.csv"
    # file_path = folder_path + "2_best_parameters_merged.csv"
    df = read_csv(file_path)
    if df is not None:
        group_columns = determine_group_columns(df)
        aggregated_data = aggregate_data(df, group_columns)
        group_columns.extend(['room_name', 'room_count'])
        grouped_by_parameters = aggregate_correct_percent_by_parameters(aggregated_data, group_columns)
        cleaned_data = remove_constant_columns(grouped_by_parameters)

        knn_euclidean_data = cleaned_data[cleaned_data['algorithm'] == 'knn_euclidean']
        knn_sorensen_data = cleaned_data[cleaned_data['algorithm'] == 'knn_sorensen']
        svm_linear_data = cleaned_data[cleaned_data['algorithm'] == 'svm_linear']
        svm_rbf_data = cleaned_data[cleaned_data['algorithm'] == 'svm_rbf']

        fig, axes = plt.subplots(4, 1, figsize=(12, 10))

        configure_axes(axes)

        palette = sns.color_palette("viridis", 8)

        create_lineplot(axes[0], knn_euclidean_data, 'KNN (euclidean)', 'Correct Percent', None, palette, 'k_value')
        create_lineplot(axes[1], knn_sorensen_data, 'KNN (sørensen)', 'Correct Percent', None, palette, 'k_value')
        create_lineplot(axes[2], svm_linear_data, 'SVM (linear)', 'Correct Percent', None, palette, 'c_value')
        create_lineplot(axes[3], svm_rbf_data, 'SVM (rbf)', 'Correct Percent', None, palette, 'c_value')

        for ax in axes:
            ax.set_xlabel('Room Count')

        plt.tight_layout(rect=(0, 0, 1, 1))

        plt.savefig('05_best_parameters_all_01.png', dpi=300)
        plt.show()

        avg_correct_percent = cleaned_data.groupby(['algorithm', 'algorithm_value'])[
            'correct_percent'].mean().reset_index()

        cleaned_data['weighted_correct_percent'] = cleaned_data['correct_percent'] * cleaned_data['room_count']
        weighted_avg_correct_percent = cleaned_data.groupby(['algorithm', 'algorithm_value']).apply(
            lambda x: pd.Series({'weighted_correct_percent': x['weighted_correct_percent'].sum() / x[
                'room_count'].sum()})).reset_index()

        avg_correct_percent = avg_correct_percent.merge(weighted_avg_correct_percent,
                                                        on=['algorithm', 'algorithm_value'])

        avg_correct_percent.to_csv('05_best_parameters_all_02.csv', index=False)

        fig, axes = plt.subplots(4, 1, figsize=(12, 10))

        configure_axes(axes)

        plot_algorithm(avg_correct_percent, 'knn_euclidean', axes[0], 'KNN (euclidean)', 'k_value')
        plot_algorithm(avg_correct_percent, 'knn_sorensen', axes[1], 'KNN (sørensen)', 'k_value')
        plot_algorithm(avg_correct_percent, 'svm_linear', axes[2], 'SVM (linear)', 'c_value')
        plot_algorithm(avg_correct_percent, 'svm_rbf', axes[3], 'SVM (RBF-Kernel)', 'c_value')

        legend_patches = [
            plt.Line2D([0], [0], color='grey', lw=4, label='average'),
            plt.Line2D([0], [0], color='lightgrey', lw=4, label='weighted average')
        ]

        fig.legend(handles=legend_patches, loc='lower center', ncol=2, bbox_to_anchor=(0.14, -0.01))

        plt.tight_layout(rect=(0, 0, 1, 0.99))
        plt.savefig('05_best_parameters_all_03.png', dpi=300, bbox_inches='tight')
        plt.show()

if __name__ == "__main__":
    main()
