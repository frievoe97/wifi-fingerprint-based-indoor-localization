import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import matplotlib.ticker as mtick

from viz_utils import read_csv, aggregate_data, remove_constant_columns, \
    aggregate_correct_percent_by_parameters, determine_group_columns

# Set font size
plt.rcParams.update({'font.size': 7})

def configure_axes(axes):
    """
    Configures the axes with grid, limits, and format.

    Parameters:
        axes (array): An array of matplotlib Axes.
    """

    axes.grid(True, which='both', linestyle='--', linewidth=0.5)
    axes.set_ylim(0, 100)
    axes.yaxis.set_major_formatter(mtick.PercentFormatter())

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
    sns.barplot(data=data, x='algorithm_value', y='correct_percent', hue='algorithm_value', ax=ax, palette=palette, dodge=False)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.set_ylim(0, data['correct_percent'].max() + 10)
    ax.set_xlabel('n_estimators')

def plot_algorithm(data, ax, title):
    """
    Plots the average and weighted average correct percent for a given algorithm.

    Parameters:
        data (DataFrame): The data to plot.
        algorithm (str): The algorithm to filter the data.
        ax (Axes): The matplotlib Axes to plot on.
        title (str): The title of the plot.
    """
    palette = sns.color_palette("viridis", len(data['algorithm_value'].unique()))
    data_to_plot = data

    bar_width = 0.4

    x = range(len(data_to_plot))

    # Plot for average
    ax.bar(x, data_to_plot['correct_percent'], width=bar_width, label='Average', color=palette)
    print(data_to_plot)
    # Plot for weighted average
    ax.bar([p + bar_width for p in x], data_to_plot['weighted_correct_percent'],
           width=bar_width, label='Weighted Average', color=palette, alpha=0.6)

    ax.set_title(title)
    ax.set_ylabel('Average Correct Percent')
    ax.set_xlabel('n_estimators')
    ax.set_xticks([p + bar_width / 2 for p in x])
    ax.set_xticklabels(data_to_plot['algorithm_value'])

    add_bar_labels(ax)

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
    sns.lineplot(data=data, x='room_count', y='correct_percent', hue='algorithm_value', ax=ax, palette=palette, errorbar=None)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    ax.legend(title='n_estimators')
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.set_ylim(0, 100)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax.xaxis.set_major_locator(mtick.MultipleLocator(1))

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


def plot_avg_duration_by_algorithm_value(df):
    """
    Aggregates the data by 'algorithm_value' and plots the average duration for each algorithm value as a line plot.

    Parameters:
        df (DataFrame): The input dataframe containing 'algorithm_value' and 'duration'.
    """
    avg_duration = df.groupby('algorithm_value')['duration'].mean().reset_index()

    plt.figure(figsize=(12, 3))
    palette = sns.color_palette("viridis", len(avg_duration['algorithm_value'].unique()))

    sns.lineplot(data=avg_duration, x='algorithm_value', y='duration', palette=palette, marker='o')

    plt.title('Average Duration by n_estimators')
    plt.xlabel('Number of n_estimators')
    plt.ylabel('Average Duration [s]')
    plt.xticks(rotation=45)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)

    plt.tight_layout()

    plt.savefig('04_random_forest_n_estimators_04.png', dpi=300)

def main():
    """
    Main function to read the CSV file, process the data, and generate multiple plots
    showing the correct percent by algorithm and room count.
    """
    file_path = "04_random_forest_n_estimators.csv"
    df = read_csv(file_path)

    if df is not None:
        plot_avg_duration_by_algorithm_value(df)

        group_columns = determine_group_columns(df)

        df["algorithm_value"] = df["algorithm_value"].fillna('None')

        aggregated_data = aggregate_data(df, group_columns)
        group_columns.extend(['room_name', 'room_count'])
        grouped_by_parameters = aggregate_correct_percent_by_parameters(aggregated_data, group_columns)
        cleaned_data = remove_constant_columns(grouped_by_parameters)

        # Weighted average
        cleaned_data['weighted_correct_percent'] = cleaned_data['correct_percent'] * cleaned_data['room_count']
        weighted_avg_correct_percent = cleaned_data.groupby(['algorithm_value']).apply(
            lambda x: pd.Series({'weighted_correct_percent': x['weighted_correct_percent'].sum() / x[
                'room_count'].sum()})).reset_index()

        fig, axes = plt.subplots(1, 1, figsize=(14, 6))

        palette = sns.color_palette("viridis", 19)

        plot_correct_percent(cleaned_data, 'SVM (RBF)', axes, palette)


        plt.tight_layout(rect=(0, 0, 1, 1))

        plt.savefig('04_random_forest_n_estimators_01.png', dpi=300)
        plt.show()

        avg_correct_percent = cleaned_data.groupby(['algorithm_value'])['correct_percent'].mean().reset_index()

        fig, axes = plt.subplots(1, 1, figsize=(18, 12))
        fig.suptitle('Average Correct Percent by Algorithm and Algorithm Value', fontsize=16)

        knn_euclidean_palette = sns.color_palette("viridis", len(
            avg_correct_percent['algorithm_value'].unique()))
        plot_avg_correct_percent(avg_correct_percent,
                                 'KNN (euclidean)', axes, knn_euclidean_palette)

        avg_correct_percent = avg_correct_percent.merge(weighted_avg_correct_percent,
                                                        on=['algorithm_value'])



        plt.tight_layout(rect=(0, 0, 1, 0.95))
        plt.savefig('04_random_forest_n_estimators_02.png', dpi=300)
        plt.show()

        fig, axes = plt.subplots(1, 1, figsize=(12, 4))

        configure_axes(axes)

        plot_algorithm(avg_correct_percent, axes, 'Random Forest')

        legend_patches = [
            plt.Line2D([0], [0], color='grey', lw=4, label='average'),
            plt.Line2D([0], [0], color='lightgrey', lw=4, label='weighted a verage')
        ]

        fig.legend(handles=legend_patches, loc='lower center', ncol=2, bbox_to_anchor=(0.14, -0.01))

        plt.tight_layout(rect=(0, 0, 1, 0.99))
        plt.savefig('04_random_forest_n_estimators_03.png', dpi=300, bbox_inches='tight')
        plt.show()

if __name__ == "__main__":
    main()
