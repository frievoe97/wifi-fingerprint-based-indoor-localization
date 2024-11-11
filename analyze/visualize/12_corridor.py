import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from viz_utils import read_csv, aggregate_data, aggregate_correct_percent_by_parameters, determine_group_columns

plt.rcParams.update({'font.size': 7})

def plot_heatmap(data, title, ax, cbar=False):
    """
    Plots a heatmap using the provided data.

    Parameters:
        data (DataFrame): The data to plot.
        title (str): The title of the heatmap.
        ax (Axes): The matplotlib Axes to plot on.
        cbar (bool): Whether to include a color bar. Default is False.
    """
    sns.heatmap(data, annot=True, fmt=".1f", cmap='viridis', ax=ax, vmin=0, vmax=100, cbar=cbar,
                annot_kws={"size": 6}, cbar_kws={'format': '%.0f%%'})
    ax.set_title(title)
    ax.set_xlabel('Measurements per Room')
    ax.set_ylabel('Measurements per Corridor')

    for text in ax.texts:
        text.set_text(f'{text.get_text()}%')


def main():
    """
    Main function to read the CSV file, process the data, and generate heatmaps
    showing the true percent, not false percent, and combined true + not false percent
    by algorithm and parameters.
    """
    file_path = "test_corrdior_3.csv"
    df = read_csv(file_path)
    if df is not None:

        group_columns = determine_group_columns(df)
        aggregated_data = aggregate_data(df, group_columns, new_pattern=True)
        grouped_by_parameters = aggregate_correct_percent_by_parameters(aggregated_data, group_columns,
                                                                        new_pattern=True)

        grouped_by_parameters = grouped_by_parameters.drop(
            ['handle_missing_values_strategy', 'router_selection', "router_presence_threshold", "router_rssi_threshold",
             "value_scaling_strategy", "weights"],
            axis=1
        )

        filtered_data = grouped_by_parameters[grouped_by_parameters['algorithm'] == 'knn_sorensen']
        algorithm_value = filtered_data['algorithm_value'].iloc[0]

        heatmap_data_true = filtered_data.pivot_table(
            index='measurements_per_corridor',
            columns='measurements_per_room',
            values='True_percent'
        ).sort_index(ascending=False).sort_index(axis=1, ascending=True)

        heatmap_data_combined = (filtered_data.pivot_table(
            index='measurements_per_corridor',
            columns='measurements_per_room',
            values='True_percent'
        ).sort_index(ascending=False).sort_index(axis=1, ascending=True) +
                                 filtered_data.pivot_table(
                                     index='measurements_per_corridor',
                                     columns='measurements_per_room',
                                     values='Not_False_percent'
                                 ).sort_index(ascending=False).sort_index(axis=1, ascending=True))

        heatmap_data_not_false = filtered_data.pivot_table(
            index='measurements_per_corridor',
            columns='measurements_per_room',
            values='Not_False_percent'
        ).sort_index(ascending=False).sort_index(axis=1, ascending=True)

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))

        plot_heatmap(heatmap_data_true, f'True Percent',
                     axes[0])
        plot_heatmap(heatmap_data_combined,
                     f'True + Not False Percent', axes[1],
                     cbar=False)
        plot_heatmap(heatmap_data_not_false, f'Not False Percent',
                     axes[2], cbar=True)

        cbar = axes[2].collections[0].colorbar
        cbar.set_ticks([0, 20, 40, 60, 80, 100])
        cbar.set_ticklabels(['0%', '20%', '40%', '60%', '80%', '100%'])

        plt.tight_layout()
        plt.savefig('12_corridor_01.png', dpi=300)
        plt.show()


if __name__ == "__main__":
    main()
