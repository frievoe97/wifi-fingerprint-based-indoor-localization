import seaborn as sns
from matplotlib import pyplot as plt
import matplotlib.ticker as mtick

from viz_utils import read_csv, aggregate_data, remove_constant_columns, \
    aggregate_correct_percent_by_parameters, determine_group_columns

# Set font size
plt.rcParams.update({'font.size': 12})

def configure_axes(axes):
    """
    Configures the axes with grid, limits, and format.
    """
    for ax in axes.flat:
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        ax.set_ylim(0, 100)
        ax.yaxis.set_major_formatter(mtick.PercentFormatter())

def add_bar_labels(ax):
    """
    Adds labels on top of the bars in a bar plot.
    """
    for p in ax.patches:
        value = format(p.get_height(), '.1f')
        if value != "0.0":
            ax.annotate(value,
                        (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center', va='center',
                        xytext=(0, 9),
                        textcoords='offset points')

def plot_bar(data, algorithm, ax, title, x_order, strategy_col):
    """
    Plots the bar plot for a given algorithm.
    """
    plot = sns.barplot(data=data,
                       x=strategy_col, y='correct_percent',
                       hue=strategy_col, ax=ax,
                       order=x_order,
                       palette=sns.color_palette("viridis", 4), errorbar=None)
    ax.set_title(title)
    ax.set_xlabel('')
    ax.set_ylabel(f'{algorithm}\nCorrect Percent')

    # Add bar labels
    add_bar_labels(ax)

def plot_line(data, algorithm, ax, title, x_order, strategy_col):
    """
    Plots the line plot for a given algorithm.
    """
    sns.lineplot(data=data,
                 x='room_count', y='correct_percent', hue=strategy_col,
                 hue_order=x_order, ax=ax, palette=sns.color_palette("viridis", 4), errorbar=None)
    ax.set_title(title)
    ax.set_xlabel('Room Count')
    ax.set_ylabel(f'{algorithm}\nCorrect Percent')
    ax.legend().remove()

def create_plot(ax, data, algorithm_value, title, ylabel, hue_order, palette, strategy_col):
    """
    Creates a bar plot for the given algorithm value.
    """
    data_avg = data[data['algorithm_value'] == algorithm_value]
    bar_width = 0.4

    bar_positions = list(range(len(data_avg[strategy_col].unique())))

    data_avg = data_avg.copy()
    data_avg['weighted_correct_percent'] = data_avg['correct_percent'] * data_avg['room_count']
    weighted_avg = data_avg.groupby(strategy_col)['weighted_correct_percent'].sum() / \
                   data_avg.groupby(strategy_col)['room_count'].sum()
    weighted_avg = weighted_avg.reset_index()
    weighted_avg.columns = [strategy_col, 'weighted_correct_percent']

    for i, threshold in enumerate(weighted_avg[strategy_col]):
        correct_avg = data_avg[data_avg[strategy_col] == threshold]['correct_percent'].mean()
        weighted_correct_avg = \
        weighted_avg[weighted_avg[strategy_col] == threshold]['weighted_correct_percent'].values[0]

        ax.bar(bar_positions[i] - bar_width / 2, correct_avg, bar_width, label='Average', color=palette[i], alpha=1.0)

        ax.bar(bar_positions[i] + bar_width / 2, weighted_correct_avg, bar_width, label='Weighted Average',
               color=palette[i],  alpha=0.5)

    ax.set_title(title)
    ax.set_xlabel('Router RSSI Threshold')
    ax.set_ylabel(ylabel)
    ax.set_xticks(bar_positions)
    ax.set_xticklabels(weighted_avg[strategy_col])

    add_bar_labels(ax)

def main():
    """
    Main function to read the CSV file, process the data, and generate multiple plots
    showing the correct percent by algorithm and handling missing values strategy.
    """
    strategy_col = 'value_scaling_strategy'
    hue_order = ["exponential",  "none", "positive", "powed"]
    file_path = "11_value_scaling_strategy.csv"

    df = read_csv(file_path)
    if df is not None:
        group_columns = determine_group_columns(df)
        print(df)
        aggregated_data = aggregate_data(df, group_columns)
        group_columns.extend(['room_name', 'room_count'])
        grouped_by_parameters = aggregate_correct_percent_by_parameters(aggregated_data, group_columns)
        cleaned_data = remove_constant_columns(grouped_by_parameters)

        knn_euclidean_data = cleaned_data[cleaned_data['algorithm'] == 'knn_euclidean']
        knn_sorensen_data = cleaned_data[cleaned_data['algorithm'] == 'knn_sorensen']
        rf_data = cleaned_data[cleaned_data['algorithm'] == 'random_forest']
        svm_linear_data = cleaned_data[cleaned_data['algorithm'] == 'svm_linear']
        svm_rbf_data = cleaned_data[cleaned_data['algorithm'] == 'svm_rbf']


        fig, axes = plt.subplots(5, 3, figsize=(18, 12), sharey=True)

        configure_axes(axes)

        for i, value in enumerate([5, 7, 9]):
            plot_bar(knn_euclidean_data[knn_euclidean_data['algorithm_value'] == value],
                     'KNN (euclidean)', axes[0, i], f'n_neighbors = {value}', hue_order, strategy_col)

        for i, value in enumerate([5, 7, 9]):
            plot_bar(knn_sorensen_data[knn_sorensen_data['algorithm_value'] == value],
                     'KNN (sorensen)', axes[1, i], f'n_neighbors = {value}', hue_order, strategy_col)

        for i, value in enumerate([0.8, 2, 8.0]):
            plot_bar(rf_data[rf_data['algorithm_value'] == value],
                     'Random Forest', axes[2, i], f'n_estimators = {value}', hue_order, strategy_col)

        for i, value in enumerate([0.001, 0.005, 0.01]):
            plot_bar(svm_linear_data[svm_linear_data['algorithm_value'] == value],
                     'SVM (linear)', axes[3, i], f'C = {value}', hue_order, strategy_col)

        for i, value in enumerate([0.5, 1, 5]):
            plot_bar(svm_rbf_data[svm_rbf_data['algorithm_value'] == value],
                     'SVM (rbf)', axes[4, i], f'C = {value}', hue_order, strategy_col)

        plt.tight_layout(rect=(0, 0, 1, 1))

        plt.savefig(file_path.split(".")[0] + "_01.png", dpi=300)
        plt.show()

        fig, axes = plt.subplots(5, 3, figsize=(18, 12), sharey=True)

        configure_axes(axes)

        for i, value in enumerate([5, 7, 9]):
            plot_line(knn_euclidean_data[knn_euclidean_data['algorithm_value'] == value],
                      'KNN (euclidean)', axes[0, i], f'n_neighbors = {value}', hue_order, strategy_col)

        for i, value in enumerate([5, 7, 9]):
            plot_line(knn_sorensen_data[knn_sorensen_data['algorithm_value'] == value],
                      'KNN (sorensen)', axes[1, i], f'n_neighbors = {value}', hue_order, strategy_col)

        for i, value in enumerate([0.8, 2, 8.0]):
            plot_line(rf_data[rf_data['algorithm_value'] == value],
                      'Random Forest', axes[2, i], f'n_estimators = {value}', hue_order, strategy_col)

        for i, value in enumerate([0.001, 0.005, 0.01]):
            plot_line(svm_linear_data[svm_linear_data['algorithm_value'] == value],
                      'SVM (linear)', axes[3, i], f'C = {value}', hue_order, strategy_col)

        for i, value in enumerate([0.5, 1, 5]):
            plot_line(svm_rbf_data[svm_rbf_data['algorithm_value'] == value],
                      'SVM (rbf)', axes[4, i], f'C = {value}', hue_order, strategy_col)

        plt.tight_layout(rect=(0, 0.03, 1, 1))
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center', ncol=len(hue_order))
        plt.savefig(file_path.split(".")[0] + "_02.png", dpi=300)
        plt.show()

        palette = sns.color_palette("viridis", 4)

        fig, axes = plt.subplots(5, 3, figsize=(18, 12), sharey=True)
        configure_axes(axes)

        for i, value in enumerate([5, 7, 9]):
            create_plot(axes[0, i], knn_euclidean_data, value, f'n_neighbors = {value}', 'KNN\nCorrect Percent',
                        hue_order, palette, strategy_col)

        for i, value in enumerate([5, 7, 9]):
            create_plot(axes[1, i], knn_sorensen_data, value, f'n_neighbors = {value}', 'KNN\nCorrect Percent',
                        hue_order, palette, strategy_col)

        for i, value in enumerate([0.8, 2.0, 8.0]):
            create_plot(axes[2, i], rf_data, value, f'n_estimators = {value}', 'Random Forest\nCorrect Percent',
                        hue_order, palette, strategy_col)

        for i, value in enumerate([0.001, 0.005, 0.01]):
            create_plot(axes[3, i], svm_linear_data, value, f'C = {value}', 'SVM (linear)\nCorrect Percent', hue_order,
                        palette, strategy_col)
            axes[3, i].set_xlabel('Router RSSI Threshold')

        for i, value in enumerate([0.5, 1, 5]):
            create_plot(axes[4, i], svm_rbf_data, value, f'C = {value}', 'SVM (rbf)\nCorrect Percent', hue_order,
                        palette, strategy_col)
            axes[4, i].set_xlabel('Router RSSI Threshold')

        plt.tight_layout(rect=(0, 0, 1, 1))

        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.05), ncol=len(hue_order))

        plt.savefig(file_path.split(".")[0] + "_03.png", dpi=300)
        plt.show()

if __name__ == "__main__":
    main()
