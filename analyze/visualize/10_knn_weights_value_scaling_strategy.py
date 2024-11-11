import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import matplotlib.ticker as mtick

from viz_utils import read_csv, aggregate_data, remove_constant_columns, \
    aggregate_correct_percent_by_parameters, determine_group_columns, measurements_per_room

plt.rcParams.update({'font.size': 12})


def weighted_average(df, value_column, weight_column):
    return (df[value_column] * df[weight_column]).sum() / df[weight_column].sum()


def plot_line_charts(cleaned_data):
    knn_euclidean_distance = cleaned_data[
        (cleaned_data['algorithm'] == 'knn_euclidean') & (cleaned_data['weights'] == 'distance')]
    knn_euclidean_uniform = cleaned_data[
        (cleaned_data['algorithm'] == 'knn_euclidean') & (cleaned_data['weights'] == 'uniform')]
    knn_sorensen_distance = cleaned_data[
        (cleaned_data['algorithm'] == 'knn_sorensen') & (cleaned_data['weights'] == 'distance')]
    knn_sorensen_uniform = cleaned_data[
        (cleaned_data['algorithm'] == 'knn_sorensen') & (cleaned_data['weights'] == 'uniform')]

    fig, axes = plt.subplots(2, 2, figsize=(18, 10))

    for ax in axes.flat:
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        ax.set_ylim(0, 100)
        ax.yaxis.set_major_formatter(mtick.PercentFormatter())
        ax.yaxis.set_major_locator(mtick.MultipleLocator(10))
        ax.xaxis.set_major_locator(mtick.MultipleLocator(1))

    palette = sns.color_palette("viridis", len(cleaned_data['value_scaling_strategy'].unique()))


    sns.lineplot(data=knn_euclidean_distance, x='room_count', y='correct_percent', hue='value_scaling_strategy',
                 ax=axes[0, 0], palette=palette, errorbar=None)
    axes[0, 0].set_title('KNN (euclidean) + Distance')
    axes[0, 0].set_ylabel('Correct Percent')

    sns.lineplot(data=knn_euclidean_uniform, x='room_count', y='correct_percent', hue='value_scaling_strategy',
                 ax=axes[0, 1], palette=palette, errorbar=None)
    axes[0, 1].set_title('KNN (euclidean) + Uniform')
    axes[0, 1].set_ylabel('Correct Percent')

    sns.lineplot(data=knn_sorensen_distance, x='room_count', y='correct_percent', hue='value_scaling_strategy',
                 ax=axes[1, 0], palette=palette, errorbar=None)
    axes[1, 0].set_title('KNN (sorensen) + Distance')
    axes[1, 0].set_ylabel('Correct Percent')

    sns.lineplot(data=knn_sorensen_uniform, x='room_count', y='correct_percent', hue='value_scaling_strategy',
                 ax=axes[1, 1], palette=palette, errorbar=None)
    axes[1, 1].set_title('KNN (sorensen) + Uniform')
    axes[1, 1].set_ylabel('Correct Percent')

    plt.tight_layout(rect=(0, 0, 1, 1))
    plt.savefig('10_knn_weights_value_scaling_strategy_01.png', dpi=300)
    plt.show()


def plot_combined_line_charts(cleaned_data):
    cleaned_data['algorithm_weights'] = cleaned_data['algorithm'] + ' + ' + cleaned_data['weights']
    value_scaling_strategies = cleaned_data['value_scaling_strategy'].unique()

    fig, axes = plt.subplots(2, 2, figsize=(18, 10))

    for ax in axes.flat:
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        ax.set_ylim(0, 100)
        ax.yaxis.set_major_formatter(mtick.PercentFormatter())
        ax.yaxis.set_major_locator(mtick.MultipleLocator(10))
        ax.xaxis.set_major_locator(mtick.MultipleLocator(1))

    palette = sns.color_palette("viridis", len(cleaned_data['algorithm_weights'].unique()))

    for i, strategy in enumerate(value_scaling_strategies):
        ax = axes.flat[i]
        subset = cleaned_data[cleaned_data['value_scaling_strategy'] == strategy]
        sns.lineplot(data=subset, x='room_count', y='correct_percent', hue='algorithm_weights', ax=ax, palette=palette,
                     errorbar=None)
        ax.set_title(strategy)
        ax.set_ylabel('Correct Percent')
        ax.legend(title='Algorithm + Weights')

    plt.tight_layout(rect=(0, 0, 1, 1))
    plt.savefig('10_knn_weights_value_scaling_strategy_03.png', dpi=300)
    plt.show()


def plot_bar_charts(cleaned_data):
    averages = cleaned_data.groupby(['weights', 'value_scaling_strategy'])['correct_percent'].mean().reset_index()
    weighted_averages = cleaned_data.groupby(['weights', 'value_scaling_strategy'], group_keys=False).apply(
        lambda x: weighted_average(x, 'correct_percent', 'room_count')).reset_index()
    weighted_averages.columns = ['weights', 'value_scaling_strategy', 'weighted_correct_percent']
    results = pd.merge(averages, weighted_averages, on=['weights', 'value_scaling_strategy'])

    value_scaling_strategies = results['value_scaling_strategy'].unique()
    fig, axes = plt.subplots(1, len(value_scaling_strategies), figsize=(24, 6), sharey=True)

    palette = sns.color_palette("viridis", 1)
    bar_width = 0.4

    for ax, strategy in zip(axes, value_scaling_strategies):
        subset = results[results['value_scaling_strategy'] == strategy]
        index = np.arange(len(subset))

        bars1 = ax.bar(index, subset['correct_percent'], bar_width, label='Durchschnitt', color=palette[0])
        bars2 = ax.bar(index + bar_width, subset['weighted_correct_percent'], bar_width,
                       label='Gewichteter Durchschnitt', color=palette[0], alpha=0.6)

        ax.set_xlabel('Weights')
        ax.set_title(strategy)
        ax.set_xticks(index + bar_width / 2)
        ax.set_xticklabels(subset['weights'])
        ax.set_ylim(0, 100)
        ax.yaxis.set_major_formatter(mtick.PercentFormatter())
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        if ax == axes[0]:
            ax.set_ylabel('Correct Percent')

        for bar in bars1:
            yval = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, yval + 1, f'{yval:.2f}%', ha='center', va='bottom', fontsize=10)

        for bar in bars2:
            yval = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, yval + 1, f'{yval:.2f}%', ha='center', va='bottom', fontsize=10)

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=2)

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig('10_knn_weights_value_scaling_strategy_02.png', dpi=300)
    plt.show()


if __name__ == "__main__":
    # output_dir = "output/2024-06-20_23-10-27_current"
    # file_path = f"{output_dir}/7_value_scaling_strategy_knn_distance.csv"
    file_path = "10_knn_weights_value_scaling_strategy.csv"

    df = read_csv(file_path)
    if df is not None:
        group_columns = determine_group_columns(df)
        aggregated_data = aggregate_data(df, group_columns)
        group_columns.extend(['room_name', 'room_count'])
        grouped_by_parameters = aggregate_correct_percent_by_parameters(aggregated_data, group_columns)
        cleaned_data = remove_constant_columns(grouped_by_parameters)
        measurements_per_room = measurements_per_room(df)

        plot_line_charts(cleaned_data)
        plot_combined_line_charts(cleaned_data)
        plot_bar_charts(cleaned_data)
