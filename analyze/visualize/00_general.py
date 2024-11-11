import seaborn as sns
from matplotlib import pyplot as plt
import matplotlib.ticker as mtick
from viz_utils import read_csv, measurements_per_room

# Set font size
plt.rcParams.update({'font.size': 7})

def main():
    """
    Main function to read the CSV file, process the data, and generate a bar plot
    showing the number of measurements per room.
    """
    file_path = "01_knn_weights.csv"
    df = read_csv(file_path)

    if df is not None:
        room_measurements = measurements_per_room(df)

        # Sort values in descending order by num_measurements
        room_measurements = room_measurements.sort_values(by='num_measurements', ascending=False)

        plt.figure(figsize=(12, 5))

        # Define the color ramp
        color = sns.color_palette("viridis", 1)[0]

        bar_plot = sns.barplot(
            x='room_name', y='num_measurements', data=room_measurements,
            color=color
        )
        bar_plot.set_title('Number of Measurements per Room')
        bar_plot.set_xlabel('Room Name')
        bar_plot.set_ylabel('Number of Measurements')
        plt.xticks(rotation=45, ha='right')

        # Make sure that the y-axis shows only integer values
        bar_plot.yaxis.set_major_locator(mtick.MaxNLocator(integer=True))

        # Add values above each bar
        for p in bar_plot.patches:
            bar_plot.annotate(
                format(p.get_height(), '.0f'),
                (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center', xytext=(0, 9),
                textcoords='offset points'
            )

        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.tight_layout()

        # Save the plot as an image
        plt.savefig('00_general_01.png', dpi=300)

        plt.show()

if __name__ == "__main__":
    main()
