import random
import os
import datetime
import yaml
from fetch_data import fetch_data
from process_data import compare_predictions, write_to_csv


def load_config(config_path='config.yaml'):
    """
    Load configuration from a YAML file.

    Args:
        config_path (str): Path to the YAML configuration file.

    Returns:
        dict: Configuration dictionary.
    """
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)


def create_output_directory():
    """
    Create an output directory with the current timestamp.

    Returns:
        str: Path to the created output directory.
    """
    output_dir = "output/" + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def filter_data(data, rooms, corridors):
    """
    Filter data to include only specified rooms and corridors.

    Args:
        data (list): List of data elements.
        rooms (list): List of room names to include.
        corridors (list): List of corridor names to include.

    Returns:
        list: Filtered list of data elements.
    """
    return [data_element for data_element in data if
            data_element['room_name'] in rooms or data_element['room_name'] in corridors]


def main():
    config = load_config()
    url_fetch = config['url_fetch']
    url_predict = config['url_predict']
    num_measurements = config['num_measurements']
    parameter_sets = config['parameter_sets']
    rooms = config.get('rooms', [])
    corridors = config.get('corridors', [])

    output_dir = create_output_directory()
    data = fetch_data(url_fetch)

    if data:
        if rooms or corridors:
            data = filter_data(data, rooms, corridors)

        random.shuffle(data)

        for step, parameters in enumerate(parameter_sets, start=1):
            api_parameters = parameters["parameters"]
            parameter_names = list(api_parameters.keys()) + ["algorithm_value"]
            results = compare_predictions(data, num_measurements, url_predict, api_parameters, parameter_names, rooms,
                                          corridors)
            filename = os.path.join(output_dir, f"{parameters['name']}.csv")
            write_to_csv(results, filename, parameter_names)
            print(f"Results have been written to {filename}")


if __name__ == "__main__":
    main()
