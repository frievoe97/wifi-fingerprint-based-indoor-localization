import random
import requests
import csv
import time
from itertools import product

def predict_room(measurement, param_values, url_predict, timeout=100, remaining_corridor_measurements=None, remaining_room_measurements=None):
    """
    Predict the room based on measurement data and parameter values.

    Args:
        measurement (dict): Measurement data.
        param_values (dict): Parameter values for the prediction.
        url_predict (str): URL for the prediction API.
        timeout (int): Timeout for the API request.
        remaining_corridor_measurements (list): Remaining corridor measurements to ignore.
        remaining_room_measurements (list): Remaining room measurements to ignore.

    Returns:
        dict: Prediction result from the API, or None if the request fails.
    """
    payload = {
        "routers": measurement['routers'],
        "ignore_measurements": [measurement['measurement_id']],
    }

    if remaining_corridor_measurements:
        payload["ignore_measurements"].extend(remaining_corridor_measurements)

    if remaining_room_measurements:
        payload["ignore_measurements"].extend(remaining_room_measurements)

    payload.update(param_values)

    # print(f"Paying load: {payload}")

    try:
        response_predict = requests.post(url_predict, json=payload, timeout=timeout)
        if response_predict.status_code == 200:
            return response_predict.json()
        else:
            print(f"Failed to send data. Status code: {response_predict.status_code}")
            return None
    except requests.exceptions.Timeout:
        print(f"Request timed out after {timeout} seconds.")
        return None
    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")
        return None

def generate_parameter_combinations(parameters):
    """
    Generate all possible combinations of parameter values.

    Args:
        parameters (dict): Dictionary of parameter values.

    Returns:
        list: List of all possible parameter combinations.
    """
    common_params = {k: v for k, v in parameters.items() if k != 'algorithm'}
    algorithm_params = parameters['algorithm']

    common_combinations = list(product(*common_params.values()))

    complete_combinations = []
    for combination in common_combinations:
        for algo, algo_params in algorithm_params.items():
            algo_param_combinations = list(product(*algo_params.values()))
            for algo_param_comb in algo_param_combinations:
                param_dict = dict(zip(common_params.keys(), combination))
                param_dict['algorithm'] = algo
                for i, param_name in enumerate(algo_params.keys()):
                    param_dict[param_name] = algo_param_comb[i]
                complete_combinations.append(param_dict)

    return complete_combinations

def get_measurement_ids(count, room_names, complete_data, ignore_id=None):
    """
    Get a list of measurement IDs to ignore for predictions.

    Args:
        count (int): Number of measurements per room to ignore.
        room_names (list): List of room names to consider.
        complete_data (list): Complete data set.
        ignore_id (int): ID of the measurement to exclude from the ignore list.

    Returns:
        list: List of measurement IDs to ignore.
    """
    room_measurements = {room: [] for room in room_names}

    for data in complete_data:
        if data['room_name'] in room_names and data['measurement_id'] != ignore_id:
            room_measurements[data['room_name']].append(data['measurement_id'])

    remaining_measurements = []

    for room, measurements in room_measurements.items():
        if len(measurements) > count:
            selected_measurements = random.sample(measurements, count)
            remaining_measurements.extend([m for m in measurements if m not in selected_measurements])
        else:
            selected_measurements = measurements

    return remaining_measurements

def compare_predictions(data, num_measurements, url_predict, parameters, parameter_names, rooms, corridors):
    """
    Compare predictions with actual room names.

    Args:
        data (list): List of data elements.
        num_measurements (int): Number of measurements to process.
        url_predict (str): URL for the prediction API.
        parameters (dict): Dictionary of parameter values.
        parameter_names (list): List of parameter names.
        rooms (list): List of room names.
        corridors (list): List of corridor names.

    Returns:
        list: Results of the comparison.
    """
    complete_data = data
    if rooms and corridors:
        data = [data_element for data_element in data if data_element['room_name'] in rooms]

    total_measurements = len(data)
    if num_measurements > 0:
        data = data[:num_measurements]
        total_measurements = num_measurements

    parameter_combinations = generate_parameter_combinations(parameters)
    start_time = time.time()
    results = []

    for i, measurement in enumerate(data):
        actual_room = measurement['room_name']
        device_id = measurement['device_id']
        measurement_id = measurement['measurement_id']
        room_id = measurement['room_id']
        print(f"Measuremtent ID: {measurement_id}")
        print(f"Actual room: {actual_room}")

        for param_values in parameter_combinations:
            remaining_corridor_measurements = []
            remaining_room_measurements = []

            if "measurements_per_room" in param_values and "measurements_per_corridor" in param_values:
                remaining_corridor_measurements = get_measurement_ids(param_values["measurements_per_corridor"], corridors, complete_data)
                remaining_room_measurements = get_measurement_ids(param_values["measurements_per_room"], rooms, complete_data, measurement_id)


            # print(f"Remaining corridor measurements: {remaining_corridor_measurements}")
            # print(f"Remaining room measurements: {remaining_room_measurements}")

            start_prediction_time = time.time()
            print(param_values)
            result = predict_room(measurement, param_values, url_predict, remaining_corridor_measurements=remaining_corridor_measurements, remaining_room_measurements=remaining_room_measurements)

            if result:
                prediction = result.get("room_name")
                distance = result.get("distance")
                end_prediction_time = time.time()
                duration = end_prediction_time - start_prediction_time
                correct = (prediction == actual_room)

                if "measurements_per_corridor" in param_values and prediction in corridors:
                    correct = "Not False"

                params_str = ", ".join(f"{name}={value}" for name, value in param_values.items())
                print(f"Prediction with params ({params_str}): {prediction} (took {duration:.2f} seconds) (distance: {distance}))")

                results.append([
                    device_id, measurement_id, actual_room, room_id, prediction, distance,
                    *param_values.values(), duration, correct
                ])

        elapsed_time = time.time() - start_time
        avg_time_per_iteration = elapsed_time / (i + 1)
        remaining_iterations = total_measurements - (i + 1)
        remaining_time = remaining_iterations * avg_time_per_iteration
        print(f"Average time per iteration: {avg_time_per_iteration:.2f} seconds")
        print(f"Estimated remaining time: {remaining_time:.2f} seconds")

    return results

def write_to_csv(results, filename, parameter_names):
    """
    Write results to a CSV file.

    Args:
        results (list): List of result data.
        filename (str): Name of the CSV file.
        parameter_names (list): List of parameter names.
    """
    headers = [
        "device_id", "measurement_id", "room_name", "room_id", "predict_room", "distance",
        *parameter_names, "duration", "correct"
    ]
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(headers)
        writer.writerows(results)
