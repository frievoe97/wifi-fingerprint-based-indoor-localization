import pandas as pd
import ast
import requests

# Path to the CSV file
csv_file_path = 'backup_without_corridor.csv'

# IP addresses
ip_address_1 = "141.45.212.246"
ip_address_2 = "127.0.0.1"
current_ip = ip_address_1  # Set the IP address to be used

port = 8000

# API endpoints
ping_url = f"http://{current_ip}:{port}/health"
reset_url = f"http://{current_ip}:{port}/reset-database"
measurements_url = f"http://{current_ip}:{port}/measurements/add"

# Read the CSV file
df = pd.read_csv("data/" + csv_file_path)

def parse_routers(routers_str):
    """
    Parse the routers field from the CSV.

    Args:
        routers_str (str): String representation of the routers list.

    Returns:
        list: Parsed list of routers. Returns an empty list if parsing fails.
    """
    try:
        return ast.literal_eval(routers_str)
    except (ValueError, SyntaxError) as e:
        print(f"Error parsing routers: {e}")
        return []

# Apply the parse_routers function to the 'routers' column
df['parsed_routers'] = df['routers'].apply(parse_routers)

# Create JSON strings
json_list = []
for index, row in df.iterrows():
    entry = {
        "room_name": row['room_name'],
        "device_id": row['device_id'],
        "timestamp": row['timestamp'],
        "routers": row['parsed_routers']
    }
    json_list.append(entry)

def exit_on_failure(response, step_name):
    """
    Exit the program if a step fails.

    Args:
        response (requests.Response): The response object from the request.
        step_name (str): Name of the step for error reporting.

    Raises:
        SystemExit: Exits the program if the response is not successful.
    """
    if not response.ok:
        print(f"Error during {step_name}: {response.status_code}, {response.text}")
        exit(1)

# Send ping request
ping_response = requests.get(ping_url)
exit_on_failure(ping_response, "Ping")
print("Ping successful.")

# Reset the database
reset_response = requests.post(reset_url)
exit_on_failure(reset_response, "Database reset")
print("Database reset successful.")

# Send data to the API
for entry in json_list:
    print(entry)
    response = requests.post(measurements_url, json=entry)
    if response.status_code == 200:
        print(f"Successfully sent: {entry}")
    else:
        print(f"Error sending data: {response.status_code}, {response.text}")
        # exit(1)
