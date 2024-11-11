import requests
import csv
from datetime import datetime

# IP addresses
ip_address_1 = "141.45.212.246"
ip_address_2 = "127.0.0.1"
current_ip = ip_address_2

# API URL
url_fetch = f"http://{current_ip}:8000/measurements/all"

def fetch_data(url):
    """
    Fetch data from the given URL.

    Args:
        url (str): The URL to fetch data from.

    Returns:
        dict: The JSON response from the URL.

    Raises:
        requests.HTTPError: If an HTTP error occurs.
    """
    response = requests.get(url)
    response.raise_for_status()  # Error handling for HTTP requests
    return response.json()

def generate_filename():
    """
    Generate a filename with the current timestamp.

    Returns:
        str: The generated filename.
    """
    # Generate current timestamp
    current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    return f"backup_wifi_fingerprints_virtual_machine_{current_time}.csv"

def save_to_csv(data, filename):
    """
    Save the given data to a CSV file.

    Args:
        data (list): The data to save.
        filename (str): The name of the CSV file.
    """
    # Determine headers from the keys of the first element
    headers = data[0].keys()

    with open(filename, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=headers)
        writer.writeheader()
        for row in data:
            # Reformat timestamp
            # row['timestamp'] = format_timestamp(row['timestamp'])
            writer.writerow(row)

def main():
    data = fetch_data(url_fetch)
    filename = generate_filename()
    save_to_csv(data, filename)
    print(f"Backup successfully saved in {filename}")

if __name__ == "__main__":
    main()
