import requests

def fetch_data(url):
    """
    Fetch data from the specified URL.

    Args:
        url (str): URL to fetch data from.

    Returns:
        list: Fetched data as a list of dictionaries, or None if fetching fails.
    """
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Failed to fetch data. Status code: {response.status_code}")
        return None
