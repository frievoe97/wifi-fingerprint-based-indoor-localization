import network
import urequests
import utime

# Initialize the Wi-Fi interface globally
wlan = network.WLAN(network.STA_IF)

# Function to connect to a Wi-Fi network
def connect_to_wifi(ssid, password):
    global wlan
    wlan.active(True)
    
    if not wlan.isconnected():
        print(f"Connecting to {ssid}...")
        wlan.connect(ssid, password)
        
        while not wlan.isconnected():
            print("Attempting to establish connection...")
            utime.sleep(1)
    
    print("Connected to Wi-Fi!")

# Function to scan available Wi-Fi networks and send the data to an API
def scan_and_send(api_route):
    global wlan
    
    # Scan for available networks
    networks = wlan.scan()
    
    # Prepare the data structure for the API
    routers = []
    for network in networks:
        bssid = ":".join("{:02x}".format(byte) for byte in network[1])
        signal_strength = network[3]
        ssid = network[0].decode('utf-8')
        
        if len(ssid) > 0:
            routers.append({
                "ssid": ssid,
                "bssid": bssid,
                "signal_strength": signal_strength
            })
    
    data = {
        "routers": routers
    }
    
    # Send the data via a POST request to the API
    try:
        response = urequests.post(api_route, json=data)
        if response.status_code == 200:
            response_json = response.json()
            print(f"Room Prediction: {response_json['room_name']}")
        else:
            print(f"Error in room prediction: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"Error sending data: {e}")

# Main function
def main(ssid, password, api_route):
    connect_to_wifi(ssid, password)
    
    # Perform the Wi-Fi scan and send the data to the API
    scan_and_send(api_route)

if __name__ == '__main__':
    # Define SSID, password, and API route here
    ssid = 'Rechnernetze'
    password = ''
    api_route = 'http://141.45.212.246:8000/measurements/predict'
    
    main(ssid, password, api_route)
