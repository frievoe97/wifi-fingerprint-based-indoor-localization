from pydantic import BaseModel
from typing import List, Optional, Union

class RouterData(BaseModel):
    """
    RouterData represents the structure of a single Wi-Fi router's data.

    Attributes:
    - ssid (str): The SSID (Service Set Identifier) of the Wi-Fi network, which is the name of the network.
    - bssid (str): The BSSID (Basic Service Set Identifier), which is the MAC address of the Wi-Fi router.
    - signal_strength (int): The signal strength of the Wi-Fi router, typically measured in dBm.
    """
    ssid: str
    bssid: str
    signal_strength: int

class MeasurementData(BaseModel):
    """
    MeasurementData represents the structure of a single Wi-Fi fingerprinting measurement.

    Attributes:
    - room_name (str): The name of the room where the measurement was taken.
    - device_id (str): The ID of the device that performed the measurement.
    - timestamp (int): The time when the measurement was taken, represented as a Unix timestamp in seconds.
    - routers (List[RouterData]): A list of RouterData objects representing the routers detected during the measurement.
    """
    room_name: str
    device_id: str
    timestamp: int
    routers: List[RouterData]

class PredictData(BaseModel):
    """
    PredictData represents the structure of the data required for predicting the room based on Wi-Fi fingerprints.

    Attributes:
    - routers (List[RouterData]): A list of RouterData objects representing the routers detected in the current environment.
    - ignore_measurements (Optional[List[int]]): A list of measurement IDs that should be ignored during prediction.
    - use_remove_unreceived_bssids (Optional[bool]): Whether to remove BSSIDs that were not received during prediction. Default is True.
    - handle_missing_values_strategy (Optional[str]): Strategy for handling missing values in the data. Default is "use_received".
    - router_selection (Optional[str]): Strategy for selecting routers to use in the prediction. Default is 'all'.
    - router_presence_threshold (Optional[float]): A threshold for the presence of routers. Routers below this threshold may be ignored. Default is 0.0.
    - value_scaling_strategy (Optional[str]): Strategy for scaling the values, such as signal strengths. Default is 'none'.
    - router_rssi_threshold (Optional[int]): The minimum signal strength (RSSI) threshold for considering a router in the prediction. Default is -100.
    - algorithm (Optional[str]): The algorithm to use for room prediction, such as 'knn_euclidean', 'random_forest', etc. Default is 'knn_euclidean'.
    - k_value (Optional[int]): The 'k' value to use for K-Nearest Neighbors (KNN) algorithms. Default is 5.
    - weights (Optional[str]): The weighting strategy for KNN algorithms. Default is 'uniform'.
    - n_estimators (Optional[int]): Number of trees in the Random Forest algorithm. Default is 300.
    - c_value (Optional[float]): The regularization parameter for Support Vector Machines (SVM). Default is 1.0.
    - gamma_value (Optional[float]): The kernel coefficient for SVM. Default is 1.0.
    - max_depth (Optional[int]): The maximum depth of trees in the Random Forest algorithm. Default is None.
    """
    routers: List[RouterData]
    ignore_measurements: Optional[List[int]] = None
    use_remove_unreceived_bssids: Optional[bool] = True
    handle_missing_values_strategy: Optional[str] = "use_received"
    router_selection: Optional[str] = 'all'
    router_presence_threshold: Optional[float] = 0.0
    value_scaling_strategy: Optional[str] = 'none'
    router_rssi_threshold: Optional[int] = -100
    algorithm: Optional[str] = 'knn_euclidean'
    k_value: Optional[int] = 5
    weights: Optional[str] = 'uniform'
    n_estimators: Optional[int] = 300
    c_value: Optional[float] = 1.0
    gamma_value: Optional[str] = "auto"
    max_depth: Optional[Union[int, str]] = "None"
    max_features: Optional[Union[int, float, str]] = "sqrt"
