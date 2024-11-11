import logging
import os
import time
from fastapi import FastAPI, HTTPException, Depends, Body
from sqlalchemy.orm import Session
from sqlalchemy.exc import OperationalError
from prometheus_fastapi_instrumentator import Instrumentator
from models import Base, Room, Measurement, Router, MeasurementRouter, SessionLocal, engine
from schemas import MeasurementData, PredictData
from datetime import datetime
import numpy as np
from typing import List

from utils import process_received_data, remove_non_eduroam_bssids, remove_unreceived_bssids, process_fingerprint_data, \
    remove_rare_routers, prepare_data, handle_missing_values, prepare_received_data, handle_router_rssi_threshold, \
    value_scaling, knn, random_forest, svm

DATABASE_URL = os.getenv("DATABASE_URL")

EXAMPLE_PREDICT_DATA = {
            "routers": [
                {"bssid": "dc:b8:08:c9:54:a1", "signal_strength": -88, "ssid": "HowToUseEduroam"},
                {"bssid": "dc:b8:08:c9:54:a2", "signal_strength": -88, "ssid": "Gast@HTW"},
                {"bssid": "02:2a:da:c8:29:9c", "signal_strength": -90, "ssid": "HP51417C"},
                {"bssid": "dc:b8:08:c9:4b:e1", "signal_strength": -85, "ssid": "HowToUseEduroam"},
                {"bssid": "dc:b8:08:c9:4b:e2", "signal_strength": -85, "ssid": "Gast@HTW"},
                {"bssid": "dc:b8:08:c9:4b:e0", "signal_strength": -85, "ssid": "eduroam"},
                {"bssid": "2c:3a:fd:0e:83:f1", "signal_strength": -84, "ssid": "NAT-FZTD009"},
                {"bssid": "dc:b8:08:c9:5e:c2", "signal_strength": -89, "ssid": "Gast@HTW"},
                {"bssid": "dc:b8:08:c9:38:b0", "signal_strength": -84, "ssid": "eduroam"},
                {"bssid": "c8:7f:54:39:6e:f0", "signal_strength": -89, "ssid": "HTW_FRL"},
                {"bssid": "00:09:9a:00:b6:43", "signal_strength": -54, "ssid": "ELTX1001901"},
                {"bssid": "dc:b8:08:c9:04:a0", "signal_strength": -64, "ssid": "eduroam"},
                {"bssid": "dc:b8:08:c9:04:a2", "signal_strength": -63, "ssid": "Gast@HTW"},
                {"bssid": "dc:b8:08:c9:04:a1", "signal_strength": -63, "ssid": "HowToUseEduroam"},
                {"bssid": "dc:b8:08:c9:04:b0", "signal_strength": -65, "ssid": "eduroam"},
                {"bssid": "dc:b8:08:c9:10:90", "signal_strength": -81, "ssid": "eduroam"},
                {"bssid": "dc:b8:08:c8:fe:e1", "signal_strength": -80, "ssid": "HowToUseEduroam"},
                {"bssid": "dc:b8:08:c8:fe:e2", "signal_strength": -80, "ssid": "Gast@HTW"},
                {"bssid": "dc:b8:08:c8:fe:e0", "signal_strength": -81, "ssid": "eduroam"},
                {"bssid": "dc:b8:08:c8:fe:f0", "signal_strength": -87, "ssid": "eduroam"},
                {"bssid": "dc:b8:08:c9:56:41", "signal_strength": -88, "ssid": "HowToUseEduroam"},
                {"bssid": "dc:b8:08:c9:56:40", "signal_strength": -88, "ssid": "eduroam"},
                {"bssid": "34:64:a9:d6:43:f1", "signal_strength": -91, "ssid": "DIRECT-F0-HP Officejet 5740"},
                {"bssid": "dc:b8:08:c9:56:50", "signal_strength": -89, "ssid": "eduroam"}
            ]
        }

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Indoor Localization API",
    description="API for indoor localization using Wi-Fi fingerprinting",
    version="1.0.0"
)

Instrumentator().instrument(app).expose(app)

Base.metadata.create_all(bind=engine)

def get_db():
    retries = 5
    delay = 5

    db = None
    for attempt in range(retries):
        try:
            db = SessionLocal()
            yield db
            break
        except OperationalError as e:
            if attempt < retries - 1:
                logger.warning(f"Verbindung zur Datenbank fehlgeschlagen. Versuch {attempt + 1} von {retries}. Fehler: {e}")
                time.sleep(delay)
            else:
                logger.error(f"Verbindung zur Datenbank fehlgeschlagen nach {retries} Versuchen. Fehler: {e}")
                raise
        finally:
            if db:
                db.close()

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/measurements/add", response_model=dict)
def add_measurement(data: MeasurementData, db: Session = Depends(get_db)):
    logger.info("Adding new measurement")
    room_name = data.room_name
    device_id = data.device_id
    timestamp = data.timestamp
    routers = data.routers

    if not room_name or not device_id or not timestamp or not routers:
        logger.error("Missing data in request")
        raise HTTPException(status_code=400, detail="Missing data")

    room = db.query(Room).filter_by(room_name=room_name).first()
    if not room:
        room = Room(room_name=room_name)
        db.add(room)
        db.commit()
        db.refresh(room)

    timestamp_obj = datetime.utcfromtimestamp(timestamp)
    existing_measurement = db.query(Measurement).filter_by(device_id=device_id, timestamp=timestamp_obj).first()
    if existing_measurement:
        logger.info("Measurement already exists")
        raise HTTPException(status_code=409, detail="Measurement with the same device_id and timestamp already exists")

    measurement = Measurement(
        timestamp=timestamp_obj,
        device_id=device_id,
        room_id=room.room_id
    )
    db.add(measurement)
    db.commit()
    db.refresh(measurement)

    for router_data in routers:
        bssid = router_data.bssid
        ssid = router_data.ssid
        signal_strength = router_data.signal_strength

        if not bssid or signal_strength is None:
            continue

        router = db.query(Router).filter_by(bssid=bssid).first()
        if not router:
            router = Router(bssid=bssid, ssid=ssid)
            db.add(router)
            db.commit()
            db.refresh(router)

        measurement_router = MeasurementRouter(
            measurement_id=measurement.measurement_id,
            router_id=router.router_id,
            signal_strength=signal_strength
        )
        db.add(measurement_router)

    db.commit()
    logger.info("Measurement added successfully")
    return {"message": "Measurement added successfully"}

@app.post("/measurements/predict", response_model=dict)
def predict_room(
    data: PredictData = Body(
        example=EXAMPLE_PREDICT_DATA
    ),
    db: Session = Depends(get_db)
):
    logger.info("Predicting room")
    logger.info(f"Data: {data}")
    routers = data.routers
    ignore_measurements = data.ignore_measurements

    if not routers:
        logger.error("Missing data in request")
        raise HTTPException(status_code=400, detail="Missing data")

    use_remove_unreceived_bssids = data.use_remove_unreceived_bssids
    handle_missing_values_strategy = data.handle_missing_values_strategy
    router_selection = data.router_selection
    router_presence_threshold = data.router_presence_threshold
    value_scaling_strategy = data.value_scaling_strategy
    router_rssi_threshold = data.router_rssi_threshold
    algorithm = data.algorithm
    k_value = data.k_value
    weights = data.weights
    n_estimators = data.n_estimators
    c_value = data.c_value
    gamma_value = data.gamma_value
    max_depth = data.max_depth

    max_features = data.max_features

    if max_depth == "None":
        max_depth = None  

    if max_features == "None":
        max_features = None          


    received_data = process_received_data(routers)
    measurements = db.query(Measurement).all()
    result = []

    for measurement in measurements:
        if ignore_measurements and measurement.measurement_id in ignore_measurements:
            logger.info(f"Ignoring measurement with ID {ignore_measurements}")
            continue

        measurement_routers = db.query(MeasurementRouter).filter_by(measurement_id=measurement.measurement_id).all()
        for router in measurement_routers:
            router_info = db.query(Router).filter_by(router_id=router.router_id).first()
            result.append({
                'measurement_id': measurement.measurement_id,
                'timestamp': measurement.timestamp,
                'device_id': measurement.device_id,
                'room_id': measurement.room_id,
                'bssid': router_info.bssid,
                'ssid': router_info.ssid,
                'signal_strength': router.signal_strength
            })

    rooms = process_fingerprint_data(result)

    if use_remove_unreceived_bssids:
        rooms = remove_unreceived_bssids(rooms, received_data)

    if router_selection == 'eduroam':
        rooms = remove_non_eduroam_bssids(rooms)

    if router_presence_threshold > 0:
        rooms = remove_rare_routers(rooms, threshold=router_presence_threshold)

    X, y, mac_address_list = prepare_data(rooms)

    if X.size == 0 or y.size == 0:
        return {"error": "Training data is empty. Check the input data."}, 400

    X = handle_missing_values(X, mac_address_list, received_data, handle_missing_values_strategy)
    X_new = prepare_received_data(received_data, mac_address_list)
    min_rssi_value = np.array(min(X.min(), X_new.min()))

    X, X_new = handle_router_rssi_threshold(X, X_new, router_rssi_threshold=router_rssi_threshold)
    X, X_new = value_scaling(X, X_new, min_rssi_value=min_rssi_value, value_scaling_strategy=value_scaling_strategy)

    if X_new.size == 0:
        raise ValueError("Received data is empty. Check the input data.")

    optional_value = -1

    if algorithm == 'knn_sorensen':
        predicted_room, distance = knn(X, X_new, y, k_value, metric='sorensen', weights=weights)
    elif algorithm == 'knn_euclidean':
        predicted_room, distance = knn(X, X_new, y, k_value, metric='euclidean', weights=weights)
    elif algorithm == 'random_forest':
        predicted_room, distance = random_forest(X, X_new, y, n_estimators, max_depth, max_features)
    elif algorithm == 'svm_linear':
        predicted_room, distance, optional_value = svm(X, X_new, y, C=c_value, kernel='linear', gamma=gamma_value)
    elif algorithm == 'svm_rbf':
        predicted_room, distance, optional_value = svm(X, X_new, y, C=c_value, kernel='rbf', gamma=gamma_value)

    room_name = handle_get_room_name_by_id(predicted_room, db)

    logger.info(f"Predicted room: {predicted_room}")
    logger.info(f"Predicted room name: {room_name}")
    logger.info(f"Distance: {distance}")

    return {"room_name": room_name, "distance": distance, "optional_value": optional_value}

@app.get("/measurements/all", response_model=List[dict])
def get_all_measurements(db: Session = Depends(get_db)):
    logger.info("Fetching all measurements")
    measurements = db.query(
        Measurement.measurement_id,
        Measurement.timestamp,
        Measurement.device_id,
        Measurement.room_id,
        Room.room_name
    ).join(Room, Measurement.room_id == Room.room_id).all()

    result = []
    for measurement in measurements:
        routers = db.query(MeasurementRouter).filter_by(measurement_id=measurement.measurement_id).all()
        router_data = []
        for router in routers:
            router_info = db.query(Router).filter_by(router_id=router.router_id).first()
            router_data.append({
                'bssid': router_info.bssid,
                'ssid': router_info.ssid,
                'signal_strength': router.signal_strength
            })

        timestamp_seconds = int(measurement.timestamp.timestamp())

        result.append({
            'measurement_id': measurement.measurement_id,
            'timestamp': timestamp_seconds,
            'device_id': measurement.device_id,
            'room_id': measurement.room_id,
            'room_name': measurement.room_name,
            'routers': router_data
        })
    logger.info("All measurements fetched successfully")
    return result

@app.get("/rooms/all", response_model=List[dict])
def get_all_rooms(db: Session = Depends(get_db)):
    logger.info("Fetching all rooms")
    rooms = db.query(Room).all()

    result = []
    for room in rooms:
        result.append({
            'room_id': room.room_id,
            'room_name': room.room_name,
            'description': room.description,
            'coordinates': room.coordinates,
            'picture_path': room.picture_path,
            'additional_info': room.additional_info
        })
    logger.info("All rooms fetched successfully")
    return result

@app.get("/rooms/{room_id}", response_model=dict)
def get_room_by_id(room_id: int, db: Session = Depends(get_db)):
    logger.info(f"Fetching room with ID {room_id}")
    room = db.query(Room).filter_by(room_id=room_id).first()

    if not room:
        logger.error(f"Room with ID {room_id} not found")
        raise HTTPException(status_code=404, detail="Room not found")

    result = {
        'room_id': room.room_id,
        'room_name': room.room_name,
        'description': room.description,
        'coordinates': room.coordinates,
        'picture_path': room.picture_path,
        'additional_info': room.additional_info
    }
    logger.info(f"Room with ID {room_id} fetched successfully")
    return result

@app.get("/routers/all", response_model=List[dict])
def get_all_routers(db: Session = Depends(get_db)):
    logger.info("Fetching all routers")
    routers = db.query(Router).all()

    result = []
    for router in routers:
        result.append({
            'router_id': router.router_id,
            'ssid': router.ssid,
            'bssid': router.bssid
        })
    logger.info("All routers fetched successfully")
    return result

@app.get("/measurements/all", response_model=List[dict])
def get_all_measurements(db: Session = Depends(get_db)):
    logger.info("Fetching all measurements")
    measurements = db.query(
        Measurement.measurement_id,
        Measurement.timestamp,
        Measurement.device_id,
        Measurement.room_id,
        Room.room_name
    ).join(Room, Measurement.room_id == Room.room_id).all()

    result = []
    for measurement in measurements:
        routers = db.query(MeasurementRouter).filter_by(measurement_id=measurement.measurement_id).all()
        router_data = []
        for router in routers:
            router_info = db.query(Router).filter_by(router_id=router.router_id).first()
            router_data.append({
                'bssid': router_info.bssid,
                'ssid': router_info.ssid,
                'signal_strength': router.signal_strength
            })

        timestamp_seconds = int(measurement.timestamp.timestamp())

        result.append({
            'measurement_id': measurement.measurement_id,
            'timestamp': timestamp_seconds,
            'device_id': measurement.device_id,
            'room_id': measurement.room_id,
            'room_name': measurement.room_name,
            'routers': router_data
        })
    logger.info("All measurements fetched successfully")
    return result

@app.get("/measurements/{measurement_id}", response_model=dict)
def get_measurement_by_id(measurement_id: int, db: Session = Depends(get_db)):
    logger.info(f"Fetching measurement with ID {measurement_id}")
    measurement = db.query(Measurement).filter_by(measurement_id=measurement_id).first()

    if not measurement:
        logger.error(f"Measurement with ID {measurement_id} not found")
        raise HTTPException(status_code=404, detail="Measurement not found")

    routers = db.query(MeasurementRouter).filter_by(measurement_id=measurement.measurement_id).all()
    router_data = []
    for router in routers:
        router_info = db.query(Router).filter_by(router_id=router.router_id).first()
        router_data.append({
            'bssid': router_info.bssid,
            'ssid': router_info.ssid,
            'signal_strength': router.signal_strength
        })

    timestamp_seconds = int(measurement.timestamp.timestamp())

    result = {
        'measurement_id': measurement.measurement_id,
        'timestamp': timestamp_seconds,
        'device_id': measurement.device_id,
        'room_id': measurement.room_id,
        'routers': router_data
    }
    logger.info(f"Measurement with ID {measurement_id} fetched successfully")
    return result

@app.get("/routers/{router_id}", response_model=dict)
def get_router_by_id(router_id: int, db: Session = Depends(get_db)):
    logger.info(f"Fetching router with ID {router_id}")
    router = db.query(Router).filter_by(router_id=router_id).first()

    if not router:
        logger.error(f"Router with ID {router_id} not found")
        raise HTTPException(status_code=404, detail="Router not found")

    result = {
        'router_id': router.router_id,
        'ssid': router.ssid,
        'bssid': router.bssid
    }
    logger.info(f"Router with ID {router_id} fetched successfully")
    return result

@app.post("/reset-database")
def reset_database(db: Session = Depends(get_db)):
    try:
        db.query(MeasurementRouter).delete()
        db.query(Measurement).delete()
        db.query(Router).delete()
        db.query(Room).delete()
        
        db.commit()

        logger.info("Datenbank erfolgreich zurückgesetzt (Daten gelöscht)")
        return {"message": "Database reset successfully"}
    except Exception as e:
        db.rollback()
        logger.error(f"Fehler beim Zurücksetzen der Datenbank: {e}")
        raise HTTPException(status_code=500, detail="Fehler beim Zurücksetzen der Datenbank")



def handle_get_room_name_by_id(room_id: int, db: Session):
    """
    Retrieve the room name based on the room ID.

    Parameters:
    room_id (int): The ID of the room.
    db (Session): The database session.

    Returns:
    str: The name of the room or "Unknown" if not found.
    """
    room = db.query(Room).filter_by(room_id=room_id).first()
    if not room:
        return "Unknown"
    return room.room_name
