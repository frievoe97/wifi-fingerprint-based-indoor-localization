import os
from sqlalchemy import Column, Integer, String, ForeignKey, TIMESTAMP
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker
from sqlalchemy import create_engine

DATABASE_URL = os.getenv("DATABASE_URL")

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class Room(Base):
    """
    Room model represents the 'rooms' table in the database.

    Attributes:
    - room_id (int): Primary key, auto-incremented unique identifier for the room.
    - room_name (str): Name of the room, must be unique and non-nullable.
    - description (str): Optional description of the room.
    - coordinates (str): Optional coordinates representing the room's location.
    - picture_path (str): Optional path to a picture representing the room.
    - additional_info (str): Optional field for any additional information about the room.
    """
    __tablename__ = 'rooms'

    room_id = Column(Integer, primary_key=True, index=True)
    room_name = Column(String(255), unique=True, index=True, nullable=False)
    description = Column(String(255))
    coordinates = Column(String(255))
    picture_path = Column(String(255))
    additional_info = Column(String(255))

class Measurement(Base):
    """
    Measurement model represents the 'measurements' table in the database.

    Attributes:
    - measurement_id (int): Primary key, auto-incremented unique identifier for the measurement.
    - timestamp (TIMESTAMP): Timestamp when the measurement was taken, non-nullable.
    - device_id (str): Identifier for the device that took the measurement, non-nullable.
    - room_id (int): Foreign key linking to the 'rooms' table, representing the room where the measurement was taken.
    
    Relationships:
    - room: Establishes a relationship with the Room model, linking each measurement to a specific room.
    """
    __tablename__ = 'measurements'

    measurement_id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(TIMESTAMP, nullable=False)
    device_id = Column(String(255), nullable=False)
    room_id = Column(Integer, ForeignKey('rooms.room_id'), nullable=False)

    room = relationship("Room")

class Router(Base):
    """
    Router model represents the 'routers' table in the database.

    Attributes:
    - router_id (int): Primary key, auto-incremented unique identifier for the router.
    - ssid (str): SSID (Service Set Identifier) of the Wi-Fi network, non-nullable.
    - bssid (str): BSSID (Basic Service Set Identifier) or MAC address of the router, must be unique and non-nullable.
    """
    __tablename__ = 'routers'

    router_id = Column(Integer, primary_key=True, index=True)
    ssid = Column(String(255), nullable=False)
    bssid = Column(String(255), unique=True, nullable=False)

class MeasurementRouter(Base):
    """
    MeasurementRouter model represents the many-to-many relationship between measurements and routers.

    This model is used to map which routers were detected during each measurement, along with the signal strength.

    Attributes:
    - measurement_id (int): Foreign key linking to the 'measurements' table, part of the composite primary key.
    - router_id (int): Foreign key linking to the 'routers' table, part of the composite primary key.
    - signal_strength (int): Signal strength of the router during the measurement.

    Relationships:
    - measurement: Establishes a relationship with the Measurement model, linking each entry to a specific measurement.
    - router: Establishes a relationship with the Router model, linking each entry to a specific router.
    """
    __tablename__ = 'measurement_router'

    measurement_id = Column(Integer, ForeignKey('measurements.measurement_id'), primary_key=True)
    router_id = Column(Integer, ForeignKey('routers.router_id'), primary_key=True)
    signal_strength = Column(Integer)

    measurement = relationship("Measurement")
    router = relationship("Router")
