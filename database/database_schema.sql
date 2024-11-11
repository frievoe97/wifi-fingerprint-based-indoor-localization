DROP DATABASE IF EXISTS wifi_fingerprints;

CREATE DATABASE wifi_fingerprints;

USE wifi_fingerprints;

DROP TABLE IF EXISTS measurement_router;
DROP TABLE IF EXISTS measurements;
DROP TABLE IF EXISTS routers;
DROP TABLE IF EXISTS rooms;

CREATE TABLE rooms (
      room_id INT AUTO_INCREMENT PRIMARY KEY,
      room_name VARCHAR(255) NOT NULL,
      description VARCHAR(255),
      coordinates VARCHAR(255),
      picture_path VARCHAR(255),
      additional_info VARCHAR(255),
      UNIQUE KEY (room_name)
);

CREATE TABLE measurements (
      measurement_id INT AUTO_INCREMENT PRIMARY KEY,
      timestamp TIMESTAMP NOT NULL,
      device_id VARCHAR(255) NOT NULL,
      room_id INT NOT NULL,
      FOREIGN KEY (room_id) REFERENCES rooms(room_id)
);

CREATE TABLE routers (
      router_id INT AUTO_INCREMENT PRIMARY KEY,
      ssid VARCHAR(255) NOT NULL,
      bssid VARCHAR(255) NOT NULL,
      UNIQUE KEY (bssid)
);

CREATE TABLE measurement_router (
      measurement_id INT NOT NULL,
      router_id INT NOT NULL,
      signal_strength INT,
      PRIMARY KEY (measurement_id, router_id),
      FOREIGN KEY (measurement_id) REFERENCES measurements(measurement_id),
      FOREIGN KEY (router_id) REFERENCES routers(router_id)
);
