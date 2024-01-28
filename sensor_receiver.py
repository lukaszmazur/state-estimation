import logging
import queue

import numpy as np

import carla


class SensorReceiver():
    """
    Class for receiving measurements from ego vehicle sensors and ground truth
    from world snapshot. It provides thread safe queues used in sensor callbacks.
    """

    def __init__(self, world_map, ego_id):
        self._gt_queue = queue.Queue()
        self._imu_queue = queue.Queue()
        self._gnss_queue = queue.Queue()

        self._gnss_transform = self._calculate_gnss_transform(world_map)
        self._ego_id = ego_id

    def get_gt_queue(self):
        return self._gt_queue

    def get_imu_queue(self):
        return self._imu_queue

    def get_gnss_queue(self):
        return self._gnss_queue

    def retrieve_data(self, frame):
        """
        Retrieves data from all sensor queues for a given frame.

        Returns:
            tuple: A tuple (gt_data, imu_data, gnss_data).
                   Only gt_data must be present, others can be None.
        """

        # do not catch queue.Empty exception for Ground Truth, because it
        # must be present
        world_snapshot = self._retrieve_data(
            self._gt_queue, frame, timeout=0.05)

        if not world_snapshot.has_actor(self._ego_id):
            return None, None, None
        actor_snapshot = world_snapshot.find(self._ego_id)
        gt_data = (
            actor_snapshot.get_transform().location.x,
            actor_snapshot.get_transform().location.y,
            actor_snapshot.get_transform().location.z,
            actor_snapshot.get_transform().rotation.roll,
            actor_snapshot.get_transform().rotation.pitch,
            actor_snapshot.get_transform().rotation.yaw,
            actor_snapshot.get_velocity().x,
            actor_snapshot.get_velocity().y,
            actor_snapshot.get_velocity().z,
            actor_snapshot.get_angular_velocity().x,
            actor_snapshot.get_angular_velocity().y,
            actor_snapshot.get_angular_velocity().z,
            actor_snapshot.get_acceleration().x,
            actor_snapshot.get_acceleration().y,
            actor_snapshot.get_acceleration().z,
            world_snapshot.timestamp.elapsed_seconds)

        try:
            imu_sensor_data = self._retrieve_data(
                self._imu_queue, frame, timeout=0.01)
            imu_data = (
                imu_sensor_data.accelerometer.x, imu_sensor_data.accelerometer.y, imu_sensor_data.accelerometer.z,
                imu_sensor_data.gyroscope.x, imu_sensor_data.gyroscope.y, imu_sensor_data.gyroscope.z,
                imu_sensor_data.timestamp
            )
            logging.info(f'IMU data: {imu_data}')
        except queue.Empty:
            imu_data = None

        try:
            gnss_sensor_data = self._retrieve_data(
                self._gnss_queue, frame, timeout=0.01)
            location = self._geolocation_to_location(
                carla.GeoLocation(gnss_sensor_data.latitude, gnss_sensor_data.longitude, gnss_sensor_data.altitude))
            gnss_data = (location.x, location.y, location.z,
                         gnss_sensor_data.timestamp)
            logging.info(f'GNSS data: {gnss_data}')
        except queue.Empty:
            gnss_data = None

        return gt_data, imu_data, gnss_data

    def _retrieve_data(self, sensor_queue, current_frame, timeout=0.1):
        # retrieve data in the loop, because we want to get data corresponding
        # to the current frame
        while True:
            data = sensor_queue.get(timeout=timeout)
            if data.frame == current_frame:
                return data

    @staticmethod
    def _calculate_gnss_transform(world_map):
        """
        Calculates matrix to transform from carla.GeoLocation to carla.Location.
        """
        # Transformation between GeoLocation and Location can be written as:
        # L = C @ G, where L is a Location matrix, C is transformation matrix
        # and G is GeoLocation matrix.
        # Here we solve for C = L @ G^(-1), for given Locations and GeoLocations.

        # Arbitrary noncolinear and noncoplanar points and their corresponding
        # GeoLocations.
        vectors = ((1, 0, 0), (0, 1, 0), (0, 0, 1))
        locs = [carla.Location(*v) for v in vectors]
        geolocs = [world_map.transform_to_geolocation(l) for l in locs]

        l = np.array([
            [loc.x for loc in locs],
            [loc.y for loc in locs],
            [loc.z for loc in locs]
        ], dtype=np.float)

        g = np.array([
            [geoloc.latitude for geoloc in geolocs],
            [geoloc.longitude for geoloc in geolocs],
            [geoloc.altitude for geoloc in geolocs]
        ], dtype=np.float)

        c = l.dot(np.linalg.inv(g))
        return c

    def _geolocation_to_location(self, geoloc):
        """
        Transform from carla.GeoLocation to carla.Location.
        """
        geoloc_array = np.array(
            [geoloc.latitude, geoloc.longitude, geoloc.altitude])
        loc_array = self._gnss_transform.dot(geoloc_array.T)
        return carla.Location(*loc_array)
