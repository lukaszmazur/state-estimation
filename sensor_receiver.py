import logging
import queue

import carla
from carla_utils import Geo2Location


class SensorReceiver():

    def __init__(self, world_map, ego_id):
        self._gt_queue = queue.Queue()
        self._imu_queue = queue.Queue()
        self._gnss_queue = queue.Queue()

        self._geo2location = Geo2Location(world_map)
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

        Returns: a tuple (gt_data, imu_data, gnss_data)
                 only gt_data must be present, others can be None
        """

        # do not catch queue.Empty exception for Ground Truth, because it
        # must be present
        world_snapshot = self._retrieve_data(self._gt_queue, frame, timeout=0.05)

        if not world_snapshot.has_actor(self._ego_id):
            return None, None, None
        actor_snapshot = world_snapshot.find(self._ego_id)
        gt_data = (
            actor_snapshot.get_transform().location.x, actor_snapshot.get_transform().location.y, actor_snapshot.get_transform().location.z,
            actor_snapshot.get_transform().rotation.roll, actor_snapshot.get_transform().rotation.pitch, actor_snapshot.get_transform().rotation.yaw,
            actor_snapshot.get_velocity().x, actor_snapshot.get_velocity().y, actor_snapshot.get_velocity().z,
            actor_snapshot.get_angular_velocity().x, actor_snapshot.get_angular_velocity().y, actor_snapshot.get_angular_velocity().z,
            actor_snapshot.get_acceleration().x, actor_snapshot.get_acceleration().y, actor_snapshot.get_acceleration().z,
            world_snapshot.timestamp.elapsed_seconds)

        try:
            imu_sensor_data = self._retrieve_data(self._imu_queue, frame, timeout=0.01)
            imu_data = (
                imu_sensor_data.accelerometer.x, imu_sensor_data.accelerometer.y, imu_sensor_data.accelerometer.z,
                imu_sensor_data.gyroscope.x, imu_sensor_data.gyroscope.y, imu_sensor_data.gyroscope.z,
                imu_sensor_data.timestamp
                )
            logging.info(f'IMU data: {imu_data}')
        except queue.Empty:
            imu_data = None

        try:
            gnss_sensor_data = self._retrieve_data(self._gnss_queue, frame, timeout=0.01)
            location = self._geo2location.transform(
                carla.GeoLocation(gnss_sensor_data.latitude, gnss_sensor_data.longitude, gnss_sensor_data.altitude))
            gnss_data = (location.x, location.y, location.z, gnss_sensor_data.timestamp)
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
