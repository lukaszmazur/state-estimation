#!/usr/bin/env python

import carla

import random
import time
import logging

import numpy as np
import matplotlib.pyplot as plt


class Plotter():
    def __init__(self):
        pass

    @staticmethod
    def plot_ground_truth_and_gnss(gt_data, gnss_data):
        # logging.info(f'gt: \n{gt_data}')
        # logging.info(f'gnss: \n{gnss_data}')
        est_traj_fig = plt.figure(figsize=(18, 12))
        ax = est_traj_fig.add_subplot(111, projection='3d')
        ax.plot(gnss_data[:, 0], gnss_data[:, 1], gnss_data[:, 2], label='GNSS')
        ax.plot(gt_data[:, 0], gt_data[:, 1], gt_data[:, 2], label='Ground Truth')
        ax.set_xlabel('Easting [m]')
        ax.set_ylabel('Northing [m]')
        ax.set_zlabel('Up [m]')
        ax.set_title('Ground Truth and GNSS')
        ax.legend(loc=(0.62,0.77))
        ax.view_init(elev=45, azim=-50)
        plt.show()

    @staticmethod
    def plot_imu_data(imu_data):
        logging.info(f'imu: \n{imu_data}')
        imu_data_fig = plt.figure(figsize=(36, 12))
        ax_a_x = imu_data_fig.add_subplot(231)
        ax_a_x.plot(imu_data[:, 0])
        ax_a_x.set_title('Acceleration x-axis')
        ax_a_y = imu_data_fig.add_subplot(232)
        ax_a_y.plot(imu_data[:, 1])
        ax_a_y.set_title('Acceleration y-axis')
        ax_a_z = imu_data_fig.add_subplot(233)
        ax_a_z.plot(imu_data[:, 2])
        ax_a_z.set_title('Acceleration z-axis')
        ax_w_x = imu_data_fig.add_subplot(234)
        ax_w_x.plot(imu_data[:, 3])
        ax_w_x.set_title('Angular velocity x-axis')
        ax_w_y = imu_data_fig.add_subplot(235)
        ax_w_y.plot(imu_data[:, 4])
        ax_w_y.set_title('Angular velocity y-axis')
        ax_w_z = imu_data_fig.add_subplot(236)
        ax_w_z.plot(imu_data[:, 5])
        ax_w_z.set_title('Angular velocity z-axis')
        plt.show()

    @staticmethod
    def online_plot_figure(data):
        with plt.ion():
            plt.figure(figsize=(9, 3))

            for i in range(20):
                logging.info(f'plotting loop: iteration {i}')
                
                plt.subplot(121)
                x_pos = [location[2][0] for location in data]
                y_pos = [location[2][1] for location in data]

                plt.plot(x_pos, y_pos, 'b-')
                # plt.show(block=False)
                plt.draw()
                # fig.canvas.draw()
                plt.pause(0.5)

    @staticmethod
    def print_data(data):
        print('-'*30)
        print(data)
        print('-'*30)
# from https://github.com/lian999111/carla-semantic-localization/blob/c4844f2f6b8bbc21c8e3e4962954cf01eb673e85/carlasim/data_collect.py
class Geo2Location(object):
    """
    Helper class for homogeneous transform from geolocation

    This class is used by GNSS class to transform from carla.GeoLocation to carla.Location.
    This transform is not provided by Carla, but it can be solved using 4 chosen points.
    Note that carla.Location is in the left-handed coordinate system.
    """

    def __init__(self, carla_map):
        """ Constructor method """
        self._map = carla_map
        # Pick 4 points of carla.Location
        loc1 = carla.Location(0, 0, 0)
        loc2 = carla.Location(1, 0, 0)
        loc3 = carla.Location(0, 1, 0)
        loc4 = carla.Location(0, 0, 1)
        # Get the corresponding carla.GeoLocation points using carla's transform_to_geolocation()
        geoloc1 = self._map.transform_to_geolocation(loc1)
        geoloc2 = self._map.transform_to_geolocation(loc2)
        geoloc3 = self._map.transform_to_geolocation(loc3)
        geoloc4 = self._map.transform_to_geolocation(loc4)
        # Solve the transform from geolocation to location (geolocation_to_location)
        l = np.array([[loc1.x, loc2.x, loc3.x, loc4.x],
                      [loc1.y, loc2.y, loc3.y, loc4.y],
                      [loc1.z, loc2.z, loc3.z, loc4.z],
                      [1, 1, 1, 1]], dtype=np.float)
        g = np.array([[geoloc1.latitude, geoloc2.latitude, geoloc3.latitude, geoloc4.latitude],
                      [geoloc1.longitude, geoloc2.longitude,
                          geoloc3.longitude, geoloc4.longitude],
                      [geoloc1.altitude, geoloc2.altitude,
                          geoloc3.altitude, geoloc4.altitude],
                      [1, 1, 1, 1]], dtype=np.float)
        # Tform = (G*L^-1)^-1
        self._tform = np.linalg.inv(g.dot(np.linalg.inv(l)))

    def transform(self, geolocation):
        """
        Transform from carla.GeoLocation to carla.Location (left_handed z-up).

        Numerical error may exist. Experiments show error is about under 1 cm in Town03.
        """
        geoloc = np.array(
            [geolocation.latitude, geolocation.longitude, geolocation.altitude, 1])
        loc = self._tform.dot(geoloc.T)
        return carla.Location(loc[0], loc[1], loc[2])

    def get_matrix(self):
        """ Get the 4-by-4 transform matrix """
        return self._tform

class RingBuffer():
    """
    Base class for ring data buffers.
    """
    def __init__(self, element_size, buffer_size):
        self._buffer_size = buffer_size
        self._data = np.zeros((self._buffer_size, element_size))
        self._number_of_elements_in_buffer = 0

    def insert_element(self, element):
        if (self._number_of_elements_in_buffer < self._buffer_size):
            self._data[self._number_of_elements_in_buffer, :] = element
            self._number_of_elements_in_buffer += 1
        else:
            # TODO: check for more efficient options
            self._data = np.roll(self._data, -1, axis=0)
            self._data[-1, :] = element
    
    def get_data(self):
        return self._data[:self._number_of_elements_in_buffer]

class GnssDataBuffer(RingBuffer):
    """
    Class storing GNSS data from measurements and transforming GeoLocation to Location.
    """
    def __init__(self, carla_map):
        # storing just x, y, z location after transformation for each measurement
        super().__init__(element_size=3, buffer_size=1000)
        self._geo2location = Geo2Location(carla_map)

    def on_measurement(self, gnss_data):
        location = self._geo2location.transform(
                carla.GeoLocation(gnss_data.latitude, gnss_data.longitude, gnss_data.altitude))
        logging.debug(f'GnssDataBuffer: received GNSS measurement with location {location}')
        data_array = np.array([location.x, location.y, location.z])
        self.insert_element(data_array)
        logging.debug(f'GnssDataBuffer: elements in buffer: {self._number_of_elements_in_buffer}')
        logging.debug(f'GnssDataBuffer: data: \n{self._data}')

class ImuDataBuffer(RingBuffer):
    """
    Class storing IMU data from measurements.
    """
    def __init__(self):
        # storing acceleration (3d) and angular velocity (3d)
        super().__init__(element_size=6, buffer_size=1000)

    def on_measurement(self, imu_data):
        logging.debug(f'ImuDataBuffer: received IMU measurement with data {imu_data}')
        data_array = np.array([
            imu_data.accelerometer.x, imu_data.accelerometer.y, imu_data.accelerometer.z,
            imu_data.gyroscope.x, imu_data.gyroscope.y, imu_data.gyroscope.z
            ])
        self.insert_element(data_array)
        logging.debug(f'ImuDataBuffer: elements in buffer: {self._number_of_elements_in_buffer}')
        logging.debug(f'ImuDataBuffer: data: \n{self._data}')

class GroundTruthBuffer(RingBuffer):
    """
    Class storing location of ego vehicle directly from the simulator.
    """
    def __init__(self, ego_vehicle_id):
        # storing all information about the agent from the snapshot
        super().__init__(element_size=15, buffer_size=1000)
        self._ego_vehicle_id = ego_vehicle_id

    def on_world_tick(self, snapshot):
            if not snapshot.has_actor(self._ego_vehicle_id):
                return
            actor_snapshot = snapshot.find(self._ego_vehicle_id)
            logging.debug(f'GroundTruthBuffer: received snapshot (frame: {snapshot.frame}, elapsed: {snapshot.timestamp.elapsed_seconds})')

            data_array = np.array([
                actor_snapshot.get_transform().location.x, actor_snapshot.get_transform().location.y, actor_snapshot.get_transform().location.z,
                actor_snapshot.get_transform().rotation.pitch, actor_snapshot.get_transform().rotation.yaw, actor_snapshot.get_transform().rotation.roll,
                actor_snapshot.get_velocity().x, actor_snapshot.get_velocity().y, actor_snapshot.get_velocity().z,
                actor_snapshot.get_angular_velocity().x, actor_snapshot.get_angular_velocity().y, actor_snapshot.get_angular_velocity().z,
                actor_snapshot.get_acceleration().x, actor_snapshot.get_acceleration().y, actor_snapshot.get_acceleration().z
                ])
            
            self.insert_element(data_array)
            logging.debug(f'GroundTruthBuffer: elements in buffer: {self._number_of_elements_in_buffer}')


def main():

    logging.basicConfig(format='%(levelname)s: %(funcName)s: %(message)s', level=logging.INFO)

    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)

    try:
        # load specific map and set fixed time-step to reliably collect data
        # from the simulation
        # world = client.load_world('Town02')

        # get existing world, leave map changing to config script
        world = client.get_world()

        settings = world.get_settings()
        seconds_per_tick = 0.05
        settings.fixed_delta_seconds = seconds_per_tick
        world.apply_settings(settings)

        blueprint_library = world.get_blueprint_library()

        # spawn ego vehicle
        ego_bp = blueprint_library.find('vehicle.tesla.model3')
        ego_transform = random.choice(world.get_map().get_spawn_points())
        ego_vehicle = world.spawn_actor(ego_bp, ego_transform)
        logging.info(f'created ego vehicle {ego_vehicle.type_id} with id {ego_vehicle.id}')

        # collect ground truth location, etc.
        gt_buffer = GroundTruthBuffer(ego_vehicle.id)
        world.on_tick(lambda snapshot: gt_buffer.on_world_tick(snapshot))

        # place spectator on ego position
        spectator = world.get_spectator()
        world_snapshot = world.wait_for_tick()  # TODO: is this needed?
        spectator.set_transform(ego_vehicle.get_transform())

        # enable autopilot for ego vehicle
        ego_vehicle.set_autopilot(True)

        # create IMU sensor
        imu_bp = blueprint_library.find('sensor.other.imu')
        imu_bp.set_attribute('sensor_tick', '0.1')
        # TODO: check relative location
        imu_transform = carla.Transform(carla.Location(x=0.0, y=0.0, z=0.0))
        imu = world.spawn_actor(imu_bp, imu_transform, attach_to=ego_vehicle, attachment_type=carla.AttachmentType.Rigid)
        logging.info('created %s' % imu.type_id)
        imu_data_buffer = ImuDataBuffer()
        imu.listen(lambda data: imu_data_buffer.on_measurement(data))

        # create GNSS sensor
        gnss_bp = blueprint_library.find('sensor.other.gnss')
        gnss_bp.set_attribute('sensor_tick', '1.0')
        # TODO: check relative location
        gnss_transform = carla.Transform(carla.Location(x=0.0, y=0.0, z=0.0))
        gnss = world.spawn_actor(gnss_bp, gnss_transform, attach_to=ego_vehicle, attachment_type=carla.AttachmentType.Rigid)
        logging.info('created %s' % gnss.type_id)
        gnss_data_buffer = GnssDataBuffer(world.get_map())
        gnss.listen(lambda data: gnss_data_buffer.on_measurement(data))

        # wait for some time to collect data
        simulation_timeout_seconds = 10
        timeout_ticks = int(simulation_timeout_seconds / seconds_per_tick)
        logging.info(f'waiting for {simulation_timeout_seconds} seconds ({timeout_ticks} ticks)')

        for _ in range(timeout_ticks):
            world.wait_for_tick()

        logging.info('plotting results')
        Plotter.plot_ground_truth_and_gnss(gt_buffer.get_data(), gnss_data_buffer.get_data())
        Plotter.plot_imu_data(imu_data_buffer.get_data())

    finally:
        logging.info('destroying actors')
        imu.stop()
        imu.destroy()
        gnss.stop()
        gnss.destroy()
        ego_vehicle.destroy()

    logging.info('done')

if __name__ == '__main__':

    main()