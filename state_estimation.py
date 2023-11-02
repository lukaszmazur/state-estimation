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
        gt_x = [location[2][0] for location in gt_data]
        gt_y = [location[2][1] for location in gt_data]
        gt_z = [location[2][2] for location in gt_data]

        logging.info(f'gnss: {gnss_data}')
        # gnss_x = [x[1].x for x in gnss_data]
        # gnss_y = [x[1].y for x in gnss_data]
        # gnss_z = [x[1].z for x in gnss_data]

        est_traj_fig = plt.figure(figsize=(18, 12))
        ax = est_traj_fig.add_subplot(111, projection='3d')
        ax.plot(gnss_data[:, 0], gnss_data[:, 1], gnss_data[:, 2], label='GNSS')
        ax.plot(gt_x, gt_y, gt_z, label='Ground Truth')
        ax.set_xlabel('Easting [m]')
        ax.set_ylabel('Northing [m]')
        ax.set_zlabel('Up [m]')
        # ax.set_title('Ground Truth and Estimated Trajectory')
        ax.set_title('Ground Truth')
        # ax.set_xlim(0, 200)
        # ax.set_ylim(0, 200)
        # ax.set_zlim(-2, 2)
        # ax.set_xticks([0, 50, 100, 150, 200])
        # ax.set_yticks([0, 50, 100, 150, 200])
        # ax.set_zticks([-2, -1, 0, 1, 2])
        ax.legend(loc=(0.62,0.77))
        ax.view_init(elev=45, azim=-50)
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

class GnssDataBuffer():
    """
    Class storing GNSS data from measurements and transforming GeoLocation to Location.
    """
    def __init__(self, carla_map):
        self.BUFFER_SIZE = 1000
        # storing just x, y, z location after transformation for each measurement
        self._data = np.zeros((self.BUFFER_SIZE, 3))
        self._number_of_elements_in_buffer = 0
        self._geo2location = Geo2Location(carla_map)

    def on_measurement(self, gnss_data):
        location = self._geo2location.transform(
                carla.GeoLocation(gnss_data.latitude, gnss_data.longitude, gnss_data.altitude))
        logging.debug(f'GnssDataBuffer: received GNSS measurement with location {location}')
        location_array = np.array([location.x, location.y, location.z])
        if (self._number_of_elements_in_buffer < self.BUFFER_SIZE):
            # logging.debug('GnssDataBuffer: buffer not yet full, adding elements inside')
            self._data[self._number_of_elements_in_buffer, :] = location_array
            self._number_of_elements_in_buffer += 1
        else:
            # logging.debug('GnssDataBuffer: buffer full, rolling elements')
            # TODO: check for more efficient options
            self._data = np.roll(self._data, -1, axis=0)
            self._data[-1, :] = location_array
        logging.debug(f'GnssDataBuffer: elements in buffer: {self._number_of_elements_in_buffer}')
        logging.debug(f'GnssDataBuffer: data: \n{self._data}')

    def get_data(self):
        return self._data[:self._number_of_elements_in_buffer]


def main():

    logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)

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
        ego_vehicle_snapshots = []
        def on_world_tick(snapshot):
            if not ego_vehicle.is_alive:
                return
            if not snapshot.has_actor(ego_vehicle.id):
                return
            actor_snapshot = snapshot.find(ego_vehicle.id)
            ego_vehicle_snapshots.append((
                snapshot.frame,
                snapshot.timestamp.elapsed_seconds,
                (actor_snapshot.get_transform().location.x, actor_snapshot.get_transform().location.y, actor_snapshot.get_transform().location.z),
                (actor_snapshot.get_transform().rotation.pitch, actor_snapshot.get_transform().rotation.yaw, actor_snapshot.get_transform().rotation.roll),
                (actor_snapshot.get_velocity().x, actor_snapshot.get_velocity().y, actor_snapshot.get_velocity().z),
                (actor_snapshot.get_angular_velocity().x, actor_snapshot.get_angular_velocity().y, actor_snapshot.get_angular_velocity().z),
                (actor_snapshot.get_acceleration().x, actor_snapshot.get_acceleration().y, actor_snapshot.get_acceleration().z)
            ))
        world.on_tick(on_world_tick)

        # place spectator on ego position
        spectator = world.get_spectator()
        world_snapshot = world.wait_for_tick()  # TODO: is this needed?
        spectator.set_transform(ego_vehicle.get_transform())

        # enable autopilot for ego vehicle
        ego_vehicle.set_autopilot(True)

        # # vehicle_locations = []

        # imu_bp = blueprint_library.find('sensor.other.imu')
        # imu_bp.set_attribute('sensor_tick', '0.1')
        # # TODO: check relative location
        # imu_transform = carla.Transform(carla.Location(x=0.0, y=0.0, z=0.0))
        # imu = world.spawn_actor(imu_bp, imu_transform, attach_to=ego_vehicle)
        # actor_list.append(imu)
        # print('created %s' % imu.type_id)
        # imu_measurements = []

        # def imu_callback(measurement):
        #     imu_measurements.append(measurement)
        #     # snapshot = world.get_snapshot()
        #     # vehicle_locations.append((snapshot.frame, snapshot.timestamp.elapsed_seconds,
        #     #                           vehicle.get_location(), vehicle.get_velocity(), vehicle.get_acceleration()))
        #     # print(f'IMU measurement: {measurement} \nvehicle location: {vehicle_locations[-1]}\n')

        # imu.listen(imu_callback)

        gnss_bp = blueprint_library.find('sensor.other.gnss')
        gnss_bp.set_attribute('sensor_tick', '1.0')
        # TODO: check relative location
        gnss_transform = carla.Transform(carla.Location(x=0.0, y=0.0, z=0.0))
        gnss = world.spawn_actor(gnss_bp, gnss_transform, attach_to=ego_vehicle, attachment_type=carla.AttachmentType.Rigid)
        print('created %s' % gnss.type_id)
        gnssDataBuffer = GnssDataBuffer(world.get_map())
        gnss.listen(lambda data: gnssDataBuffer.on_measurement(data))

        simulation_timeout_seconds = 30
        timeout_ticks = int(simulation_timeout_seconds / seconds_per_tick)
        logging.info(f'waiting for {simulation_timeout_seconds} seconds ({timeout_ticks} ticks)')

        for _ in range(timeout_ticks):
            world.wait_for_tick()

        logging.info('plotting results')
        Plotter.plot_ground_truth_and_gnss(ego_vehicle_snapshots, gnssDataBuffer.get_data())

    finally:
        logging.info('destroying actors')
        # imu.destroy()
        gnss.stop()
        gnss.destroy()
        ego_vehicle.destroy()

    logging.info('done')

if __name__ == '__main__':

    main()