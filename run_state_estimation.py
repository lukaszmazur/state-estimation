#!/usr/bin/env python

import carla

import os
import random
import time
import logging
import queue
import threading
import multiprocessing as mp
import copy
from argparse import ArgumentParser

import numpy as np
from scipy.spatial.transform import Rotation as R

from state_estimator import StateEstimator, StateEstimatesBuffer
from utils import RingBuffer
from carla_utils import Geo2Location
from live_plotter import LivePlotterProcess
from plot_utils import plot_ground_truth_and_estimated, plot_ground_truth_and_estimated_3d


class GnssDataBuffer(RingBuffer):
    """
    Class storing GNSS data from measurements and transforming GeoLocation to Location.
    """
    def __init__(self, carla_map):
        # storing just x, y, z location after transformation for each measurement + timestamp
        super().__init__(element_size=4, buffer_size=200)
        self._geo2location = Geo2Location(carla_map)

    def on_measurement(self, gnss_data):
        location = self._geo2location.transform(
                carla.GeoLocation(gnss_data.latitude, gnss_data.longitude, gnss_data.altitude))
        logging.debug(f'GnssDataBuffer: received GNSS measurement with location {location}')
        data_array = np.array([location.x, location.y, location.z, gnss_data.timestamp])
        self.insert_element(data_array)
        logging.debug(f'GnssDataBuffer: elements in buffer: {self._number_of_elements_in_buffer}')
        logging.debug(f'GnssDataBuffer: data: \n{self._data}')

class ImuDataBuffer(RingBuffer):
    """
    Class storing IMU data from measurements.
    """
    def __init__(self):
        # storing acceleration (3d) and angular velocity (3d), timestamp
        super().__init__(element_size=7, buffer_size=2000)

    def on_measurement(self, imu_data):
        logging.debug(f'ImuDataBuffer: received IMU measurement with data {imu_data}, transform: {imu_data.transform}')
        data_array = np.array([
            imu_data.accelerometer.x, imu_data.accelerometer.y, imu_data.accelerometer.z,
            imu_data.gyroscope.x, imu_data.gyroscope.y, imu_data.gyroscope.z,
            imu_data.timestamp
            ])
        self.insert_element(data_array)
        logging.debug(f'ImuDataBuffer: elements in buffer: {self._number_of_elements_in_buffer}')
        logging.debug(f'ImuDataBuffer: data: \n{self._data}')

class GroundTruthBuffer(RingBuffer):
    """
    Class storing location of ego vehicle directly from the simulator.
    """
    def __init__(self, ego_vehicle_id):
        # storing all information about the agent from the snapshot + timestamp
        super().__init__(element_size=16, buffer_size=2100)
        self._ego_vehicle_id = ego_vehicle_id

    def on_world_tick(self, snapshot):
            if not snapshot.has_actor(self._ego_vehicle_id):
                return
            actor_snapshot = snapshot.find(self._ego_vehicle_id)
            logging.debug(f'GroundTruthBuffer: received snapshot (frame: {snapshot.frame}, elapsed: {snapshot.timestamp.elapsed_seconds})')

            data_array = np.array([
                actor_snapshot.get_transform().location.x, actor_snapshot.get_transform().location.y, actor_snapshot.get_transform().location.z,
                # angles (yaw, pitch, roll) correspond to Euler angles
                #actor_snapshot.get_transform().rotation.yaw, actor_snapshot.get_transform().rotation.pitch, actor_snapshot.get_transform().rotation.roll,
                actor_snapshot.get_transform().rotation.roll, actor_snapshot.get_transform().rotation.pitch, actor_snapshot.get_transform().rotation.yaw,
                actor_snapshot.get_velocity().x, actor_snapshot.get_velocity().y, actor_snapshot.get_velocity().z,
                actor_snapshot.get_angular_velocity().x, actor_snapshot.get_angular_velocity().y, actor_snapshot.get_angular_velocity().z,
                actor_snapshot.get_acceleration().x, actor_snapshot.get_acceleration().y, actor_snapshot.get_acceleration().z,
                snapshot.timestamp.elapsed_seconds
                ])

            self.insert_element(data_array)
            logging.debug(f'GroundTruthBuffer: elements in buffer: {self._number_of_elements_in_buffer}')


def main():

    arg_parser = ArgumentParser(description="State estimator demo for CARLA simulator")
    arg_parser.add_argument('--output', type=str, default='./output', help='Specify the output path for logs and figures')
    args = arg_parser.parse_args()
    output_path = args.output

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    logging.basicConfig(format='%(asctime)s %(levelname)s: %(funcName)s: %(message)s',
                        level=logging.INFO, filename=os.path.join(output_path, 'output.log'))
    logging.info('='*20 + ' STARTING ' + '='*20)

    plotter_process = LivePlotterProcess()
    plotter_process.start()

    def put_to_plotter_queue(est_buffer, gt_buffer):
        plotter_queue = plotter_process.get_queue()
        est_t = est_buffer.get_data()[:, 10][-1]
        est_x = est_buffer.get_data()[:, 0][-1]
        est_y = est_buffer.get_data()[:, 1][-1]
        est_z = est_buffer.get_data()[:, 2][-1]
        gt_t = gt_buffer.get_data()[:, 15][-1]
        gt_x = gt_buffer.get_data()[:, 0][-1]
        gt_y = gt_buffer.get_data()[:, 1][-1]
        gt_z = gt_buffer.get_data()[:, 2][-1]
        plotter_queue.put((est_t, est_x, est_y, est_z,
                           gt_t, gt_x, gt_y, gt_z))

    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)

    try:
        # load specific map and set fixed time-step to reliably collect data
        # from the simulation
        # world = client.load_world('Town02')

        # get existing world, leave map changing to config script
        world = client.get_world()
        debug = world.debug

        settings = world.get_settings()
        seconds_per_tick = 0.05
        settings.fixed_delta_seconds = seconds_per_tick
        settings.synchronous_mode = True
        world.apply_settings(settings)

        blueprint_library = world.get_blueprint_library()

        # spawn ego vehicle
        ego_bp = blueprint_library.find('vehicle.tesla.model3')
        ego_transform = random.choice(world.get_map().get_spawn_points())
        ego_vehicle = world.spawn_actor(ego_bp, ego_transform)
        logging.info(f'created ego vehicle {ego_vehicle.type_id} with id {ego_vehicle.id}')

        # enable autopilot for ego vehicle
        ego_vehicle.set_autopilot(True)

        # wait a few seconds before starting collecting measurements
        simulation_timeout_seconds = 30
        timeout_ticks = int(simulation_timeout_seconds / seconds_per_tick)
        logging.info(f'waiting for {simulation_timeout_seconds} seconds ({timeout_ticks} ticks)')

        for _ in range(timeout_ticks):
            # world.wait_for_tick()
            world.tick()

        # collect ground truth location, etc.
        gt_buffer = GroundTruthBuffer(ego_vehicle.id)
        world.on_tick(lambda snapshot: gt_buffer.on_world_tick(snapshot))

        # place spectator on ego position
        spectator = world.get_spectator()
        # world_snapshot = world.wait_for_tick()  # TODO: is this needed?
        world.tick()
        spectator.set_transform(ego_vehicle.get_transform())

        # create Kalman filter
        # es_ekf = EsEkfSolver()
        state_estimator = StateEstimator()
        est_buffer = StateEstimatesBuffer(buffer_size=2200)  # size = IMU + GNSS buffer sizes

        # create IMU sensor
        imu_bp = blueprint_library.find('sensor.other.imu')
        imu_bp.set_attribute('sensor_tick', '0.05')
        # TODO: check relative location
        imu_transform = carla.Transform(carla.Location(x=0.0, y=0.0, z=0.0))
        imu = world.spawn_actor(imu_bp, imu_transform, attach_to=ego_vehicle, attachment_type=carla.AttachmentType.Rigid)
        logging.info('created %s' % imu.type_id)
        imu_data_buffer = ImuDataBuffer()
        # imu_queue = Queue()
        mutex = threading.Lock()

        # trigger Kalman filter on IMU measurement
        def on_imu_measurement(data):
            mutex.acquire()
            imu_data_buffer.on_measurement(data)
            state_estimator.on_imu_measurement(imu_data_buffer.get_data()[-1])
            est_buffer.on_estimation_update(state_estimator.get_estimates())
            # TODO: gt_buffer is not protected by mutex
            put_to_plotter_queue(est_buffer, gt_buffer)
            mutex.release()

        # imu.listen(lambda data: imu_data_buffer.on_measurement(data))
        imu.listen(lambda data: on_imu_measurement(data))
        # imu.listen(imu_queue.put)

        # create GNSS sensor
        gnss_bp = blueprint_library.find('sensor.other.gnss')
        gnss_bp.set_attribute('sensor_tick', '0.5')
        # TODO: check relative location
        gnss_transform = carla.Transform(carla.Location(x=0.0, y=0.0, z=0.0))
        gnss = world.spawn_actor(gnss_bp, gnss_transform, attach_to=ego_vehicle, attachment_type=carla.AttachmentType.Rigid)
        logging.info('created %s' % gnss.type_id)
        gnss_data_buffer = GnssDataBuffer(world.get_map())
        # gnss_queue = Queue()

        def on_gnss_measurement(data):
            mutex.acquire()
            gnss_data_buffer.on_measurement(data)
            state_estimator.on_gnss_measurement(gnss_data_buffer.get_data()[-1])
            est_buffer.on_estimation_update(state_estimator.get_estimates())
            # TODO: gt_buffer is not protected by mutex
            put_to_plotter_queue(est_buffer, gt_buffer)
            mutex.release()

        # gnss.listen(lambda data: gnss_data_buffer.on_measurement(data))
        gnss.listen(lambda data: on_gnss_measurement(data))
        # gnss.listen(gnss_queue.put)

        # wait for some time to collect data
        simulation_timeout_seconds = 120
        timeout_ticks = int(simulation_timeout_seconds / seconds_per_tick)
        logging.info(f'waiting for {simulation_timeout_seconds} seconds ({timeout_ticks} ticks)')

        for _ in range(timeout_ticks):
        # while True:
            # world.wait_for_tick()
            world.tick()

        # offline processing of Kalman filter
        # remove first few measurements, just after sensor creation (spikes)
        collected_imu_data = imu_data_buffer.get_data()[5:]
        collected_gnss_data = gnss_data_buffer.get_data()[5:]
        collected_gt_data = gt_buffer.get_data()[5:]
        # p_est, v_est, a_est, q_est, p_cov, t_est, gt_values = es_ekf.process_data(collected_imu_data, collected_gnss_data, collected_gt_data)
        collected_est_data = est_buffer.get_data()[5:]
        p_est = collected_est_data[:, 0:3]
        v_est = collected_est_data[:, 3:6]
        q_est = collected_est_data[:, 6:10]
        t_est = collected_est_data[:, 10]

        logging.info('plotting results')
        plot_ground_truth_and_estimated(collected_gt_data, p_est, v_est, q_est, t_est, output_path)
        plot_ground_truth_and_estimated_3d(collected_gt_data, p_est, output_path)
        # plot_ground_truth_and_gnss(gt_buffer.get_data(), gnss_data_buffer.get_data())
        # plot_imu_data(imu_data_buffer.get_data())

    finally:
        logging.info('destroying actors')
        imu.stop()
        imu.destroy()
        gnss.stop()
        gnss.destroy()
        ego_vehicle.destroy()

        plotter_process.terminate()

    logging.info('='*20 + ' FINISHED ' + '='*20)

if __name__ == '__main__':

    main()
