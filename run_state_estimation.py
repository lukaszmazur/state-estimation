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
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

from state_estimator import StateEstimator, StateEstimatesBuffer
from utils import Quaternion, angle_normalize, skew_symmetric, RingBuffer
from live_plotter import LivePlotterProcess
from plot_utils import plot_ground_truth_and_estimated, plot_ground_truth_and_estimated_3d


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


class EsEkfSolver():
    """
    Error State Extended Kalman Filter (ES-EKF) Solver.
    """
    def __init__(self):
        self._previous_timestamp = 0.0

    def find_initial_values(self, timestamp, gt_data):
        """
        Looks for snapshot in ground truth data that corresponds to provided snapshot.
        """
        logging.info(f'gt_data[:10]={gt_data[:10]}')
        logging.info(f'gt_data.shape={gt_data.shape}')
        logging.info(f'timestamp={timestamp}')

        last_gt_snapshot = None
        for gt_snapshot in gt_data:
            gt_timestamp = gt_snapshot[15]
            # TODO: add more accurate criterion
            if gt_timestamp < timestamp:
                last_gt_snapshot = gt_snapshot
            else:
                logging.info(f'gt_timestamp={gt_timestamp}')
                break

        logging.info(f'last_gt_snapshot={last_gt_snapshot}')
        # return just location and velocity
        location = last_gt_snapshot[:3]
        velocity = last_gt_snapshot[6:9]
        orientation = last_gt_snapshot[3:6]
        return location, velocity, orientation
    
    def _measurement_update(self, sensor_var, p_cov_check, y_k, p_check, v_check, q_check, h_jac):
        logging.info("="*15 + " measurement update START " + "="*15)
        # 3.1 Compute Kalman Gain
        r_cov = np.eye(3) * (sensor_var**2)  # SOLUTION: sensor var is not squared
        logging.info(f'p_cov_check={p_cov_check}, h_jac={h_jac}, r_cov={r_cov}')
        k_gain = p_cov_check @ h_jac.T @ np.linalg.inv(h_jac @ p_cov_check @ h_jac.T + r_cov)  # 9x3
        logging.info(f'k_gain={k_gain}, y_k={y_k}, p_check={p_check}')

        # 3.2 Compute error state
        delta_x = k_gain @ (y_k - p_check)  # 9x1
        logging.info(f'delta_x={delta_x}')

        # 3.3 Correct predicted state
        p_hat = p_check + delta_x[:3]
        v_hat = v_check + delta_x[3:6]
        
        # q_hat = Quaternion(axis_angle=angle_normalize(delta_x[6:])).quat_mult_left(q_check)
        # q_hat = Quaternion(euler=angle_normalize(delta_x[6:])).quat_mult_left(q_check)
        delta_angles_r = R.from_euler('xyz', delta_x[6:])
        logging.info(f'delta_angles_r = {delta_angles_r.as_euler("xyz", degrees=True)}')
        q_hat = (delta_angles_r * R.from_quat(q_check)).as_quat()
        logging.info(f'q_hat = {R.from_quat(q_hat).as_euler("xyz", degrees=True)}')

        # 3.4 Compute corrected covariance
        p_cov_hat = (np.eye(9) - k_gain @ h_jac) @ p_cov_check

        logging.info("="*15 + " measurement update END " + "="*15)

        return p_hat, v_hat, q_hat, p_cov_hat

    # def on_data_change(self, data):
    def process_data(self, imu_data, gnss_data, gt_data):
        logging.info(f'imu_data.shape={imu_data.shape}')
        logging.info(f'gnss_data.shape={gnss_data.shape}')

        var_imu_f = 0.1 # 0.10
        var_imu_w = 0.1 # 0.25
        var_gnss  = 0.1 # 0.01

        g = np.array([0, 0, -9.81])  # gravity
        l_jac = np.zeros([9, 6])
        l_jac[3:, :] = np.eye(6)  # motion model noise jacobian
        h_jac = np.zeros([3, 9])
        h_jac[:, :3] = np.eye(3)  # measurement model jacobian

        q_var_const = np.eye(6)
        q_var_const[:3,:] *= var_imu_f**2  # SOLUTION: this was not squared
        q_var_const[3:,:] *= var_imu_w**2  # SOLUTION: this was not squared


        p_est = np.zeros([imu_data.shape[0], 3])  # position estimates
        v_est = np.zeros([imu_data.shape[0], 3])  # velocity estimates
        a_est = np.zeros([imu_data.shape[0], 3])  # acceleration estimates (TEMP)
        q_est = np.zeros([imu_data.shape[0], 4])  # orientation estimates as quaternions
        p_cov = np.zeros([imu_data.shape[0], 9, 9])  # covariance matrices at each timestep

        gt_p0, gt_v0, gt_r0 = self.find_initial_values(imu_data[0, 6], gt_data)

        # Set initial values.
        p_est[0] = gt_p0
        v_est[0] = gt_v0
        # q_est[0] = Quaternion(euler=gt_r0).to_numpy()
        # q_est[0] = Quaternion(axis_angle=gt_r0).to_numpy()
        
        # from snapshot we have rotation based on Unreal Engine's axis system (pitch, yaw, roll))
        q_est[0] = R.from_euler('xyz', np.array([gt_r0[0], gt_r0[1], gt_r0[2]]), degrees=True).as_quat()
        p_cov[0] = np.zeros(9)  # covariance of estimate
        gnss_i  = 0

        imu_f = imu_data[:, 0:3]
        imu_w = imu_data[:, 3:6]
        imu_t = imu_data[:, 6]

        gnss = gnss_data[:, :3]
        gnss_t = gnss_data[:, 3]

        # logging.info(f'\nimu_f={imu_f[:3]}, \nimu_w={imu_w[:3]}, \nimu_t={imu_t[:3]}')

        for k in range(1, imu_t.shape[0]):  # start at 1 b/c we have initial prediction from gt
            delta_t = imu_t[k] - imu_t[k - 1]

            # 1. Update state with IMU inputs
            # c_ns = Quaternion(*q_est[k-1]).to_mat()
            c_ns = R.from_quat(q_est[k-1]).as_matrix()
            rotated_imu_acc = c_ns.dot(imu_f[k-1])
            acceleration = rotated_imu_acc + g  # SOLUTION: uses np.dot instead of @
            # acceleration = R.from_quat(q_est[k-1]).apply(imu_f[k-1]) + g
            # print(f'acceleration from IMU:       {imu_f[k-1]}, length {np.linalg.norm(imu_f[k-1])}')
            # print(f'acceleration after rotation: {acceleration}, length {np.linalg.norm(acceleration)}')

            # logging.info(f'imu_f[k-1]={imu_f[k-1]}, acceleration={acceleration}')
            p_check = p_est[k-1] + delta_t * v_est[k-1] + ((delta_t ** 2) / 2) * acceleration
            v_check = v_est[k-1] + delta_t * acceleration
            # logging.info(f'p_check={p_check}, v_check={v_check}')

            # delta_angles = angle_normalize(imu_w[k-1] * delta_t)
            delta_angles = imu_w[k-1] * delta_t
            # q_check = Quaternion(*q_est[k-1]).quat_mult_left(Quaternion(axis_angle=delta_angles))
            # q_check = Quaternion(*q_est[k-1]).quat_mult_left(Quaternion(euler=delta_angles))
            delta_rotation = R.from_euler('xyz', delta_angles)  # should be XYZ or xyz?
            logging.info(f'delta_rotation = {delta_rotation.as_euler("xyz", degrees=True)}')
            q_check = (delta_rotation * R.from_quat(q_est[k-1])).as_quat()
            logging.info(f'q_check = {R.from_quat(q_check).as_euler("xyz", degrees=True)}')

            # 1.1 Linearize the motion model and compute Jacobians
            f_jac = np.eye(9)
            f_jac[:3, 3:6] = delta_t * np.eye(3)
            f_jac[3:6, 6:] = -delta_t * skew_symmetric(rotated_imu_acc)

            # 2. Propagate uncertainty
            q_var = (delta_t**2) * q_var_const
            p_cov_check = f_jac @ p_cov[k-1] @ f_jac.T + l_jac @ q_var @ l_jac.T

            # 3. Check availability of GNSS and LIDAR measurements

            if gnss_i < gnss_t.shape[0] and gnss_t[gnss_i] <= imu_t[k]:
                logging.info(f"GNSS measurement available at timestep {gnss_t[gnss_i]} (IMU timestep {imu_t[k]})")
                p_check, v_check, q_check, p_cov_check = self._measurement_update(
                    var_gnss, p_cov_check, gnss[gnss_i], p_check, v_check, q_check, h_jac)
                gnss_i += 1

            # Update states (save)

            p_est[k] = p_check
            v_est[k] = v_check
            a_est[k] = acceleration
            q_est[k] = q_check
            p_cov[k] = p_cov_check

        return p_est, v_est, a_est, q_est, p_cov, imu_t, (gt_p0, gt_v0, gt_r0)


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