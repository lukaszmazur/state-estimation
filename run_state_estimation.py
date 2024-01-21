#!/usr/bin/env python

import carla

import os
import random
import logging
from argparse import ArgumentParser

from state_estimator import StateEstimator
from sensor_receiver import SensorReceiver
from live_plotter import LivePlotterProcess


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

    def put_to_plotter_queue(est_data, gt_data):
        p_est, v_est, q_est, p_cov, est_t = est_data
        est_x, est_y, est_z = p_est

        gt_x, gt_y, gt_z, *_, gt_t = gt_data

        plotter_queue = plotter_process.get_queue()
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
        # synchronous mode must be also set for Traffic Manager
        tm = client.get_trafficmanager()
        tm.set_synchronous_mode(True)
        world.apply_settings(settings)
        world.tick()

        blueprint_library = world.get_blueprint_library()

        # spawn ego vehicle
        ego_bp = blueprint_library.find('vehicle.tesla.model3')
        ego_transform = random.choice(world.get_map().get_spawn_points())
        ego_vehicle = world.spawn_actor(ego_bp, ego_transform)
        logging.info(f'created ego vehicle {ego_vehicle.type_id} with id {ego_vehicle.id}')

        # enable autopilot for ego vehicle
        ego_vehicle.set_autopilot(True)

        # wait a few seconds before starting collecting measurements
        simulation_timeout_seconds = 10
        timeout_ticks = int(simulation_timeout_seconds / seconds_per_tick)
        logging.info(f'waiting for {simulation_timeout_seconds} seconds ({timeout_ticks} ticks)')

        for _ in range(timeout_ticks):
            world.tick()

        # collect ground truth location, etc.
        sensor_receiver = SensorReceiver(world.get_map(), ego_vehicle.id)
        gt_queue = sensor_receiver.get_gt_queue()
        world.on_tick(gt_queue.put)

        # place spectator on ego position
        spectator = world.get_spectator()
        transform = ego_vehicle.get_transform()
        spectator.set_transform(carla.Transform(transform.location + carla.Location(z=50),
                                                carla.Rotation(pitch=-90)))

        # create Kalman filter
        state_estimator = StateEstimator()

        # create IMU sensor
        imu_bp = blueprint_library.find('sensor.other.imu')
        imu_bp.set_attribute('sensor_tick', '0.05')
        # TODO: check relative location
        imu_transform = carla.Transform(carla.Location(x=0.0, y=0.0, z=0.0))
        imu = world.spawn_actor(imu_bp, imu_transform, attach_to=ego_vehicle, attachment_type=carla.AttachmentType.Rigid)
        logging.info('created %s' % imu.type_id)
        imu_queue = sensor_receiver.get_imu_queue()
        imu.listen(imu_queue.put)

        # create GNSS sensor
        gnss_bp = blueprint_library.find('sensor.other.gnss')
        gnss_bp.set_attribute('sensor_tick', '0.5')
        # TODO: check relative location
        gnss_transform = carla.Transform(carla.Location(x=0.0, y=0.0, z=0.0))
        gnss = world.spawn_actor(gnss_bp, gnss_transform, attach_to=ego_vehicle, attachment_type=carla.AttachmentType.Rigid)
        logging.info('created %s' % gnss.type_id)
        gnss_queue = sensor_receiver.get_gnss_queue()
        gnss.listen(gnss_queue.put)

        # wait for some time to collect data
        simulation_timeout_seconds = 120
        timeout_ticks = int(simulation_timeout_seconds / seconds_per_tick)
        logging.info(f'waiting for {simulation_timeout_seconds} seconds ({timeout_ticks} ticks)')

        for _ in range(timeout_ticks):
            frame = world.tick()
            gt_data, imu_data, gnss_data = sensor_receiver.retrieve_data(frame)

            if gt_data and not state_estimator.is_initialized():
                # simplification, initialize based on the ground truth
                state_estimator.initialize_state(gt_data)

            if imu_data:
                state_estimator.on_imu_measurement(imu_data)

            if gnss_data:
                state_estimator.on_gnss_measurement(gnss_data)

            est_data = state_estimator.get_estimates()
            put_to_plotter_queue(est_data, gt_data)

            spectator_transform = carla.Transform(carla.Location(x=gt_data[0], y=gt_data[1], z=gt_data[2]+2.5),
                                                  carla.Rotation(roll=gt_data[3], pitch=gt_data[4]-20, yaw=gt_data[5]))
            client.apply_batch_sync([carla.command.ApplyTransform(spectator.id, spectator_transform)])


    finally:
        logging.info('terminating plotter')
        plotter_process.terminate()

        # disable sync mode before the script ends to prevent the server blocking
        logging.info('disabling synchronous mode')
        settings = world.get_settings()
        settings.synchronous_mode = False
        tm.set_synchronous_mode(False)
        world.apply_settings(settings)
        world.tick()

        logging.info('destroying actors')
        imu.stop()
        imu.destroy()
        gnss.stop()
        gnss.destroy()
        ego_vehicle.destroy()

    logging.info('='*20 + ' FINISHED ' + '='*20)

if __name__ == '__main__':

    try:
        main()
    except KeyboardInterrupt:
        logging.info('Cancelled by user. Bye!')
