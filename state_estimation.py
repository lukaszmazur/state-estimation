#!/usr/bin/env python

import carla

import random
import time
import logging

import matplotlib.pyplot as plt


class Plotter():
    def __init__(self):
        pass

    def plot_figure(self, data):
        with plt.ion():
            plt.figure(figsize=(9, 3))
            for i in range(20):
                print(f'plotting loop: iteration {i}')
                
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


def main():

    logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)

    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)

    actor_list = []

    try:
        # load specific map and set fixed time-step to reliably collect data
        # from the simulation
        # world = client.load_world('Town02')

        # get existing world, leave map changing to config script
        world = client.get_world()

        settings = world.get_settings()
        settings.fixed_delta_seconds = 0.05
        world.apply_settings(settings)

        blueprint_library = world.get_blueprint_library()

        # spawn ego vehicle
        ego_bp = blueprint_library.find('vehicle.tesla.model3')
        ego_transform = random.choice(world.get_map().get_spawn_points())
        vehicle = world.spawn_actor(ego_bp, ego_transform)
        actor_list.append(vehicle)
        logging.info(f'created ego vehicle {vehicle.type_id} with id {vehicle.id}')

        # collect ground truth location, etc.
        ego_vehicle_snapshots = []
        def on_world_tick(snapshot):
            if not snapshot.has_actor(vehicle.id):
                # print(f'WORLD TICK: actor with id {vehicle.id} not present')
                return
            actor_snapshot = snapshot.find(vehicle.id)
            ego_vehicle_snapshots.append((
                snapshot.frame,
                snapshot.timestamp.elapsed_seconds,
                (actor_snapshot.get_transform().location.x, actor_snapshot.get_transform().location.y, actor_snapshot.get_transform().location.z),
                (actor_snapshot.get_transform().rotation.pitch, actor_snapshot.get_transform().rotation.yaw, actor_snapshot.get_transform().rotation.roll),
                (actor_snapshot.get_velocity().x, actor_snapshot.get_velocity().y, actor_snapshot.get_velocity().z),
                (actor_snapshot.get_angular_velocity().x, actor_snapshot.get_angular_velocity().y, actor_snapshot.get_angular_velocity().z),
                (actor_snapshot.get_acceleration().x, actor_snapshot.get_acceleration().y, actor_snapshot.get_acceleration().z)
            ))
            # print(f'WORLD TICK: actor snapshot: {ego_vehicle_snapshots[-1]}')
        world.on_tick(on_world_tick)

        # place spectator on ego position
        spectator = world.get_spectator()
        world_snapshot = world.wait_for_tick()  # TODO: is this needed?
        spectator.set_transform(vehicle.get_transform())

        # enable autopilot for ego vehicle
        vehicle.set_autopilot(True)

        # # vehicle_locations = []

        # imu_bp = blueprint_library.find('sensor.other.imu')
        # imu_bp.set_attribute('sensor_tick', '0.1')
        # # TODO: check relative location
        # imu_transform = carla.Transform(carla.Location(x=0.0, y=0.0, z=0.0))
        # imu = world.spawn_actor(imu_bp, imu_transform, attach_to=vehicle)
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

        # gnss_bp = blueprint_library.find('sensor.other.gnss')
        # gnss_bp.set_attribute('sensor_tick', '1.0')
        # # TODO: check relative location
        # gnss_transform = carla.Transform(carla.Location(x=0.0, y=0.0, z=0.0))
        # gnss = world.spawn_actor(gnss_bp, gnss_transform, attach_to=vehicle)
        # actor_list.append(gnss)
        # print('created %s' % gnss.type_id)
        # gnss_measurements = []
        # gnss.listen(lambda measurement: gnss_measurements.append(measurement))

        for i in range(5):
            Plotter.print_data(ego_vehicle_snapshots)
            time.sleep(1)

        # time.sleep(5)

    finally:

        print('destroying actors')
        imu.destroy()
        gnss.destroy()
        client.apply_batch([carla.command.DestroyActor(x) for x in actor_list])
        print('done.')

if __name__ == '__main__':

    main()