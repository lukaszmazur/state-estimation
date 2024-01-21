This folder contains virtual environment for operating CARLA simulator (system wide Debian installation).

carla-simulator (server, version **0.9.13**) is installed in `/opt/carla-simulator`

carla (client, version **0.9.13**)  in installed in this virtual environment

Dependencies for examples (`/opt/carla-simulator/PythonAPI/examples/requirements.txt`).


# TODOs
Kalman filter:
- [x] implement proper (thread safe) data retrieval from sensors
    - [x] read: https://carla.readthedocs.io/en/0.9.13/adv_synchrony_timestep/
    - [x] read: https://github.com/carla-simulator/carla/blob/master/PythonAPI/examples/synchronous_mode.py
- [x] make Kalman filter "online"
- [x] initialize Kalman filter based on Ground Truth

Plotting:
- [x] set minima axis span (esp. for position Z)
- [ ] __plot more data (orientation, velocity, 3D, 2D XY plane, etc.)__

Simulation setup:
- [ ] add measurement noises
- [ ] change sensor location relative to ego vehicle
- [ ] add LIDAR with transformation to get localization
- [x] make spectator follow the ego vehicle

General:
- [x] create Github repo
- [ ] __add license file__
- [ ] __improve this README__
- [ ] __add screenshots__
- [ ] cleanup Git history
- [ ] review entire codebase
