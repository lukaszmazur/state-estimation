This folder contains virtual environment for operating CARLA simulator (system wide Debian installation).

carla-simulator (server, version **0.9.13**) is installed in `/opt/carla-simulator`

carla (client, version **0.9.13**)  in installed in this virtual environment

Dependencies for examples (`/opt/carla-simulator/PythonAPI/examples/requirements.txt`).


# TODOs
Live plotting of state estimation:
- [x] read basic matplotlib tutorials
- [x] ask ChatGPT about live plotting using matplotlib
- [x] read saved articles
- [x] new search
- [x] try to implement live plotter
- [x] make live plotters non-blocking - move to separate process
    - [x] https://stackoverflow.com/questions/51949185/non-blocking-matplotlib-animation

Kalman filter:
- [ ] refactor Kalman filter into separate class (StateEstimator)
- [ ] implement proper (thread safe) data retrieval from sensors
    - [ ] read: https://carla.readthedocs.io/en/0.9.13/adv_synchrony_timestep/
    - [ ] read: https://github.com/carla-simulator/carla/blob/master/PythonAPI/examples/synchronous_mode.py
- [ ] make Kalman filter "online"

Simulation setup:
- [ ] add measurement noises
- [ ] change sensor location relative to ego vehicle
- [ ] add LIDAR with transformation to get localization

General:
- [x] create Github repo
