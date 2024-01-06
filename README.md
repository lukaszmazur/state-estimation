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
- [ ] make live plotters non-blocking

Kalman filter:
- [ ] refactor Kalman filter into separate class (StateEstimator)
- [ ] make Kalman filter "online"

Simulation setup:
- [ ] add measurement noises
- [ ] change sensor location relative to ego vehicle
- [ ] add LIDAR with transformation to get localization

General:
- [ ] create Github repo
