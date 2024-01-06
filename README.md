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
- try to implement live plotter

Other:
- refactor Kalman filter into separate class (StateEstimator)
- create Github repo

- extend buffered data with timestamp/frame
- transform measurement data to fit state estimation
- add state estimation algorithm
- add measurement noises
- change sensor location relative to ego vehicle
- add LIDAR with transformation to get localization