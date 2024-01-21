import logging
import numpy as np
from scipy.spatial.transform import Rotation as R

from utils import skew_symmetric


class StateEstimator():
    """
    Error State Extended Kalman Filter (ES-EKF) Solver.
    """
    def __init__(self):
        self._previous_prediction_timestamp = 0.0

        self._var_imu_f = 0.1 # 0.10
        self._var_imu_w = 0.1 # 0.25
        self._var_gnss  = 0.1 # 0.01

        self._g = np.array([0, 0, -9.81])  # gravity
        self._l_jac = np.zeros([9, 6])
        self._l_jac[3:, :] = np.eye(6)  # motion model noise jacobian
        self._h_jac = np.zeros([3, 9])
        self._h_jac[:, :3] = np.eye(3)  # measurement model jacobian

        self._q_var_const = np.eye(6)
        self._q_var_const[:3,:] *= self._var_imu_f**2  # SOLUTION: this was not squared
        self._q_var_const[3:,:] *= self._var_imu_w**2  # SOLUTION: this was not squared

        # state estimates
        self._p_est = np.zeros(3)  # position estimates
        self._v_est = np.zeros(3)  # velocity estimates
        self._q_est = R.from_euler('xyz', np.array([0., 0., 0.])).as_quat() # orientation estimates as quaternions
        self._p_cov = np.zeros((9, 9))  # covariance matrix
        self._timestamp = 0.0

        self._is_initialized = False

    def get_estimates(self):
        return self._p_est, self._v_est, self._q_est, self._p_cov, self._timestamp

    def is_initialized(self):
        return self._is_initialized

    def initialize_state(self, gt_data):
        """
        Set initial values for estimates based on the ground truth.
        """
        logging.info(f'initializing state estimator with {gt_data}')
        gt_p0 = np.array(gt_data[0:3])
        gt_r0 = np.array(gt_data[3:6])
        gt_v0 = np.array(gt_data[6:9])

        self._p_est = gt_p0
        self._v_est = gt_v0
        self._q_est = R.from_euler('xyz', np.array([gt_r0[0], gt_r0[1], gt_r0[2]]), degrees=True).as_quat()

        self._is_initialized = True

    def on_imu_measurement(self, imu_data):
        imu_f = np.array(imu_data[0:3])
        imu_w = np.array(imu_data[3:6])
        imu_t = np.array(imu_data[6])
        self.state_prediction(imu_t, imu_f, imu_w)

    def on_gnss_measurement(self, gnss_data):
        gnss = np.array(gnss_data[:3])
        gnss_t = np.array(gnss_data[3])
        self.state_correction(gnss, gnss_t)

    def state_prediction(self, imu_t, imu_f, imu_w):
        """
        State update based on IMU measurement.
        """
        logging.info('performing state prediction')
        # skip initial measurement when timestamp is not set
        if self._previous_prediction_timestamp == 0.0:
            self._previous_prediction_timestamp = imu_t
            return
        delta_t = imu_t - self._previous_prediction_timestamp
        self._previous_prediction_timestamp = imu_t

        # 1. Update state with IMU inputs
        c_ns = R.from_quat(self._q_est).as_matrix()
        rotated_imu_acc = c_ns.dot(imu_f)
        acceleration = rotated_imu_acc + self._g

        p_check = self._p_est + delta_t * self._v_est + ((delta_t ** 2) / 2) * acceleration
        v_check = self._v_est + delta_t * acceleration

        delta_angles = imu_w * delta_t
        delta_rotation = R.from_euler('xyz', delta_angles)
        q_check = (delta_rotation * R.from_quat(self._q_est)).as_quat()

        # 1.1 Linearize the motion model and compute Jacobians
        f_jac = np.eye(9)
        f_jac[:3, 3:6] = delta_t * np.eye(3)
        f_jac[3:6, 6:] = -delta_t * skew_symmetric(rotated_imu_acc)

        # 2. Propagate uncertainty
        q_var = (delta_t**2) * self._q_var_const
        p_cov_check = f_jac @ self._p_cov @ f_jac.T + self._l_jac @ q_var @ self._l_jac.T

        # Update states (save)
        self._p_est = p_check
        self._v_est = v_check
        self._q_est = q_check
        self._p_cov = p_cov_check
        self._timestamp = imu_t

    def state_correction(self, y_k, gnss_t):
        """
        Correct predicted state using GNSS measurement.
        """
        logging.info('initializing state correction')
        sensor_var = self._var_gnss
        h_jac = self._h_jac

        p_cov_check = self._p_cov
        p_check = self._p_est
        v_check = self._v_est
        q_check = self._q_est

        # 3.1 Compute Kalman Gain
        r_cov = np.eye(3) * (sensor_var**2)  # SOLUTION: sensor var is not squared
        k_gain = p_cov_check @ h_jac.T @ np.linalg.inv(h_jac @ p_cov_check @ h_jac.T + r_cov)  # 9x3

        # 3.2 Compute error state
        delta_x = k_gain @ (y_k - p_check)  # 9x1

        # 3.3 Correct predicted state
        p_hat = p_check + delta_x[:3]
        v_hat = v_check + delta_x[3:6]

        delta_angles_r = R.from_euler('xyz', delta_x[6:])
        q_hat = (delta_angles_r * R.from_quat(q_check)).as_quat()

        # 3.4 Compute corrected covariance
        p_cov_hat = (np.eye(9) - k_gain @ h_jac) @ p_cov_check

        # Update states (save)
        self._p_est = p_hat
        self._v_est = v_hat
        self._q_est = q_hat
        self._p_cov = p_cov_hat
        self._timestamp = gnss_t
