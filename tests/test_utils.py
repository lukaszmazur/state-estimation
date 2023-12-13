import unittest
import numpy as np
from scipy.spatial.transform import Rotation as R

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from utils import Quaternion

class TestQuaternionFromToEuler(unittest.TestCase):
    def testCreateQuaternionFromEulerAngles(self):
        # alpha = np.pi / 2
        # beta = np.pi / 3
        # gamma = np.pi / 6
        # alpha = 1.57079633
        # beta = 1.04719755
        # gamma = 0.52359878
        alpha = 0.
        beta = np.pi / 2
        gamma = 0.

        input_angles = np.array([alpha, beta, gamma])
        print(f'input_angles={input_angles}')

        q = Quaternion(euler=input_angles)
        # print(f'q.to_euler()={q.to_euler()}')
        print(f'q.to_numpy()={q.to_numpy()}')

        expected_q = np.array([0.707107, 0, 0.707107, 0])

        np.testing.assert_allclose(q.to_numpy(), expected_q, rtol=1e-4)
        np.testing.assert_allclose(q.to_euler(), input_angles)

    def testQuaternionFromEuler1(self):
        print("test:\n")
        roll = np.pi / 2
        pitch = np.pi / 3
        yaw = np.pi / 6

        r = R.from_euler('XYZ', np.array([roll, pitch, yaw]))
        print(f'r.as_quat(xyzw)={r.as_quat()}')

        q1 = Quaternion(euler=np.array([roll, pitch, yaw]))
        print(f'q(euler)={q1}')

        q2 = Quaternion(axis_angle=np.array([roll, pitch, yaw]))
        print(f'q(axis)={q2}')

        print(f'r.as_matrix={r.as_matrix()}')

if __name__ == '__main__':
    unittest.main()
