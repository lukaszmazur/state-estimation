import unittest
import numpy as np
from scipy.spatial.transform import Rotation as R

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import carla
from utils import Quaternion

class TestQuaternionFromToEuler(unittest.TestCase):
    # def testCreateQuaternionFromEulerAngles(self):
    #     # alpha = np.pi / 2
    #     # beta = np.pi / 3
    #     # gamma = np.pi / 6
    #     # alpha = 1.57079633
    #     # beta = 1.04719755
    #     # gamma = 0.52359878
    #     alpha = 0.
    #     beta = np.pi / 2
    #     gamma = 0.

    #     input_angles = np.array([alpha, beta, gamma])
    #     print(f'input_angles={input_angles}')

    #     q = Quaternion(euler=input_angles)
    #     # print(f'q.to_euler()={q.to_euler()}')
    #     print(f'q.to_numpy()={q.to_numpy()}')

    #     expected_q = np.array([0.707107, 0, 0.707107, 0])

    #     np.testing.assert_allclose(q.to_numpy(), expected_q, rtol=1e-4)
    #     np.testing.assert_allclose(q.to_euler(), input_angles)

    # def testQuaternionFromEuler1(self):
    #     print("test:\n")
    #     roll = np.pi / 2
    #     pitch = np.pi / 3
    #     yaw = np.pi / 6

    #     r = R.from_euler('XYZ', np.array([roll, pitch, yaw]))
    #     print(f'r.as_quat(xyzw)={r.as_quat()}')

    #     q1 = Quaternion(euler=np.array([roll, pitch, yaw]))
    #     print(f'q(euler)={q1}')

    #     q2 = Quaternion(axis_angle=np.array([roll, pitch, yaw]))
    #     print(f'q(axis)={q2}')

    #     print(f'r.as_matrix={r.as_matrix()}')

    def testSciPyQuaternionUsage(self):
        r1 = R.from_euler('xyz', np.array([90.0, 0.0, 0.0]), degrees=True)
        r2 = R.from_euler('xyz', np.array([45.0, 0.0, 90.0]), degrees=True)

        r2_r1 = r2 * r1
        r1_r2 = r1 * r2
        print(f"r2_r1={r2_r1.as_euler('xyz', degrees=True)}")
        print(f"r1_r2={r1_r2.as_euler('xyz', degrees=True)}")
        print("\n")

    def _verifyRotationUpdateBasedOnGyroscope(self, gyroscope_prev, rotation_prev,
                                              rotation_next, delta_t):
        print(f"rotation_prev={rotation_prev}")
        print(f"rotation_next={rotation_next}")
        print(f"rotation_diff={(rotation_next.pitch-rotation_prev.pitch, rotation_next.yaw-rotation_prev.yaw, rotation_next.roll-rotation_prev.roll)}")
        print("="*15)

        delta_t = 59.485167 - 59.385167  # 0.1 s
        print(f'delta_t = {delta_t}')

        delta_angles = np.array([gyroscope_prev.x, gyroscope_prev.y, gyroscope_prev.z]) * delta_t
        print(f"delta_angles={delta_angles}")
        print(f"delta_angles(deg)={delta_angles * 180 / np.pi}")
        print("="*15)

        euler_type = 'xyz'
        delta_rotation = R.from_euler(euler_type, delta_angles)
        print(f"delta_rotation={delta_rotation.as_euler(euler_type, degrees=True)}")

        r_prev = R.from_euler(euler_type, np.array([rotation_prev.roll, rotation_prev.pitch, rotation_prev.yaw]), degrees=True)
        print(f"r_prev2={r_prev.as_euler(euler_type, degrees=True)}")
        print(f"r_prev2={r_prev.as_quat()}")

        r1_r2 = delta_rotation * r_prev
        r_next = R.from_euler(euler_type, np.array([rotation_next.roll, rotation_next.pitch, rotation_next.yaw]), degrees=True)

        print(f"r1_r2={r1_r2.as_euler(euler_type, degrees=True)}")
        print(f"r_next2={r_next.as_euler(euler_type, degrees=True)}")
        print(f"r1_r2={r1_r2.as_quat()}")
        print(f"r_next2={r_next.as_quat()}")

        np.testing.assert_allclose(r1_r2.as_quat(), r_next.as_quat(), atol=1e-3, rtol=5e-2)
        np.testing.assert_allclose(r1_r2.as_euler(euler_type, degrees=True),
                                   r_next.as_euler(euler_type, degrees=True),
                                   atol=5)
        print("\n")
        

    # based on rotation_prev and gyroscope_prev, get estimate of rotation_next
    # Rotation is in degrees
    # Gyroscope measurement is angular velocity in rad/sec
    def testRotationFromImu1(self):
        gyroscope_prev = carla.Vector3D(x=-0.000001, y=-0.003408, z=-0.000001)
        rotation_prev = carla.Rotation(pitch=-0.023161, yaw=0.600249, roll=-0.000061)
        rotation_next = carla.Rotation(pitch=0.000280, yaw=0.600226, roll=-0.000061)
        delta_t = 59.485167 - 59.385167  # 0.1 s

        self._verifyRotationUpdateBasedOnGyroscope(gyroscope_prev, rotation_prev, rotation_next, delta_t)

    def testRotationFromImu2(self):
        gyroscope_prev = carla.Vector3D(x=0.008996, y=0.004928, z=1.065831)
        rotation_prev = carla.Rotation(pitch=0.045120, yaw=114.229301, roll=-0.302734)
        rotation_next = carla.Rotation(pitch=0.049157, yaw=120.168129, roll=-0.299591)
        delta_t = 214.335169 - 214.235169  # 0.1 s

        self._verifyRotationUpdateBasedOnGyroscope(gyroscope_prev, rotation_prev, rotation_next, delta_t)

    def testRotationFromImu3(self):
        gyroscope_prev = carla.Vector3D(x=0.005876, y=0.002788, z=0.265303)
        rotation_prev = carla.Rotation(pitch=-0.038939, yaw=106.974487, roll=-0.073334)
        rotation_next = carla.Rotation(pitch=-0.051179, yaw=108.561646, roll=-0.085419)
        delta_t = 293.835170 - 293.735170  # 0.1 s

        self._verifyRotationUpdateBasedOnGyroscope(gyroscope_prev, rotation_prev, rotation_next, delta_t)


if __name__ == '__main__':
    unittest.main()
