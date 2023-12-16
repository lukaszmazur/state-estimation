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

    def testRotationFromImu(self):
        # based on rotation_prev and gyroscope_prev, get estimate of rotation_next
        # Rotation is in degrees
        # Gyroscope measurement is angular velocity in rad/sec

        # dataset 1
        gyroscope_prev = carla.Vector3D(x=-0.000001, y=-0.003408, z=-0.000001)
        rotation_prev = carla.Rotation(pitch=-0.023161, yaw=0.600249, roll=-0.000061)

        gyroscope_next = carla.Vector3D(x=-0.000001, y=-0.003796, z=-0.000005)
        rotation_next = carla.Rotation(pitch=0.000280, yaw=0.600226, roll=-0.000061)

        # dataset 2
        gyroscope_prev = carla.Vector3D(x=0.008996, y=0.004928, z=1.065831)
        rotation_prev = carla.Rotation(pitch=0.045120, yaw=114.229301, roll=-0.302734)

        rotation_next = carla.Rotation(pitch=0.049157, yaw=120.168129, roll=-0.299591)


        delta_t = 59.485167 - 59.385167  # 0.1 s
        print(f'delta_t = {delta_t}')

        delta_angles = np.array([gyroscope_prev.x, gyroscope_prev.y, gyroscope_prev.z]) * delta_t
        print(f"delta_angles={delta_angles}")

        euler_type = 'yzx'

        # delta_rotation = R.from_euler(euler_type, delta_angles)
        # print(f"delta_rotation={delta_rotation.as_euler(euler_type, degrees=True)}")

        # delta_rotation2 = R.from_rotvec(delta_angles)
        # print(f"delta_rotation2={delta_rotation2.as_rotvec(degrees=True)}")

        delta_rotation3 = R.from_quat([1, delta_angles[0] * 0.5, delta_angles[1] * 0.5, delta_angles[2] * 0.5])
        print(f"delta_rotation3={delta_rotation3.as_quat()}")

        r_prev = R.from_euler(euler_type, np.array([rotation_prev.pitch, -rotation_prev.yaw, -rotation_prev.roll]), degrees=True)
        print(f"r_prev={r_prev.as_quat()}")

        # r_prev2 = R.from_rotvec(np.array([rotation_prev.roll, -rotation_prev.pitch, -rotation_prev.yaw]), degrees=True)
        # print(f"r_prev2={r_prev2.as_euler(euler_type, degrees=True)}")

        r2_r1 = r_prev * delta_rotation3
        print(f"r2_r1={r2_r1.as_quat()}")
        r1_r2 = delta_rotation3 * r_prev
        print(f"r1_r2={r1_r2.as_quat()}")

        r_next2 = R.from_euler(euler_type, np.array([rotation_next.pitch, -rotation_next.yaw, -rotation_next.roll]), degrees=True)
        print(f"r_next2={r_next2.as_quat()}")

        # r_next2 = R.from_rotvec(np.array([rotation_next.roll, -rotation_next.pitch, -rotation_next.yaw]), degrees=True)
        # print(f"r_next2={r_next2.as_euler(euler_type, degrees=True)}")
        print("\n")

        q1 = Quaternion(euler=np.array([rotation_prev.roll, rotation_prev.pitch, rotation_prev.yaw]))
        # q1 = Quaternion(euler=np.array([rotation_prev.pitch, -rotation_prev.yaw, -rotation_prev.roll]))
        print(f'q1={q1}')

        q_delta = Quaternion(w=1, x=delta_angles[0]*0.5, y=delta_angles[1]*0.5, z=delta_angles[2]*0.5)
        print(f'q_delta={q_delta}')

        q_delta2 = Quaternion(axis_angle=delta_angles)
        print(f'q_delta2={q_delta2}')

        r1_r2 = q_delta2.quat_mult_right(q1, out='Quaternion')
        print(f'r1_r2={r1_r2}')
        r2_r1 = q_delta2.quat_mult_left(q1, out='Quaternion')
        print(f'r2_r1={r2_r1}')

        q2 = Quaternion(euler=np.array([rotation_next.roll, rotation_next.pitch, rotation_next.yaw]))
        # q2 = Quaternion(euler=np.array([rotation_next.pitch, -rotation_next.yaw, -rotation_next.roll]))
        print(f'q2={q2}')

        print("\n")





if __name__ == '__main__':
    unittest.main()
