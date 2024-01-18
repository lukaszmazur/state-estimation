import os
import logging
import matplotlib.pyplot as plt


def plot_ground_truth_and_gnss(gt_data, gnss_data, output_path='.'):
    # logging.info(f'gt: \n{gt_data}')
    # logging.info(f'gnss: \n{gnss_data}')
    est_traj_fig = plt.figure(figsize=(18, 12))
    ax = est_traj_fig.add_subplot(111, projection='3d')
    ax.plot(gnss_data[:, 0], gnss_data[:, 1], gnss_data[:, 2], label='GNSS')
    ax.plot(gt_data[:, 0], gt_data[:, 1], gt_data[:, 2], label='Ground Truth')
    ax.set_xlabel('Easting [m]')
    ax.set_ylabel('Northing [m]')
    ax.set_zlabel('Up [m]')
    ax.set_title('Ground Truth and GNSS')
    ax.legend(loc=(0.62,0.77))
    ax.view_init(elev=45, azim=-50)

    est_traj_fig.savefig(os.path.join(output_path, 'Figure_ground_truth_and_gnss.png'))

    plt.show()

def plot_ground_truth_and_estimated_3d(gt_data, estimated_data, output_path='.'):
    est_traj_fig = plt.figure(figsize=(18, 12))
    ax = est_traj_fig.add_subplot(111, projection='3d')
    ax.plot(estimated_data[:, 0], estimated_data[:, 1], estimated_data[:, 2], label='Estimated')
    ax.plot(gt_data[:, 0], gt_data[:, 1], gt_data[:, 2], label='Ground Truth')
    ax.set_xlabel('Easting [m]')
    ax.set_ylabel('Northing [m]')
    ax.set_zlabel('Up [m]')
    ax.set_title('Ground Truth and GNSS')
    ax.legend(loc=(0.62,0.77))
    ax.view_init(elev=45, azim=-50)

    est_traj_fig.savefig(os.path.join(output_path, 'Figure_ground_truth_and_estimated_3d.png'))

    plt.show()

def plot_ground_truth_and_estimated(gt_data, p_est, v_est, q_est, t_est, output_path='.'):
    fig = plt.figure(figsize=(26, 12))

    ax_p_x = fig.add_subplot(231)
    ax_p_x.plot(gt_data[:, 15], gt_data[:, 0], label='Ground Truth')
    ax_p_x.plot(t_est, p_est[:, 0], label='Estimated')
    ax_p_x.set_xlabel('Time [s]')
    ax_p_x.set_ylabel('Position [m]')
    ax_p_x.set_title('Position x-axis')
    ax_p_x.legend()

    ax_p_y = fig.add_subplot(232)
    ax_p_y.plot(gt_data[:, 15], gt_data[:, 1], label='Ground Truth')
    ax_p_y.plot(t_est, p_est[:, 1], label='Estimated')
    ax_p_y.set_xlabel('Time [s]')
    ax_p_y.set_ylabel('Position [m]')
    ax_p_y.set_title('Position y-axis')
    ax_p_y.legend()

    ax_p_z = fig.add_subplot(233)
    ax_p_z.plot(gt_data[:, 15], gt_data[:, 2], label='Ground Truth')
    ax_p_z.plot(t_est, p_est[:, 2], label='Estimated')
    ax_p_z.set_xlabel('Time [s]')
    ax_p_z.set_ylabel('Position [m]')
    ax_p_z.set_title('Position z-axis')
    ax_p_z.legend()

    # def convert_quaternion_to_euler(quat_array):
    #     return R.from_quat(quat_array).as_euler('xyz', degrees=True)
    # rot_est = np.apply_along_axis(convert_quaternion_to_euler, axis=1, arr=q_est)

    # ax_r_x = fig.add_subplot(234)
    # ax_r_x.plot(gt_data[:, 15], gt_data[:, 3], label='Ground Truth')
    # ax_r_x.plot(t_est, rot_est[:, 0], label='Estimated')
    # ax_r_x.set_xlabel('Time [s]')
    # ax_r_x.set_ylabel('Angle [deg]')
    # ax_r_x.set_title('Roll angle (x-axis)')
    # ax_r_x.legend()

    # ax_r_y = fig.add_subplot(235)
    # ax_r_y.plot(gt_data[:, 15], gt_data[:, 4], label='Ground Truth')
    # ax_r_y.plot(t_est, rot_est[:, 1], label='Estimated')
    # ax_r_y.set_xlabel('Time [s]')
    # ax_r_y.set_ylabel('Angle [deg]')
    # ax_r_y.set_title('Pitch angle (y-axis)')
    # ax_r_y.legend()

    # ax_r_z = fig.add_subplot(236)
    # ax_r_z.plot(gt_data[:, 15], gt_data[:, 5], label='Ground Truth')
    # ax_r_z.plot(t_est, rot_est[:, 2], label='Estimated')
    # ax_r_z.set_xlabel('Time [s]')
    # ax_r_z.set_ylabel('Angle [deg]')
    # ax_r_z.set_title('Yaw angle (z-axis)')
    # ax_r_z.legend()

    # fig2 = plt.figure(figsize=(26, 12))
    # ax_a_x = fig2.add_subplot(231)
    # ax_a_x.plot(gt_data[:, 15], gt_data[:, 12], label='Ground Truth')
    # ax_a_x.plot(t_est, a_est[:, 0], label='Estimated')
    # ax_a_x.set_xlabel('Time [s]')
    # ax_a_x.set_ylabel('Acceleration [m/s^2]')
    # ax_a_x.set_title('Acceleration x-axis')
    # ax_a_x.legend()

    # ax_a_y = fig2.add_subplot(232)
    # ax_a_y.plot(gt_data[:, 15], gt_data[:, 13], label='Ground Truth')
    # ax_a_y.plot(t_est, a_est[:, 1], label='Estimated')
    # ax_a_y.set_xlabel('Time [s]')
    # ax_a_y.set_ylabel('Acceleration [m/s^2]')
    # ax_a_y.set_title('Acceleration y-axis')
    # ax_a_y.legend()

    # ax_a_z = fig2.add_subplot(233)
    # ax_a_z.plot(gt_data[:, 15], gt_data[:, 14], label='Ground Truth')
    # ax_a_z.plot(t_est, a_est[:, 2], label='Estimated')
    # ax_a_z.set_xlabel('Time [s]')
    # ax_a_z.set_ylabel('Acceleration [m/s^2]')
    # ax_a_z.set_title('Acceleration z-axis')
    # ax_a_z.legend()

    fig.savefig(os.path.join(output_path, 'Figure_ground_truth_and_estimated.png'))

    plt.show()

def plot_imu_data(imu_data):
    logging.info(f'imu: \n{imu_data}')
    imu_data_fig = plt.figure(figsize=(36, 12))
    ax_a_x = imu_data_fig.add_subplot(231)
    ax_a_x.plot(imu_data[:, 0])
    ax_a_x.set_title('Acceleration x-axis')
    ax_a_y = imu_data_fig.add_subplot(232)
    ax_a_y.plot(imu_data[:, 1])
    ax_a_y.set_title('Acceleration y-axis')
    ax_a_z = imu_data_fig.add_subplot(233)
    ax_a_z.plot(imu_data[:, 2])
    ax_a_z.set_title('Acceleration z-axis')
    ax_w_x = imu_data_fig.add_subplot(234)
    ax_w_x.plot(imu_data[:, 3])
    ax_w_x.set_title('Angular velocity x-axis')
    ax_w_y = imu_data_fig.add_subplot(235)
    ax_w_y.plot(imu_data[:, 4])
    ax_w_y.set_title('Angular velocity y-axis')
    ax_w_z = imu_data_fig.add_subplot(236)
    ax_w_z.plot(imu_data[:, 5])
    ax_w_z.set_title('Angular velocity z-axis')
    plt.show()
