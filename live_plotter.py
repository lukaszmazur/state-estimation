import logging
import multiprocessing as mp
import threading
import queue

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

from utils import RingBuffer


class LivePlotter():
    """
    Class for handling plots (lines) on a single axes object that are updated
    (drawn) on a regular interval.

    Data for each line is stored in a separate ring buffer.
    """

    def __init__(self, figure, axes, buffer_size=4000):
        self._title = None
        self._xlabel = None
        self._ylabel = None
        self._fig = figure
        self._ax = axes
        self._lines = list()
        self._lines_data = list()
        self._buffer_size = buffer_size
        self._min_y_span = 1

        self._lock = threading.Lock()

    def set_title(self, title):
        self._title = title

    def set_labels(self, xlabel, ylabel):
        self._xlabel = xlabel
        self._ylabel = ylabel

    def set_min_y_span(self, min_y_span):
        """
        Sets minimal span of y axis limits.
        It is meant for cases where plot values oscillate around some value
        with small amplitude (e.g. 0.5 degrees around 0).
        """
        self._min_y_span = min_y_span

    def add_line(self, label):
        """
        Returns line index.
        """
        line, = self._ax.plot([], [], label=label)
        self._lines.append(line)
        self._lines_data.append(RingBuffer(2, self._buffer_size))
        return len(self._lines) - 1

    def add_data(self, x, y, line_idx):
        with self._lock:
            self._lines_data[line_idx].insert_element(np.array([x, y]))

    def draw(self):
        if not self._lines:
            logging.error('no lines to plot')
            return

        self._ax.set(xlabel=self._xlabel,
                     ylabel=self._ylabel, title=self._title)
        self._ax.legend()
        self._ax.grid(True)

        def update(frame):
            logging.info('updating plots')

            any_data_set = False
            with self._lock:
                for line_data, line in zip(self._lines_data, self._lines):
                    data = line_data.get_data()
                    x = data[:, 0]
                    y = data[:, 1]
                    if len(x) == 0 or len(y) == 0:
                        continue
                    line.set_data(x, y)
                    any_data_set = True

            if any_data_set:
                self._ax.autoscale()
                self._ax.relim()
                self._ax.autoscale_view()

                # if span of y axis is very small, set limits manually
                # not to show very small noise spanning entire plot
                current_ylim = self._ax.get_ylim()
                if abs(current_ylim[1] - current_ylim[0]) < self._min_y_span:
                    self._ax.set_ylim(current_ylim[0] - self._min_y_span/2,
                                      current_ylim[1] + self._min_y_span/2)

            return self._lines

        self._ani = FuncAnimation(fig=self._fig, func=update, interval=200)


class LivePlotterComposer():
    """
    Class for managing plotters.
    """

    def __init__(self):
        self._plotters = list()

    def add_plotters(self, nrows, ncols, figsize=(24, 12)):
        fig, axes_list = plt.subplots(nrows, ncols, figsize=figsize)
        # axes_list can be either a scalar, 1D or 2D array
        if not isinstance(axes_list, np.ndarray):
            added_plotters = [LivePlotter(fig, axes_list)]
        elif len(axes_list.shape) == 1:
            added_plotters = [LivePlotter(fig, axes) for axes in axes_list]
        else:
            added_plotters = [[LivePlotter(fig, axes)
                               for axes in row] for row in axes_list]
        self._plotters.extend(added_plotters)
        return added_plotters

    def draw(self):
        """
        NOTE: Drawing must be blocking when using FuncAnimation
              (see https://github.com/matplotlib/matplotlib/issues/24588).
        """
        for plotter in self._plotters:
            if isinstance(plotter, list):
                for p in plotter:
                    p.draw()
            else:
                plotter.draw()
        plt.show()


class LivePlotterProcess():
    """
    Class for starting a separate process for plotting state estimations.
    """

    def __init__(self):
        self._queue = mp.Queue()
        # TODO: change start method to 'spawn'?
        self._plot_process = mp.Process(
            target=self.update_plot, args=(self._queue,))

    def get_queue(self):
        return self._queue

    def update_plot(self, data_queue):
        logging.info('started plotting process')

        plotter_composer = LivePlotterComposer()
        plotters = plotter_composer.add_plotters(3, 3, figsize=(20, 20))

        def add_lines(plotter, title, labels, ylabel, xlabel='Time [s]', min_y_span=1):
            plotter.set_title(title)
            plotter.set_labels(xlabel=xlabel, ylabel=ylabel)
            plotter.set_min_y_span(min_y_span)
            return [plotter.add_line(label) for label in labels]

        labels = ('Estimated', 'Ground Truth')
        x_est_id, x_gt_id = add_lines(
            plotters[0][0], 'X position', labels, ylabel='Position [m]')
        y_est_id, y_gt_id = add_lines(
            plotters[0][1], 'Y position', labels, ylabel='Position [m]')
        z_est_id, z_gt_id = add_lines(
            plotters[0][2], 'Z position', labels, ylabel='Position [m]')
        vx_est_id, vx_gt_id = add_lines(
            plotters[1][0], 'X velocity', labels, ylabel='Velocity [m/s]')
        vy_est_id, vy_gt_id = add_lines(
            plotters[1][1], 'Y velocity', labels, ylabel='Velocity [m/s]')
        vz_est_id, vz_gt_id = add_lines(
            plotters[1][2], 'Z velocity', labels, ylabel='Velocity [m/s]')
        roll_est_id, roll_gt_id = add_lines(
            plotters[2][0], 'Roll', labels, ylabel='Rotation [deg]', min_y_span=5)
        pitch_est_id, pitch_gt_id = add_lines(
            plotters[2][1], 'Pitch', labels, ylabel='Rotation [deg]', min_y_span=5)
        yaw_est_id, yaw_gt_id = add_lines(
            plotters[2][2], 'Yaw', labels, ylabel='Rotation [deg]', min_y_span=5)

        plotter = plotter_composer.add_plotters(1, 1, figsize=(10, 10))[0]
        xy_est_id, xy_gt_id = add_lines(
            plotter, 'XY position', labels, xlabel='X position [m]', ylabel='Y position[m]')

        def retrieve_data_thread(data_queue):
            while True:
                try:
                    # Wait for 1 second to get data from the queue
                    data = data_queue.get(timeout=1)
                    logging.info(f'received new data: {data}')

                    est_t, est_x, est_y, est_z, est_vx, est_vy, est_vz, est_roll, est_pitch, est_yaw, \
                        gt_t, gt_x, gt_y, gt_z, gt_vx, gt_vy, gt_vz, gt_roll, gt_pitch, gt_yaw = data

                    plotters[0][0].add_data(est_t, est_x, x_est_id)
                    plotters[0][0].add_data(gt_t, gt_x, x_gt_id)
                    plotters[0][1].add_data(est_t, est_y, y_est_id)
                    plotters[0][1].add_data(gt_t, gt_y, y_gt_id)
                    plotters[0][2].add_data(est_t, est_z, z_est_id)
                    plotters[0][2].add_data(gt_t, gt_z, z_gt_id)

                    plotters[1][0].add_data(est_t, est_vx, vx_est_id)
                    plotters[1][0].add_data(gt_t, gt_vx, vx_gt_id)
                    plotters[1][1].add_data(est_t, est_vy, vy_est_id)
                    plotters[1][1].add_data(gt_t, gt_vy, vy_gt_id)
                    plotters[1][2].add_data(est_t, est_vz, vz_est_id)
                    plotters[1][2].add_data(gt_t, gt_vz, vz_gt_id)

                    plotters[2][0].add_data(est_t, est_roll, roll_est_id)
                    plotters[2][0].add_data(gt_t, gt_roll, roll_gt_id)
                    plotters[2][1].add_data(est_t, est_pitch, pitch_est_id)
                    plotters[2][1].add_data(gt_t, gt_pitch, pitch_gt_id)
                    plotters[2][2].add_data(est_t, est_yaw, yaw_est_id)
                    plotters[2][2].add_data(gt_t, gt_yaw, yaw_gt_id)

                    plotter.add_data(est_x, est_y, xy_est_id)
                    plotter.add_data(gt_x, gt_y, xy_gt_id)

                except queue.Empty:
                    continue

        data_thread = threading.Thread(
            target=retrieve_data_thread, args=(self._queue,))
        data_thread.start()

        plotter_composer.draw()

        data_thread.join()

    def start(self):
        self._plot_process.start()

    def terminate(self):
        self._plot_process.terminate()
