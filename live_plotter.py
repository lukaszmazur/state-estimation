import logging
import multiprocessing as mp
import threading
import queue

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

from utils import RingBuffer


class LivePlotter():
    def __init__(self, figure, axes, buffer_size=1000):
        self._title = None
        self._fig = figure
        self._ax = axes
        self._lines = list()
        self._lines_data = list()
        self._buffer_size = buffer_size

        self._lock = threading.Lock()

    def set_title(self, title):
        self._title = title

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

    def draw(self, show=False):
        if not self._lines:
            logging.error('no lines to plot')
            return

        # TODO: make labels settable
        self._ax.set(xlabel='Time [s]', ylabel='Position [m]', title=self._title)
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
                min_y_span = 1
                if abs(current_ylim[1] - current_ylim[0]) < min_y_span:
                    self._ax.set_ylim(current_ylim[0] - min_y_span/2,
                                      current_ylim[1] + min_y_span/2)

            return self._lines

        self._ani = FuncAnimation(fig=self._fig, func=update, interval=200)

        if show:
            plt.show()


class LivePlotterComposer():
    def __init__(self):
        self._nrows = None
        self._ncols = None
        self._plotters = list()

    def add_plotters(self, nrows, ncols, figsize=(26, 12)):
        self._nrows = nrows
        self._ncols = ncols
        fig, axes_list = plt.subplots(nrows, ncols, figsize=figsize)
        self._plotters = [LivePlotter(fig, axes) for axes in axes_list]
        return self._plotters

    def draw(self):
        """
        NOTE: Drawing must be blocking when using FuncAnimation
              (see https://github.com/matplotlib/matplotlib/issues/24588).
        """
        for plotter in self._plotters:
            plotter.draw()
        plt.show()


class LivePlotterProcess():
    def __init__(self):
        self._queue = mp.Queue()
        # TODO: change start method to 'spawn'?
        self._plot_process = mp.Process(target=self.update_plot, args=(self._queue,))

    def get_queue(self):
        return self._queue

    def update_plot(self, data_queue):
        logging.info('started plotting process')

        plotter_composer = LivePlotterComposer()
        plotters = plotter_composer.add_plotters(1, 3)
        plotters[0].set_title('X position')
        x_est_id = plotters[0].add_line('Estimated')
        x_gt_id = plotters[0].add_line('Ground Truth')
        plotters[1].set_title('Y position')
        y_est_id = plotters[1].add_line('Estimated')
        y_gt_id = plotters[1].add_line('Ground Truth')
        plotters[2].set_title('Z position')
        z_est_id = plotters[2].add_line('Estimated')
        z_gt_id = plotters[2].add_line('Ground Truth')

        def retrieve_data_thread(data_queue):
            while True:
                try:
                    data = data_queue.get(timeout=1)  # Wait for 1 second to get data from the queue
                    logging.info(f'received new data: {data}')

                    est_t, est_x, est_y, est_z, gt_t, gt_x, gt_y, gt_z = data
                    plotters[0].add_data(est_t, est_x, x_est_id)
                    plotters[0].add_data(gt_t, gt_x, x_gt_id)
                    plotters[1].add_data(est_t, est_y, y_est_id)
                    plotters[1].add_data(gt_t, gt_y, y_gt_id)
                    plotters[2].add_data(est_t, est_z, z_est_id)
                    plotters[2].add_data(gt_t, gt_z, z_gt_id)
                except queue.Empty:
                    continue

        data_thread = threading.Thread(target=retrieve_data_thread, args=(self._queue,))
        data_thread.start()

        plotter_composer.draw()

        data_thread.join()

    def start(self):
        self._plot_process.start()

    def terminate(self):
        self._plot_process.terminate()
