from collections import namedtuple
import logging
import multiprocessing as mp
import threading
import copy
import queue

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np


BufferFields = namedtuple('BufferFields', ['buffer', 'x_idx', 'y_idx', 'label'])

class LivePlotter():
    def __init__(self, figure=None, axes=None, title=None):
        self._buffers = list()
        self._title = title
        self._fig = figure
        self._ax = axes

    def set_title(self, title):
        self._title = title

    def add_buffer(self, buffer, x_idx, y_idx, label):
        self._buffers.append(BufferFields(buffer, x_idx, y_idx, label))

    def draw(self, show=False):
        if not self._buffers:
            logging.error('no buffers to plot')
            return

        if self._fig is None or self._ax is None:
            self._fig, self._ax = plt.subplots(figsize=(26, 12))

        self._lines = list()
        for buffer_fields in self._buffers:
            data = buffer_fields.buffer.get_data()
            line, = self._ax.plot(data[:, buffer_fields.x_idx],
                                  data[:, buffer_fields.y_idx],
                                  label=buffer_fields.label)
            self._lines.append(line)

        self._ax.set(xlabel='Time [s]', ylabel='Position [m]', title=self._title)
        self._ax.legend()
        self._ax.grid(True)

        def update(frame):
            x_min, x_max, y_min, y_max = float('inf'), -float('inf'), float('inf'), -float('inf')

            for buffer_fields, line in zip(self._buffers, self._lines):
                data = buffer_fields.buffer.get_data()
                line.set_data(data[:, buffer_fields.x_idx],
                              data[:, buffer_fields.y_idx])
                x_min = min(x_min, np.min(data[:, buffer_fields.x_idx]))
                x_max = max(x_max, np.max(data[:, buffer_fields.x_idx]))
                y_min = min(y_min, np.min(data[:, buffer_fields.y_idx]))
                y_max = max(y_max, np.max(data[:, buffer_fields.y_idx]))

            self._ax.set_xlim(x_min, x_max)
            self._ax.set_ylim(1.1 * y_min, 1.1 * y_max)  # TODO: use relim and autoscale_view?
            return self._lines
        
        self._ani = FuncAnimation(fig=self._fig, func=update,
                                            frames=50, interval=200)
        # plt.show(block=False)
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


# TODO: make it local
data = ([], [])

class LivePlotterProcess():
    def __init__(self):
        self._queue = mp.Queue()
        # TODO: change start method to 'spawn'? 
        self._plot_process = mp.Process(target=self.update_plot, args=(self._queue,))

    def get_queue(self):
        return self._queue
    
    def update_plot(self, data_queue):
        logging.info('started plotting process')

        global data

        data = ([], [])
        mutex = threading.Lock()

        def retrieve_data_thread(data_queue):
            while True:
                try:
                    global data

                    data_msg = data_queue.get(timeout=1)  # Wait for 1 second to get data from the queue
                    logging.info(f'received new data: {data_msg}')
                    mutex.acquire()
                    data = copy.deepcopy(data_msg)
                    mutex.release()
                    # logging.info(f'data = {data}')
                except queue.Empty:
                    continue

        data_thread = threading.Thread(target=retrieve_data_thread, args=(self._queue,))
        data_thread.start()

        fig, ax = plt.subplots()
        line, = ax.plot([], [], lw=2)

        def init():
            line.set_data([], [])
            return line,

        def animate(frame):
            global data

            mutex.acquire()
            x, y = data
            mutex.release()

            logging.info(f'data = {data}')

            logging.info(f'x={x}, y={y}')

            if len(x) == 0 or len(y) == 0:
                logging.info('skipping')
                return

            line.set_data(x, y)

            ax.set_xlim(np.min(x) - 0.1, np.max(x) + 0.1)
            ax.set_ylim(np.min(y) - 0.1, np.max(y) + 0.1)

            return line,

        ani = FuncAnimation(fig, animate, init_func=init, interval=200)
        plt.show()

        data_thread.join()
    
    def start(self):
        self._plot_process.start()

    def terminate(self):
        self._plot_process.terminate()
    

