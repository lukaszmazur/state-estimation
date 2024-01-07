from collections import namedtuple
import logging
import matplotlib.pyplot as plt
import matplotlib.animation as animation
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
        
        self._ani = animation.FuncAnimation(fig=self._fig, func=update,
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
        for plotter in self._plotters:
            plotter.draw()
        plt.show()
