from collections import namedtuple
import logging
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np


BufferFields = namedtuple('BufferFields', ['buffer', 'x_idx', 'y_idx', 'label'])

class LivePlotter():

    def __init__(self, title):
        self._buffers = list()
        self._title = title

    def add_buffer(self, buffer, x_idx, y_idx, label):
        self._buffers.append(BufferFields(buffer, x_idx, y_idx, label))

    def draw(self):
        if not self._buffers:
            logging.error('no buffers to plot')
            return

        fig, ax = plt.subplots(figsize=(26, 12))

        lines = list()
        for buffer_fields in self._buffers:
            data = buffer_fields.buffer.get_data()
            line, = ax.plot(data[:, buffer_fields.x_idx],
                            data[:, buffer_fields.y_idx],
                            label=buffer_fields.label)
            lines.append(line)

        ax.set(xlabel='Time [s]', ylabel='Position [m]', title=self._title)
        ax.legend()
        ax.grid(True)

        def update(frame):
            x_min, x_max, y_min, y_max = float('inf'), -float('inf'), float('inf'), -float('inf')

            for buffer_fields, line in zip(self._buffers, lines):
                data = buffer_fields.buffer.get_data()
                line.set_data(data[:, buffer_fields.x_idx],
                              data[:, buffer_fields.y_idx])
                x_min = min(x_min, np.min(data[:, buffer_fields.x_idx]))
                x_max = max(x_max, np.max(data[:, buffer_fields.x_idx]))
                y_min = min(y_min, np.min(data[:, buffer_fields.y_idx]))
                y_max = max(y_max, np.max(data[:, buffer_fields.y_idx]))

            ax.set_xlim(x_min, x_max)
            ax.set_ylim(1.1 * y_min, 1.1 * y_max)
            return lines
        
        ani = animation.FuncAnimation(fig=fig, func=update, frames=50, interval=200)
        # plt.show(block=False)
        plt.show()
        