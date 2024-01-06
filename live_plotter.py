import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation


class LivePlotter():
    def __init__(self, buffer_to_plot, x_idx, y_idx, label):
        self._buffer = buffer_to_plot
        self._x_idx = x_idx
        self._y_idx = y_idx
        self._label = label

    def draw(self):
        fig, ax = plt.subplots(figsize=(26, 12))
        data = self._buffer.get_data()
        line, = ax.plot(data[:, self._x_idx], data[:, self._y_idx],
                             label=self._label)
        ax.set(xlabel='Time [s]', ylabel='Position [m]', title='Title!')
        ax.legend()

        def update(frame):
            data = self._buffer.get_data()
            line.set_data(data[:, self._x_idx], data[:, self._y_idx])
            ax.set_xlim(np.min(data[:, self._x_idx]), np.max(data[:, self._x_idx]))
            ax.set_ylim(1.1 * np.min(data[:, self._y_idx]), 1.1 * np.max(data[:, self._y_idx]))
            return line
        
        ani = animation.FuncAnimation(fig=fig, func=update, frames=50, interval=200)
        plt.show()
        