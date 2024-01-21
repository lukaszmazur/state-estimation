import numpy as np


class RingBuffer():
    """
    Base class for ring data buffers.
    """
    def __init__(self, element_size, buffer_size):
        self._buffer_size = buffer_size
        self._data = np.zeros((self._buffer_size, element_size))
        self._number_of_elements_in_buffer = 0

    def insert_element(self, element):
        if (self._number_of_elements_in_buffer < self._buffer_size):
            self._data[self._number_of_elements_in_buffer, :] = element
            self._number_of_elements_in_buffer += 1
        else:
            # TODO: check for more efficient options
            self._data = np.roll(self._data, -1, axis=0)
            self._data[-1, :] = element

    def get_data(self):
        return self._data[:self._number_of_elements_in_buffer]


# copied from C2M5 assignment files
def angle_normalize(a):
    """Normalize angles to lie in range -pi < a[i] <= pi."""
    a = np.remainder(a, 2*np.pi)
    a[a <= -np.pi] += 2*np.pi
    a[a  >  np.pi] -= 2*np.pi
    return a

# copied from C2M5 assignment files
def skew_symmetric(v):
    """Skew symmetric form of a 3x1 vector."""
    return np.array(
        [[0, -v[2], v[1]],
         [v[2], 0, -v[0]],
         [-v[1], v[0], 0]], dtype=np.float64)
