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

# copied from C2M5 assignment files
class Quaternion():
    def __init__(self, w=1., x=0., y=0., z=0., axis_angle=None, euler=None):
        """
        Allow initialization with explicit quaterion wxyz, axis-angle, or Euler XYZ (RPY) angles.

        :param w: w (real) of quaternion.
        :param x: x (i) of quaternion.
        :param y: y (j) of quaternion.
        :param z: z (k) of quaternion.
        :param axis_angle: Set of three values from axis-angle representation, as list or [3,] or [3,1] np.ndarray.
                           See C2M5L2 for details.
        :param euler: Set of three XYZ Euler angles. 
        """
        if axis_angle is None and euler is None:
            self.w = w
            self.x = x
            self.y = y
            self.z = z
        elif euler is not None and axis_angle is not None:
            raise AttributeError("Only one of axis_angle or euler can be specified.")
        elif axis_angle is not None:
            if not (type(axis_angle) == list or type(axis_angle) == np.ndarray) or len(axis_angle) != 3:
                raise ValueError("axis_angle must be list or np.ndarray with length 3.")
            axis_angle = np.array(axis_angle)
            norm = np.linalg.norm(axis_angle)
            self.w = np.cos(norm / 2)
            if norm < 1e-50:  # to avoid instabilities and nans
                self.x = 0
                self.y = 0
                self.z = 0
            else:
                imag = axis_angle / norm * np.sin(norm / 2)
                self.x = imag[0].item()
                self.y = imag[1].item()
                self.z = imag[2].item()
        else:
            roll = euler[0]
            pitch = euler[1]
            yaw = euler[2]

            cy = np.cos(yaw * 0.5)
            sy = np.sin(yaw * 0.5)
            cr = np.cos(roll * 0.5)
            sr = np.sin(roll * 0.5)
            cp = np.cos(pitch * 0.5)
            sp = np.sin(pitch * 0.5)

            # Fixed frame
            self.w = cr * cp * cy + sr * sp * sy
            self.x = sr * cp * cy - cr * sp * sy
            self.y = cr * sp * cy + sr * cp * sy
            self.z = cr * cp * sy - sr * sp * cy

            # Rotating frame
            # self.w = cr * cp * cy - sr * sp * sy
            # self.x = cr * sp * sy + sr * cp * cy
            # self.y = cr * sp * cy - sr * cp * sy
            # self.z = cr * cp * sy + sr * sp * cy

    def __repr__(self):
        # return "Quaternion (wxyz): [%2.5f, %2.5f, %2.5f, %2.5f]" % (self.w, self.x, self.y, self.z)
        return "Quaternion (xyzw): [%2.5f, %2.5f, %2.5f, %2.5f]" % (self.x, self.y, self.z, self.w)

    def to_axis_angle(self):
        t = 2*np.arccos(self.w)
        return np.array(t*np.array([self.x, self.y, self.z])/np.sin(t/2))

    def to_mat(self):
        v = np.array([self.x, self.y, self.z]).reshape(3,1)
        return (self.w ** 2 - np.dot(v.T, v)) * np.eye(3) + \
               2 * np.dot(v, v.T) + 2 * self.w * skew_symmetric(v)

    def to_euler(self):
        """Return as xyz (roll pitch yaw) Euler angles."""
        roll = np.arctan2(2 * (self.w * self.x + self.y * self.z), 1 - 2 * (self.x**2 + self.y**2))
        pitch = np.arcsin(2 * (self.w * self.y - self.z * self.x))
        yaw = np.arctan2(2 * (self.w * self.z + self.x * self.y), 1 - 2 * (self.y**2 + self.z**2))
        return np.array([roll, pitch, yaw])

    def to_numpy(self):
        """Return numpy wxyz representation."""
        return np.array([self.w, self.x, self.y, self.z])

    def normalize(self):
        """Return a (unit) normalized version of this quaternion."""
        norm = np.linalg.norm([self.w, self.x, self.y, self.z])
        return Quaternion(self.w / norm, self.x / norm, self.y / norm, self.z / norm)

    def quat_mult_right(self, q, out='np'):
        """
        Quaternion multiplication operation - in this case, perform multiplication
        on the right, that is, q*self.

        :param q: Either a Quaternion or 4x1 ndarray.
        :param out: Output type, either np or Quaternion.
        :return: Returns quaternion of desired type.
        """
        v = np.array([self.x, self.y, self.z]).reshape(3, 1)
        sum_term = np.zeros([4,4])
        sum_term[0,1:] = -v[:,0]
        sum_term[1:, 0] = v[:,0]
        sum_term[1:, 1:] = -skew_symmetric(v)
        sigma = self.w * np.eye(4) + sum_term

        if type(q).__name__ == "Quaternion":
            quat_np = np.dot(sigma, q.to_numpy())
        else:
            quat_np = np.dot(sigma, q)

        if out == 'np':
            return quat_np
        elif out == 'Quaternion':
            quat_obj = Quaternion(quat_np[0], quat_np[1], quat_np[2], quat_np[3])
            return quat_obj

    def quat_mult_left(self, q, out='np'):
        """
        Quaternion multiplication operation - in this case, perform multiplication
        on the left, that is, self*q.

        :param q: Either a Quaternion or 4x1 ndarray.
        :param out: Output type, either np or Quaternion.
        :return: Returns quaternion of desired type.
        """
        v = np.array([self.x, self.y, self.z]).reshape(3, 1)
        sum_term = np.zeros([4,4])
        sum_term[0,1:] = -v[:,0]
        sum_term[1:, 0] = v[:,0]
        sum_term[1:, 1:] = skew_symmetric(v)
        sigma = self.w * np.eye(4) + sum_term

        if type(q).__name__ == "Quaternion":
            quat_np = np.dot(sigma, q.to_numpy())
        else:
            quat_np = np.dot(sigma, q)

        if out == 'np':
            return quat_np
        elif out == 'Quaternion':
            quat_obj = Quaternion(quat_np[0], quat_np[1], quat_np[2], quat_np[3])
            return quat_obj
