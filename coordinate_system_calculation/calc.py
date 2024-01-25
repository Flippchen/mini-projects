import numpy as np
from scipy.spatial.transform import Rotation as R

# Deactivate scientific notation for numpy
np.set_printoptions(suppress=True)

# Camera intrinsics and image coordinates
cx, cy = 636, 548
fx, fy = 241, 238

# Image coordinates
u, v = 872, 423

# Camera depth of object
d = 1.9 * 1000

# Translation values in millimeters
x, y, z = 500, 160, 1140

# Euler angles for rotation (in degrees)
roll, pitch, yaw = 100, 0, 90


