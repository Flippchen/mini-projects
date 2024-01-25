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

# Intrinsic matrix and its inverse
intrinsic_matrix = np.array([
    [fx, 0, cx],
    [0, fy, cy],
    [0, 0, 1]
])
intrinsic_matrix_inv = np.linalg.inv(intrinsic_matrix)

# Calculate the rotation matrix from Euler angles
r = R.from_euler('xyz', [roll, pitch, yaw], degrees=True)
rotation_matrix = r.as_matrix()

# Translation vector
translation_vector = np.array([x, y, z])

# Extrinsic matrix (Rt matrix)
rt_matrix = np.hstack((rotation_matrix, translation_vector.reshape(-1, 1)))
rt_matrix = np.vstack((rt_matrix, np.array([0, 0, 0, 1])))

# Position of the object in the image frame (homogeneous coordinates)
P_Image = np.array([u, v, 1])

# Convert image coordinates to camera coordinates
P_camera = intrinsic_matrix_inv @ P_Image

