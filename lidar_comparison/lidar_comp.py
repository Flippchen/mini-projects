import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tabulate import tabulate


def plot_lines_triangles_shadows(num_lines, field_of_view_degrees, line_color, num_triangles, triangle_height, triangle_width, plot_name, angle_offset=0):
    # Set the origin for all lines
    origin = [0, 1.3]  # Move the origin to the left

    # Convert field of view to radians
    field_of_view_radians = np.radians(field_of_view_degrees)

    # Generate angles for the lines
    angles = np.linspace(-field_of_view_radians / 2, field_of_view_radians / 2, num_lines)

    # Set a scaling factor for the line length
    scaling_factor = 40.0

    # Set a larger figure size
    plt.figure(figsize=(50, 10))

    # Initialize a list to store the number of lines hitting each cone
    cone_hits = [0] * num_triangles