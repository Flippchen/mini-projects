import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tabulate import tabulate
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from shapely.geometry import LineString, Polygon


# Function to add an image to the LiDAR plot
def add_image_to_plot(ax, img, zoom, x_offset, y_offset, envelope_coords):
    # Create an image box
    imagebox = OffsetImage(img, zoom=zoom)
    # Place the image at the given offset
    ab = AnnotationBbox(imagebox, (x_offset, y_offset), frameon=False, pad=0)
    # Add the image to the plot
    ax.add_artist(ab)

    envelope = Polygon(envelope_coords)
    x, y = envelope.exterior.xy
    ax.plot(x, y, color="purple")


def plot_lines_and_shadows(num_lines, field_of_view_degrees, line_color, num_triangles, triangle_height, triangle_width, plot_name, image_envelope_coords, image, angle_offset=0):
    # Set the origin for all lines
    origin = [0, 1.3]

    # Create a polygon for the image envelope
    image_envelope = Polygon(image_envelope_coords)

    # Convert field of view to radians
    field_of_view_radians = np.radians(field_of_view_degrees)

    # Generate angles for the lines
    angles = np.linspace(-field_of_view_radians / 2, field_of_view_radians / 2, num_lines)

    # Set a scaling factor for the line length
    scaling_factor = 40.0

    # Set a large figure size
    fig, ax = plt.subplots(figsize=(50, 10))

    # Initialize a list to store the number of lines hitting each cone
    cone_hits = [0] * num_triangles

    # Plot each line with an angle offset and longer length
    for angle in angles:
        # Extend the lines until the end of the graph with scaling
        angle += angle_offset / 100
        end_point = [scaling_factor * np.cos(angle), scaling_factor * np.sin(angle)]
        line_hit_envelope = False
        endpoint_copy = end_point.copy()

        # Determine if the line intersects with any triangle (cone)
        line = LineString([origin, endpoint_copy])
        if line.intersects(image_envelope):
            line_hit_envelope = True

        for i in range(1, num_triangles + 1):
            base_x = 5 * i
            # Check if the line intersects with the triangle
            if (origin[0] <= base_x <= end_point[0] + origin[0]) or (origin[0] >= base_x >= end_point[0] + origin[0]):
                # Calculate the intersection point
                intersect_y = origin[1] + (base_x - origin[0]) * np.tan(angle)
                if 0 <= intersect_y <= triangle_height:
                    # Adjust end_point to the intersection
                    end_point = [base_x - origin[0], intersect_y - origin[1]]
                    # Increment the hit count for this cone
                    cone_hits[i - 1] += 1 and not line_hit_envelope
                    break



        # Plot the line with the specified color
        plt.plot([origin[0], end_point[0] + origin[0]], [origin[1], end_point[1] + origin[1]], color=line_color if not line_hit_envelope else 'pink')

    # Set plot limits
    plt.xlim(0, 40)
    plt.ylim(0, 5)

    # Set aspect ratio to be equal
    plt.gca().set_aspect('equal', adjustable='box')

    # Plot triangles
    for i in range(1, num_triangles + 1):
        # Define the base of the triangle
        base_x = 5 * i
        base_y = 0

        # Define the vertices of the triangle
        triangle = [[base_x - triangle_width / 2, base_y],
                    [base_x + triangle_width / 2, base_y],
                    [base_x, base_y + triangle_height]]

        # Plot the triangle
        triangle = np.array(triangle)
        plt.fill(triangle[:, 0], triangle[:, 1], 'orange')

    add_image_to_plot(ax, image, 0.5, 0.65, 0.93, image_envelope_coords)

    # Add a title to the plot
    plt.title(plot_name)
    plt.show()

    return cone_hits


different_lidar_configs = [
    {
        "num_lines": 64,
        "field_of_view_degrees": 45,
        "line_color": 'green',
        "num_triangles": 8,
        "triangle_height": 0.4,
        "triangle_width": 0.4,
        "angle_offset": -5,
        "plot_name": "64 Channels, 45° FOV"
    },
    {
        "num_lines": 128,
        "field_of_view_degrees": 45,
        "line_color": 'blue',
        "num_triangles": 8,
        "triangle_height": 0.4,
        "triangle_width": 0.4,
        "plot_name": "128 Channels, 45° FOV"
    },
    {
        "num_lines": 128,
        "field_of_view_degrees": 22.5,
        "line_color": 'red',
        "num_triangles": 8,
        "triangle_height": 0.4,
        "triangle_width": 0.4,
        "plot_name": "128 Channels, 22.5° FOV",
        "angle_offset": -5
    }
]
different_lidar_configs = different_lidar_configs[:1]
# Calculate the distance to each cone
cone_distances = [5 * i for i in range(1, 9)]

# Define the coordinates for the car envelope
image_envelope_coords = [(0.2,0.6), (1.2,0.95), (2.3, 0.85), (2.3, 0.35), (1.2, 0.2), (1, 0.2), (1, 0.2), (0.2, 0.6)]

# Initialize DataFrame
df = pd.DataFrame({"Cone Distance (m)": cone_distances})

# Load car image
image = plt.imread("cm24.png")

# Plotting each configuration and updating DataFrame
for config in different_lidar_configs:
    hits = plot_lines_and_shadows(**config, image_envelope_coords=image_envelope_coords, image=image)
    df[config["plot_name"]] = hits

for config in different_lidar_configs:
    hits = df[config["plot_name"]]

    # Plotting
    plt.plot(cone_distances, hits, label=config["plot_name"], marker='o', color=config["line_color"])

# Finalizing the plot
plt.title('LiDAR Performance Comparison')
plt.xlabel('Cone Distance (m)')
plt.ylabel('Number of Hits')
plt.legend()
plt.grid(True)
plt.show()

# Set index for DataFrame
df.set_index("Cone Distance (m)", inplace=True)

# Save the dataframe to an Excel file
df.to_excel("cone_hits.xlsx")

# Print the DataFrame
print(tabulate(df, headers='keys', tablefmt='psql'))
