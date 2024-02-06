import os
import pandas as pd
import tensorflow as tf
from keras.src.applications.convnext import LayerScale, StochasticDepth
from tensorflow import keras


# Define config
img_height = 300
img_width = 300
img_folder = '/home/luke/test_data/test'
model_path = '/tmp/pycharm_project_534/training/kit_best_model.h5'
export_folder = '/home/luke'
classes = ['Bicycle', 'Bridge', 'Bus', 'Car', 'Chimney', 'Crosswalk', 'Hydrant', 'Motorcycle', 'Other', 'Palm', 'Stair', 'Traffic Light']

# Load model
model = keras.models.load_model(model_path, compile=False,custom_objects={ "LayerScale": LayerScale, "StochasticDepth": StochasticDepth})

# Load images
images = []
img_names = []

for image in os.listdir(img_folder):
    img_names.append(image)
    img = tf.keras.utils.load_img(f"{img_folder}/{image}", target_size=(img_height, img_width))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Create a batch
    images.append(img_array)


# Predict
all_predictions = {}
class_names = classes

for img_array, name in zip(images, img_names):
    predictions = model.predict(img_array)

    predictions = predictions.flatten()

    all_predictions[name] = predictions


# Export predictions to CSV or text file
df_results = pd.DataFrame.from_dict(all_predictions, orient='index', columns=class_names)

# Reset the index to turn the image names from the index into a column
df_results.reset_index(inplace=True)

# Rename the 'index' column to 'ImageName'
df_results.rename(columns={'index': 'ImageName'}, inplace=True)

# Export the DataFrame to a CSV file
df_results.to_csv(f"{export_folder}/predictions.csv", index=False)
