# Import libraries
from keras import layers
import keras
import numpy as np
from keras.models import Sequential
from keras.applications import ConvNeXtXLarge
import tensorflow as tf
from keras.optimizers import AdamW
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.utils.class_weight import compute_class_weight
import random
import warnings
from helpers import load_dataset, show_sample_batch, show_batch_shape, create_augmentation_layer, \
    show_augmented_batch, suppress_tf_warnings, plot_model_score


# Supress warnings
warnings.filterwarnings('ignore')
suppress_tf_warnings()

# Get random seed
random_int = random.randint(0, 1000)
print("Using random seed:", random_int)

# Set variables and config
AUTOTUNE = tf.data.AUTOTUNE
img_height = 300
img_width = 300
name = "kit"
# Variables to control training flow
# Set to True to load trained model
load_model = False
load_path = "./kit_best_model.h5"
# Config
path_addon = 'train_val_data'
config = {
    "path": f"/home/luke/train_val_data",
    "batch_size": 32,
    "img_height": img_height,
    "img_width": img_width,
    "seed": random_int,
}

# Load dataset and classes
train_ds, val_ds, class_names = load_dataset(**config)
print(class_names)

class_counts = {class_name: 0 for class_name in class_names}

for images, label in train_ds.unbatch():  # Iterate over each instance
    class_name = class_names[label.numpy()]  # Directly use label to get class name
    class_counts[class_name] += 1

class_count_validation = {class_name: 0 for class_name in class_names}

for images, label in val_ds.unbatch():  # Iterate over each instance
    class_name = class_names[label.numpy()]  # Directly use label to get class name
    class_count_validation[class_name] += 1

print("Validation:", {class_name: count for class_name, count in class_count_validation.items()})
print("Train:", {class_name: count for class_name, count in class_counts.items()})

# Convert class counts to a list in the order of class names
class_samples = np.array([class_counts[class_name] for class_name in class_names])

# Calculate class weights
# This requires the classes to be sequential numbers starting from 0, which they typically are if indexed by class_names
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.arange(len(class_names)),
    y=np.concatenate([np.full(count, i) for i, count in enumerate(class_samples)])
)

# Convert class weights to a dictionary where keys are numerical class indices
class_weight_dict = {i: weight for i, weight in enumerate(class_weights)}

print(class_weight_dict)

# Show sample batch
show_sample_batch(train_ds, class_names)
show_batch_shape(train_ds)

# Shuffle/cache and set prefetch buffer size
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# Create data augmentation layer and show augmented batch
data_augmentation = create_augmentation_layer(img_height, img_width)
show_augmented_batch(train_ds, data_augmentation)

# Create a new head and initialize model
num_classes = len(class_names)

# Load the pre-trained model
ConvNetXt = ConvNeXtXLarge(weights='imagenet', include_top=False, pooling='avg', input_shape=(img_height, img_width, 3))

# Set the trainable flag of the pre-trained model to False
for layer in ConvNetXt.layers:
    layer.trainable = False

# Fine-tuning: unfreeze some layers of pretrained model
for layer in ConvNetXt.layers[-20:]:
    layer.trainable = True

model = Sequential([
    data_augmentation,
    ConvNetXt,
    layers.Dense(num_classes)
]) if not load_model else keras.models.load_model(load_path)

# Define optimizer
optimizer = AdamW(learning_rate=0.001, weight_decay=0.01, use_ema=True)
# Define learning rate scheduler
initial_learning_rate = 1e-3
lr_decay_steps = 3
lr_decay_rate = 0.96
lr_scheduler = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate,
    decay_steps=lr_decay_steps,
    decay_rate=lr_decay_rate,
    staircase=True)

# Compile model
model.compile(optimizer=optimizer,
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
model.summary()

# Define callbacks
lr = tf.keras.callbacks.LearningRateScheduler(lr_scheduler)
early_stopping = EarlyStopping(monitor='loss', patience=7, verbose=1, mode='auto', restore_best_weights=True)
model_checkpoint = ModelCheckpoint(filepath=f"{name}_best_model.h5", monitor='loss', verbose=1, save_best_only=True, mode='auto')

# Train model
epochs = 50

# Use GPU if available
device = tf.test.gpu_device_name() if tf.test.is_gpu_available() else '/CPU:0'
print("Using Device:", device)

with tf.device(device):
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=[lr, early_stopping, model_checkpoint],
        class_weight=class_weight_dict,
    )
# Plot and save model score
plot_model_score(history, name)

# Save model
model.save(f"{name}.h5")
