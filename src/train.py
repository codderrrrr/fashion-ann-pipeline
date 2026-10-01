import tensorflow as tf
import keras
from keras import layers
import numpy as np
import pandas as pd
import os
import yaml

with open("params.yaml", "r") as f:
    params = yaml.safe_load(f)

train_params = params["train"]

PREPROCESSED_DATA_DIR = "data/processed"
MODEL_DIR = "models/"

os.makedirs(MODEL_DIR, exist_ok=True)

model = keras.Sequential(
    [
        layers.Flatten(),
        layers.Dense(train_params["dense_units"], activation=train_params["dense1_activation"]),
        layers.Dropout(train_params["dropout_rate"]),
        layers.Dense(train_params["dense2_units"], activation=train_params["dense2_activation"])
    ]
)

model.compile(optimizer=train_params["optimizer"], loss=train_params["loss"])

data = np.load(f"{PREPROCESSED_DATA_DIR}/Fashion-MNIST.npz")
X_train, X_test, y_train, y_test, X_val, y_val = data['a'], data['b'], data['c'], data['d'], data['e'], data['f']

history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=train_params["epochs"], batch_size=train_params["batch_size"])

model.save(f"{MODEL_DIR}model.h5")
history_df = pd.DataFrame(history.history)

history_df.to_csv(f"{MODEL_DIR}/history.csv", index_label='epoch')