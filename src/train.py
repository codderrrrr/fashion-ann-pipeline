import tensorflow as tf
import keras
from keras import layers
import numpy as np
import pandas as pd
import os

PREPROCESSED_DATA_DIR = "data/processed"
MODEL_DIR = "models/"

os.makedirs(MODEL_DIR, exist_ok=True)

model = keras.Sequential(
    [
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(10, activation='softmax')
    ]
)

model.compile(optimizer='adam', loss="sparse_categorical_crossentropy")

data = np.load(f"{PREPROCESSED_DATA_DIR}/Fashion-MNIST.npz")
X_train, X_test, y_train, y_test, X_val, y_val = data['a'], data['b'], data['c'], data['d'], data['e'], data['f']

history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=32)

model.save(f"{MODEL_DIR}model.h5")
history_df = pd.DataFrame(history.history)

history_df.to_csv(f"{MODEL_DIR}/history.csv", index_label='epoch')