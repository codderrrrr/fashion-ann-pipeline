import numpy as np
import yaml
import os

with open("params.yaml", "r") as f:
    params = yaml.safe_load(f)

preprocess_params = params["preprocess"]

DATA_DIR = "data/raw"
PREPROCESSED_DATA_DIR = "data/processed"

VAL_SIZE = preprocess_params["val_size"]
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(PREPROCESSED_DATA_DIR, exist_ok=True)

data = np.load(f"{DATA_DIR}/Fashion-MNIST.npz")
X_train, X_test, y_train, y_test = data['a'], data['b'], data['c'], data['d']

X_train = X_train / 255
X_test = X_test / 255

VAL_SPLIT = int(VAL_SIZE * len(X_train))
X_train, X_val = X_train[VAL_SPLIT:], X_train[:VAL_SPLIT]
y_train, y_val = y_train[VAL_SPLIT:], y_train[:VAL_SPLIT]


np.savez(f"{PREPROCESSED_DATA_DIR}/Fashion-MNIST.npz", a=X_train, b=X_test, c=y_train, d=y_test, e=X_val, f=y_val)
print(f"Data Saved to {PREPROCESSED_DATA_DIR}/Fashion-MNIST.npz")