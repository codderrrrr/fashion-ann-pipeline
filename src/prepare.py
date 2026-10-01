from tensorflow.keras.datasets import fashion_mnist # type: ignore
import numpy as np
import os

DATA_DIR = "data/raw"
os.makedirs(DATA_DIR, exist_ok=True)

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

np.savez(f"{DATA_DIR}/Fashion-MNIST.npz", a=x_train, b=x_test, c=y_train, d=y_test)
print(f"Data Saved to {DATA_DIR}/Fashion-MNIST.npz")