import tensorflow as tf
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import json

MODEL_DIR = "models/"
PREPROCESSED_DATA_DIR = "data/processed"

model = tf.keras.models.load_model(f"{MODEL_DIR}model.h5")
data = np.load(f"{PREPROCESSED_DATA_DIR}/Fashion-MNIST.npz")
X_train, X_test, y_train, y_test, X_val, y_val = data['a'], data['b'], data['c'], data['d'], data['e'], data['f']

test_loss = model.evaluate(X_test, y_test)
y_pred_probs = model.predict(X_test, verbose=0)
y_pred = np.argmax(y_pred_probs, axis=1)

cm = confusion_matrix(y_test, y_pred)

metrics = {
    "Test_loss": float(test_loss),
    "Accuracy_score": accuracy_score(y_test, y_pred)
}

print(metrics)

disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.savefig("confusion_matrix.png", dpi=150)
plt.close()

with open("metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)