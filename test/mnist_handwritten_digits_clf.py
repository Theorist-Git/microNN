import sys, os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from sklearn.metrics import classification_report

# make sure your MLP package is importable
sys.path.append(os.path.abspath('..'))
from MLP.nn import MLP

# 1. Load & preprocess MNIST
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# Flatten to (N, 784) and normalize to [0,1]
x_train = x_train.reshape(-1, 784).astype(np.float32) / 255.0
x_test  = x_test.reshape(-1, 784).astype(np.float32) / 255.0

# 2. Instantiate a 10‑way softmax MLP
model = MLP(
    n_inputs=784,
    layers=[
        (128, "relu"),
        (64,  "relu"),
        (10,  "softmax"),
    ],
    epochs=20,
    learning_rate=0.1,
)

# 3. Train with categorical cross‑entropy, batch size 128
model.fit(
    x_train,
    y_train,
    loss_fn="categorical_cross_entropy",
    batch_size=128,
    verbose=True
)

# 4. Evaluate on test set
#    Forward the entire test set (Value of shape (N,10))
y_pred_vals = model(x_test)
#    Grab the NumPy probabilities and pick argmax per sample
y_probs = y_pred_vals.data                # shape (10000, 10)
y_preds = np.argmax(y_probs, axis=1)      # shape (10000,)

print(classification_report(y_test, y_preds, digits=4))
accuracy = np.mean(y_preds == y_test)
print(f"Test Accuracy: {accuracy*100:.2f}%")

# 5. Inspect gradient statistics once after training (optional)
print("\nGradient stats on first 4 parameters:")
for i, p in enumerate(model.parameters()[:4]):
    print(f" param {i}: mean={p.grad.mean():.6e}, std={p.grad.std():.6e}")
    if i >= 3: break

# 6. Show up to 5 misclassified images
wrong = 0
for xi, yi, pred in zip(x_test, y_test, y_preds):
    if pred != yi:
        plt.imshow(xi.reshape(28,28), cmap='gray')
        plt.title(f"True: {yi}, Pred: {pred}")
        plt.axis('off')
        plt.show()
        wrong += 1
        if wrong == 5:
            break