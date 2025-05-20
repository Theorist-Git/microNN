import sys, os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# adjust path so that `from MLP.nn import MLP` works
sys.path.append(os.path.abspath('..'))
from MLP.nn import MLP

# Load MNIST
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x = np.concatenate([x_train, x_test])
y = np.concatenate([y_train, y_test])

# Normalize and flatten
x = x.reshape(-1, 28 * 28).astype(np.float32) / 255.0

# Binary labels: 1 if digit == 3 else 0
is_three = y == 3
not_three = y != 3

# Get all '3's
x_threes = x[is_three]
y_threes = np.ones(len(x_threes), dtype=np.float32)

# Get equal number of non-'3's (randomly sampled)
np.random.seed(42)
idx_not_threes = np.where(not_three)[0]
selected_not_three_idx = np.random.choice(idx_not_threes, size=len(x_threes), replace=False)
x_not_threes = x[selected_not_three_idx]
y_not_threes = np.zeros(len(x_not_threes), dtype=np.float32)

# Combine to form a balanced dataset
x_balanced = np.concatenate([x_threes, x_not_threes])
y_balanced = np.concatenate([y_threes, y_not_threes])

# Shuffle
indices = np.arange(len(x_balanced))
np.random.shuffle(indices)
x_balanced = x_balanced[indices]
y_balanced = y_balanced[indices]

print("Num 3’s:", len(x_threes))
print("Num not-3’s:", len(x_not_threes))
assert len(x_threes) == len(x_not_threes), "Counts are unequal!"

# Train-test split (80/20)
x_train, x_test, y_train, y_test = train_test_split(
    x_balanced, y_balanced, test_size=0.2, random_state=42
)

# Initialize your model
model = MLP(
    n_inputs=784,
    layers=[(16, "relu"), (16, "relu"), (1, "sigmoid")],
    epochs=500,
    learning_rate=0.1,
)

# Train
model.fit(x_train, y_train, loss_fn="binary_cross_entropy", batch_size=len(x_train.ravel()))

# Evaluate
y_preds = [int(model(xi).data.item() > 0.5) for xi in x_test]
print(classification_report(y_test, y_preds, digits=4))
accuracy = np.mean(np.array(y_preds) == y_test)
print(f"Test Accuracy: {accuracy * 100:.2f}%")

# inside your epoch loop, after cost.backward() but before the update:
for i, p in enumerate(model.parameters()[:4]):
    print(f" param {i} grad mean {p.grad.mean(): .6f}, std {p.grad.std(): .6f}")
    # only inspect the first few; break if you like
    if i >= 3: break


# Show some misclassified samples
wrong_count = 0
for i, (xi, yi) in enumerate(zip(x_test, y_test)):
    pred_prob = model(xi.reshape(1, -1)).data.item()
    pred_label = int(pred_prob > 0.5)
    if pred_label != int(yi):
        plt.imshow(xi.reshape(28, 28), cmap='gray')
        plt.title(f"Pred: {pred_prob:.2f}, Label: {int(yi)}")
        plt.show()
        wrong_count += 1
        if wrong_count == 5:
            break