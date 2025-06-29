import pickle
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import seaborn as sns

# Change this to 'train.p', 'test.p', etc.
file_path = "/home/intern/GTSD/test.p"

# Load the pickle file
with open(file_path, 'rb') as f:
    data = pickle.load(f)

# Inspect structure
print("Keys:", data.keys())
print("Features shape:", data['features'].shape)
print("Labels shape:", len(data['labels']))
print("Data type:", data['features'].dtype)

# Class distribution
label_counts = Counter(data['labels'])
print("Number of classes:", len(label_counts))
print("Sample label counts:", label_counts.most_common(10))

# Plot class distribution
plt.figure(figsize=(12, 4))
sns.barplot(x=list(label_counts.keys()), y=list(label_counts.values()))
plt.title("Class Distribution")
plt.xlabel("Class Label")
plt.ylabel("Count")
plt.show()

# Show 10 random images with their labels
fig, axes = plt.subplots(2, 5, figsize=(12, 5))
for i, ax in enumerate(axes.flat):
    idx = np.random.randint(0, len(data['features']))
    img = data['features'][idx]
    label = data['labels'][idx]
    ax.imshow(img)
    ax.set_title(f"Label: {label}")
    ax.axis('off')
plt.tight_layout()
plt.show()
