# -*- coding: utf-8 -*-
"""
# **Importing libraries and loading data**
"""

# Importing basic libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA, FactorAnalysis
from keras.models import Sequential, Model
from keras.layers import Input, Dense
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.cluster import MiniBatchKMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score, adjusted_rand_score
import time, random
from google.colab import drive

# Loading data
(main_img, main_lbl), (test_img, test_lbl) = tf.keras.datasets.fashion_mnist.load_data()
# Loading class names
class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
# Image flattern
main_img = main_img.reshape(main_img.shape[0], -1)
test_img = test_img.reshape(test_img.shape[0], -1)
# Normalizaton
main_img = main_img / 255.0
test_img = test_img / 255.0
# Splitting main to train and validation set
train_img, val_img, train_lbl, val_lbl = train_test_split(main_img, main_lbl, test_size=0.2, random_state=42, stratify=main_lbl)

"""# **Custom functions**"""

def add_row(dim_red_technique, clustering_algorithm,
            tt_dim_red_technique, et_clustering_algorithm,
            sugg_classes,
            calinski_harabasz_idx, davies_bouldin_idx, silhouette_score, ari_idx):
  global df_outcomes
  new_row = {"Dimensionality reduction technique name": dim_red_technique,
             "Clustering algorithm": clustering_algorithm, "Training time for the dim. red. tech.": tt_dim_red_technique,
             "Execution time for the clustering tech.": et_clustering_algorithm, "Number of suggested clusters": sugg_classes,
             "Calinski–Harabasz index": calinski_harabasz_idx, "Davies–Bouldin index": davies_bouldin_idx, "Silhouette score": silhouette_score, "Adjusted Rand Score": ari_idx}
  df_outcomes = pd.concat([df_outcomes, pd.DataFrame([new_row])], ignore_index=True)

def doClustering(model, img):
  model_name = model.__class__.__name__
  start_time = time.time()
  predicted_labels = model.fit_predict(img)
  end_time = time.time()
  execution_time = round((end_time-start_time)*1000)
  n_clusters = len(set(predicted_labels)) - (1 if -1 in predicted_labels else 0)
  calinski_harabasz_idx = round(calinski_harabasz_score(img, predicted_labels), 2)
  davies_bouldin_idx = round(davies_bouldin_score(img, predicted_labels), 2)
  silhouette_idx = round(silhouette_score(img, predicted_labels), 2)
  ari_idx = round(adjusted_rand_score(test_lbl, predicted_labels), 2)
  print(model_name)
  print(f"- Calinski–Harabasz Index: {calinski_harabasz_idx}")
  print(f"- Davies–Bouldin Index: {davies_bouldin_idx}")
  print(f"- Silhouette Score: {silhouette_idx}")
  print(f"- Adjusted Rand Score: {ari_idx}")
  print()
  return (model_name, execution_time, n_clusters, predicted_labels, calinski_harabasz_idx, davies_bouldin_idx, silhouette_idx, ari_idx)

"""# **Creating DataFrame for storage**"""

outcomes = {
  "Dimensionality reduction technique name": [],
  "Clustering algorithm": [],
  "Training time for the dim. red. tech.": [],
  "Execution time for the clustering tech.": [],
  "Number of suggested clusters": [],
  "Calinski–Harabasz index": [],
  "Davies–Bouldin index": [],
  "Silhouette score": [],
  "Adjusted Rand Score": []
}
df_outcomes = pd.DataFrame(outcomes)

"""# **Clustering in normalized images**"""

# DBSCAN
model = DBSCAN(eps=5, min_samples=5)
(model_name, execution_time, n_clusters, predicted_labels_dbscan, calinski_harabasz_idx, davies_bouldin_idx, silhouette_idx, ari_idx) = doClustering(model, test_img)
add_row("None", model_name, "None", execution_time, n_clusters, calinski_harabasz_idx, davies_bouldin_idx, silhouette_idx, ari_idx)
# MiniBatchKMeans
model = MiniBatchKMeans(n_clusters=10, random_state=42, batch_size=200, n_init=5)
(model_name, execution_time, n_clusters, predicted_labels_kmeans, calinski_harabasz_idx, davies_bouldin_idx, silhouette_idx, ari_idx) = doClustering(model, test_img)
add_row("None", model_name, "None", execution_time, n_clusters, calinski_harabasz_idx, davies_bouldin_idx, silhouette_idx, ari_idx)
# AgglomerativeClustering
model = AgglomerativeClustering(n_clusters=10)
(model_name, execution_time, n_clusters, predicted_labels_hier, calinski_harabasz_idx, davies_bouldin_idx, silhouette_idx, ari_idx) = doClustering(model, test_img)
add_row("None", model_name, "None", execution_time, n_clusters, calinski_harabasz_idx, davies_bouldin_idx, silhouette_idx, ari_idx)

"""# **Clustering in encoded images (by PCA)**"""

# Applying Principal Component Analysis
pca = PCA(n_components=0.95)
start_time = time.time()
pca.fit(train_img)
end_time = time.time()
training_time = round((end_time-start_time)*1000)
# print("Number of components retained:", pca.n_components_)
# Encode train and test images
train_encoding_img_pca = pca.transform(train_img)
test_encoding_img_pca = pca.transform(test_img)

# Do some plotting
explained_variance_ratio = pca.explained_variance_ratio_
cumulative_variance_ratio = explained_variance_ratio.cumsum()
plt.figure(figsize=(10, 6))
plt.plot(cumulative_variance_ratio, marker="o", linestyle="-", color="b")
plt.title("Explained Variance (PCA)")
plt.xlabel("Component ID")
plt.ylabel("Explained Variance")
plt.grid(True)
plt.show()

# Display for each class, one random pair (real-reconstructed) image
plt.figure(figsize=(40, 12))
for class_label in range(10):
  indices = np.where(train_lbl == class_label)[0]
  selected_indices = np.random.choice(indices, 1, replace=False)
  for i, idx in enumerate(selected_indices):
    plt.subplot(3, 10, 10 + class_label + 1)
    plt.imshow(train_img[idx].reshape(28, 28), cmap="gray")
    plt.title(f"{class_names[class_label]} \n", fontweight='bold')
    plt.text(0.5, -0.1, f"Real Image", size=10, ha="center", transform=plt.gca().transAxes)
    plt.axis("off")
    plt.subplot(3, 10, (10 + class_label + 1) + 10)
    reconstructed_image = pca.inverse_transform(train_encoding_img_pca[idx]).reshape(28, 28)
    plt.imshow(reconstructed_image, cmap="gray")
    plt.text(0.5, -0.1, f"Reconstructed Image, by PCA", size=10, ha="center", transform=plt.gca().transAxes)
    plt.axis("off")
plt.subplots_adjust(wspace=0.5, hspace=0)
plt.show()

# DBSCAN - Encoded Images
model = DBSCAN(eps=5, min_samples=3)
(model_name, execution_time, n_clusters, predicted_labels_pca_dbscan, calinski_harabasz_idx, davies_bouldin_idx, silhouette_idx, ari_idx) = doClustering(model, test_encoding_img_pca)
add_row("PCA", model_name, training_time, execution_time, n_clusters, calinski_harabasz_idx, davies_bouldin_idx, silhouette_idx, ari_idx)
# MiniBatcgKmeans - Encoded Images
model = MiniBatchKMeans(n_clusters=10, random_state=42, batch_size=200, n_init=5)
(model_name, execution_time, n_clusters, predicted_labels_pca_kmeans, calinski_harabasz_idx, davies_bouldin_idx, silhouette_idx, ari_idx) = doClustering(model, test_encoding_img_pca)
add_row("PCA", model_name, training_time, execution_time, n_clusters, calinski_harabasz_idx, davies_bouldin_idx, silhouette_idx, ari_idx)
# AgglomerativeClustering - Encoded Images
model = AgglomerativeClustering(n_clusters=10)
(model_name, execution_time, n_clusters, predicted_labels_pca_hier, calinski_harabasz_idx, davies_bouldin_idx, silhouette_idx, ari_idx) = doClustering(model, test_encoding_img_pca)
add_row("PCA", model_name, training_time, execution_time, n_clusters, calinski_harabasz_idx, davies_bouldin_idx, silhouette_idx, ari_idx)

"""# **Clustering in encoded images (by SAE)**"""

# Applying Stacked Autoencoder
stacked_encoder = Sequential([
  Dense(392, activation="relu", input_shape=(784,)),
  Dense(196, activation="relu"),
  Dense(98, activation="relu"),
  Dense(12, activation="relu")
])
stacked_decoder = Sequential([
  Dense(98, activation="relu", input_shape=(12,)),
  Dense(196, activation="relu"),
  Dense(392, activation="relu"),
  Dense(784, activation="sigmoid")
])
stacked_autoencoder = Sequential([stacked_encoder, stacked_decoder])
stacked_autoencoder.compile(optimizer="adam", loss="binary_crossentropy")
start_time = time.time()
history = stacked_autoencoder.fit(train_img, train_img, epochs=10, batch_size=256, validation_data=(val_img, val_img), verbose=0)
end_time = time.time()
training_time = round((end_time-start_time)*1000)
# print("Total loss:",  history.history["loss"][-1])
# print("Total loss:",  history.history["val_loss"][-1])
# Encode train and test images
train_encoding_img_sa = stacked_encoder.predict(train_img, verbose=0)
test_encoding_img_sa = stacked_encoder.predict(test_img, verbose=0)

# Do some plotting
plt.figure(figsize=(10, 6))
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.title("Stacked Autoencoder - Training and Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()

# Display for each class, one random pair (real-reconstructed) image
plt.figure(figsize=(40, 12))
for class_label in range(10):
  indices = np.where(train_lbl == class_label)[0]
  selected_indices = np.random.choice(indices, 1, replace=False)
  for i, idx in enumerate(selected_indices):
    plt.subplot(3, 10, 10 + class_label + 1)
    plt.imshow(train_img[idx].reshape(28, 28), cmap="gray")
    plt.title(f"{class_names[class_label]} \n", fontweight='bold')
    plt.text(0.5, -0.1, f"Real Image", size=10, ha="center", transform=plt.gca().transAxes)
    plt.axis("off")
    plt.subplot(3, 10, (10 + class_label + 1) + 10)
    reconstructed_image = stacked_decoder.predict(train_encoding_img_sa[idx].reshape(1, -1), verbose=0).reshape(28, 28)
    plt.imshow(reconstructed_image, cmap="gray")
    plt.text(0.5, -0.1, f"Reconstructed Image, by SAE", size=10, ha="center", transform=plt.gca().transAxes)
    plt.axis("off")
plt.subplots_adjust(wspace=0.5, hspace=0)
plt.show()

# DBSCAN - Encoded Images
model = DBSCAN(eps=3, min_samples=9)
(model_name, execution_time, n_clusters, predicted_labels_sa_dbscan, calinski_harabasz_idx, davies_bouldin_idx, silhouette_idx, ari_idx) = doClustering(model, test_encoding_img_sa)
add_row("SA", model_name, training_time, execution_time, n_clusters, calinski_harabasz_idx, davies_bouldin_idx, silhouette_idx, ari_idx)
# MiniBatcgKmeans - Encoded Images
model = MiniBatchKMeans(n_clusters=10, random_state=42, batch_size=200, n_init=5)
(model_name, execution_time, n_clusters, predicted_labels_sa_kmeans, calinski_harabasz_idx, davies_bouldin_idx, silhouette_idx, ari_idx) = doClustering(model, test_encoding_img_sa)
add_row("SA", model_name, training_time, execution_time, n_clusters, calinski_harabasz_idx, davies_bouldin_idx, silhouette_idx, ari_idx)
# AgglomerativeClustering - Encoded Images
model = AgglomerativeClustering(n_clusters=10)
(model_name, execution_time, n_clusters, predicted_labels_sa_hier, calinski_harabasz_idx, davies_bouldin_idx, silhouette_idx, ari_idx) = doClustering(model, test_encoding_img_sa)
add_row("SA", model_name, training_time, execution_time, n_clusters, calinski_harabasz_idx, davies_bouldin_idx, silhouette_idx, ari_idx)

"""# **Clustering in encoded images (by LDA)**"""

# Applying Linear Discriminant Analysis
lda = LinearDiscriminantAnalysis(n_components=9)
start_time = time.time()
lda.fit(train_img, train_lbl)
end_time = time.time()
training_time = round((end_time-start_time)*1000)
# print(f"Total Explained Variance: {np.sum(lda.explained_variance_ratio_) * 100:.2f}%")
# Encode test images
test_encoding_img_lda = lda.transform(test_img)

# Do some plotting
explained_variance_ratio = lda.explained_variance_ratio_
total_explained_variance = np.sum(explained_variance_ratio)
cumulative_variance_ratio = np.cumsum(explained_variance_ratio)
x_axis = np.arange(1, len(explained_variance_ratio) + 1)
plt.figure(figsize=(10, 6))
plt.plot(x_axis, cumulative_variance_ratio, marker="o", linestyle="-", color="b")
plt.title("Explained Variance (LDA)")
plt.xlabel("Component ID")
plt.ylabel("Explained Variance")
plt.grid(True)
plt.show()

# DBSCAN - Encoded images
model = DBSCAN(eps=2, min_samples=3)
(model_name, execution_time, n_clusters, predicted_labels_lda_dbscan, calinski_harabasz_idx, davies_bouldin_idx, silhouette_idx, ari_idx) = doClustering(model, test_encoding_img_lda)
add_row("LDA", model_name, training_time, execution_time, n_clusters, calinski_harabasz_idx, davies_bouldin_idx, silhouette_idx, ari_idx)
# MiniBatcgKmeans - Encoded images
model = MiniBatchKMeans(n_clusters=10, random_state=42, batch_size=200, n_init=5)
(model_name, execution_time, n_clusters, predicted_labels_lda_kmeans, calinski_harabasz_idx, davies_bouldin_idx, silhouette_idx, ari_idx) = doClustering(model, test_encoding_img_lda)
add_row("LDA", model_name, training_time, execution_time, n_clusters, calinski_harabasz_idx, davies_bouldin_idx, silhouette_idx, ari_idx)
# AgglomerativeClustering - Encoded images
model = AgglomerativeClustering(n_clusters=10)
(model_name, execution_time, n_clusters, predicted_labels_lda_hier, calinski_harabasz_idx, davies_bouldin_idx, silhouette_idx, ari_idx) = doClustering(model, test_encoding_img_lda)
add_row("LDA", model_name, training_time, execution_time, n_clusters, calinski_harabasz_idx, davies_bouldin_idx, silhouette_idx, ari_idx)

"""# **DataFrame Storage**"""

drive.mount("/content/drive")
file_path = "/content/drive/My Drive/ml2-outcomes.xlsx"
df_outcomes.to_excel(file_path, index=False)

df_outcomes

"""# **Evaluation**"""

# We consider best dimensionallity reduction τechnique and clustering model combination for specific dataset is LinearDiscriminantAnalysis and MiniBatcgKmeans
predicted_labels = predicted_labels_lda_kmeans
num_clusters = np.max(predicted_labels) + 1

# Plotting cluster distribution
fig, axs = plt.subplots(5, 2, figsize=(20, 15))
for cluster in range(num_clusters):
    cluster_indices = np.where(predicted_labels == cluster)
    cluster_labels = test_lbl[cluster_indices]
    class_counts = np.bincount(cluster_labels, minlength=10)
    total_samples = len(cluster_labels)
    class_percentages = class_counts / total_samples * 100
    majority_class = np.argmax(class_counts)
    majority_class_name = class_names[majority_class]
    row = cluster // 2
    col = cluster % 2
    axs[row, col].bar(class_names, class_counts, alpha=0.7, color="skyblue")  # Use class names on the x-axis
    axs[row, col].bar(majority_class_name, class_counts[majority_class], color="orange", label=f"Majority Class: {majority_class_name}")
    axs[row, col].set_ylim(0, 1000)
    for i, count in enumerate(class_counts):
        axs[row, col].text(i, count + 1, f"{class_percentages[i]:.2f}%", ha="center", va="bottom")
    axs[row, col].set_title(f"Distribution of Classes in Cluster {cluster}")
    axs[row, col].set_xlabel("Class")
    axs[row, col].set_ylabel("Frequency")
    axs[row, col].xaxis.set_major_locator(ticker.FixedLocator(range(len(class_names))))
    axs[row, col].set_xticklabels(class_names, ha="center")
    axs[row, col].legend()
plt.tight_layout()
plt.show()

# Plot random images from each cluster
num_samples_per_cluster = 20
cluster_samples = [[] for _ in range(num_clusters)]
for i in range(len(predicted_labels)):
  label = predicted_labels[i]
  cluster_samples[label].append(test_img[i])
for cluster in range(num_clusters):
  plt.figure(figsize=(num_clusters * 2, 2))
  for i in range(num_samples_per_cluster):
    random_sample = random.choice(cluster_samples[cluster])
    plt.subplot(1, num_samples_per_cluster, i + 1)
    plt.imshow(np.reshape(random_sample, (28, 28)), cmap="gray")
    plt.axis("off")
  plt.suptitle(f"Cluster {cluster }", x=0.1, y=0.525, fontsize=9)
plt.show()
