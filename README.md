# Evaluation-of-Dimensionality-Reduction-Techniques-and-Clustering-Models-on-Fashion-MNIST-Dataset
This work focuses on testing and evaluating various dimensionality reduction techniques and clustering models on image data. The Fashion-MNIST dataset, consisting of 60,000 training samples and 10,000 test samples of 28x28 grayscale images, is used for this purpose. Each image is labeled with a category from 0 to 9, representing different fashion items. The aim is to explore dimensionality reduction methods (PCA, SAE, LDA) and clustering models (DBSCAN, MiniBatch KMeans, Agglomerative Clustering), and evaluate their effectiveness using several performance metrics.

Key Achievements:
- Principal Component Analysis (PCA), Stacked Autoencoders (SAE), and Linear Discriminant Analysis (LDA) were tested for reducing the dimensionality of the image data.
- Demonstrated the effectiveness of each technique in retaining essential features of the dataset while reducing dimensionality.
- Implemented DBSCAN, MiniBatch KMeans, and Agglomerative Clustering for grouping similar images.
- Found MiniBatchKMeans, combined with LDA, to be the most efficient in terms of clustering quality and execution time.
- Utilized metrics like Calinski-Harabasz index, Davies-Bouldin index, Silhouette score, and Adjusted Rand score to assess the quality of clustering.
- The best combination of dimensionality reduction and clustering model was LDA with MiniBatch KMeans, which provided the best overall performance across all metrics.
- Demonstrated significant improvement in clustering uniformity and execution time after applying dimensionality reduction techniques.
