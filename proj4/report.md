## Section 1: K-Means Clustering

### Discussion

On the blobs dataset, both `random` and `k-means++` initialization produced identical inertia and silhouette scores across all K values, with the only exception being a negligible difference at K=4 (inertia 505.89 vs 505.96). This indicates that for well-separated, spherical clusters, random initialization is sufficient to find the global optimum reliably. On the ellipses dataset the results were again identical between both methods, suggesting that for this particular dataset neither method had a meaningful advantage. In general, `k-means++` is preferred because it spreads initial centroids across the data, reducing the risk of poor convergence on harder datasets — the blobs data was simply too easy to show a difference. Based on the elbow curve, the optimal K for the blobs dataset is clearly 3, as inertia drops sharply from K=2 to K=3 and then flattens from K=3 to K=4. This is confirmed by the silhouette scores, where K=3 achieves the highest score of 0.6802. On the ellipses dataset the elbow curve shows a more linear decrease without a clear bend, and silhouette scores are low across all K values (~0.45–0.47), indicating that K-Means struggles to identify the true cluster structure. This is expected because K-Means assumes spherical clusters and minimizes squared Euclidean distance, which does not capture elongated or overlapping shapes well. K-Means performs well on the blobs dataset and is not sensitive to initialization there, but it fails to meaningfully separate the elliptical clusters.

---

## K-Means: Blobs Dataset

| K | Init       | Silhouette Score | Inertia |
|---|------------|------------------|---------|
| 2 | k-means++  | 0.6119           | 1898.00 |
| 2 | random     | 0.6119           | 1898.00 |
| 3 | k-means++  | 0.6802           | 582.07  |
| 3 | random     | 0.6802           | 582.07  |
| 4 | k-means++  | 0.5674           | 505.89  |
| 4 | random     | 0.5632           | 505.96  |

## K-Means: Ellipses Dataset

| K | Init       | Silhouette Score | Inertia |
|---|------------|------------------|---------|
| 2 | k-means++  | 0.4701           | 1348.22 |
| 2 | random     | 0.4701           | 1348.22 |
| 3 | k-means++  | 0.4513           | 942.35  |
| 3 | random     | 0.4513           | 942.35  |
| 4 | k-means++  | 0.4598           | 592.88  |
| 4 | random     | 0.4598           | 592.88  |

## Section 2: Gaussian Mixture Models

### Discussion

On the blobs dataset, the lowest BIC was achieved at K=3 with `diag` covariance (2418.49), meaning the diagonal model was the best fit after penalizing for complexity. This makes sense because the blobs are spherical and axis-aligned — a diagonal covariance is sufficient to capture their structure, and BIC penalizes the additional parameters that `full` covariance introduces unnecessarily. On the ellipses dataset, `full` covariance at K=3 achieved the lowest BIC (2306.33), significantly outperforming `diag` at the same K (2528.40). The `full` covariance type allows the Gaussians to rotate and stretch in any direction, which is necessary to fit the tilted elliptical clusters in that dataset. The `diag` covariance is restricted to axis-aligned ellipses, so it cannot model clusters 1 and 2 which have diagonal orientations — this is reflected in the large BIC gap of over 220 points. In both cases K=3 was the optimal number of components, matching the ground truth of how the data was generated.

---

## GMM: Blobs Dataset

| K | Covariance | BIC     | Log Likelihood |
|---|------------|---------|----------------|
| 2 | full       | 2551.88 | -4.15          |
| 2 | diag       | 2584.33 | -4.22          |
| 3 | full       | 2434.02 | -3.90          |
| 3 | diag       | 2418.49 | -3.90          |
| 4 | full       | 2465.66 | -3.89          |
| 4 | diag       | 2444.03 | -3.89          |

## GMM: Ellipses Dataset

| K | Covariance | BIC     | Log Likelihood |
|---|------------|---------|----------------|
| 2 | full       | 2348.81 | -3.81          |
| 2 | diag       | 2577.89 | -4.21          |
| 3 | full       | 2306.33 | -3.68          |
| 3 | diag       | 2528.40 | -4.08          |
| 4 | full       | 2346.23 | -3.69          |
| 4 | diag       | 2522.48 | -4.02          |

---

## Section 3: Comparison and Analysis

### Discussion

On the blobs dataset, both K-Means and GMM produced clear cluster separations, which is expected given the data is well-separated and spherical. On the ellipses dataset, GMM with full covariance produced far clearer and more accurate separations than K-Means, which incorrectly forced spherical boundaries onto elongated clusters. K-Means is preferable when the clusters are expected to be roughly spherical and equal in size, the dataset is large and speed matters, or interpretability and simplicity are priorities — it is deterministic and computationally cheaper. GMM is preferable when clusters may be elliptical or have different shapes and sizes, when soft (probabilistic) assignments are useful, or when a principled model selection criterion like BIC is needed to choose K. Conceptually, K-Means is a deterministic algorithm that hard-assigns each point to exactly one cluster by minimizing distance to centroids. GMM is a probabilistic model where each point has a probability of belonging to each cluster, and the EM algorithm iteratively estimates both the cluster parameters and the assignments. This means GMM captures uncertainty in cluster membership and can model more complex data distributions, at the cost of additional complexity and computational overhead.