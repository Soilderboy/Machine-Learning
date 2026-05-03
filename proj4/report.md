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