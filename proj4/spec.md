CS 4375: Introduction to Machine Learning
Project 4: K-Means Clustering and Gaussian Mixture Models
1 Overview (Total: 100 points)
In this project, you will implement and compare two unsupervised machine learning algorithms using
scikit-learn: K-Means Clustering and Expectation-Maximization (EM) using Gaussian Mixture
Models. All visualizations must be clear, labeled, and color-coded for interpretability. The focus is on understanding algorithm behavior through visualization and quantitative metrics, not on extensive reporting. Important: Set a random seed at the beginning of your code for reproducibility (e.g., np.random.seed(40)).
2 Creating Your Own Dataset
In this project, you will generate and work with synthetic datasets. This is a key skill in machine learning: by
creating controlled data with known properties, you can carefully study how algorithms behave in different
scenarios. You will apply both K-Means and EM to the same two datasets to compare algorithm performance
on different cluster shapes.
Dataset 1: Spherical Clusters (make blobs)
Generate well-separated, spherical clusters:
1 from sklearn.datasets import make_blobs
2 X_blobs, _ = make_blobs(n_samples=300, n_features=2, centers=3, random_state
=40)
What this produces: 300 points in 2D with 3 clear, spherical cluster centers. This is the easy case
where K-Means should perform well.
Dataset 2: Non-Spherical Clusters (Elliptical)
Generate elongated, elliptical clusters with custom covariance structures:
1 import numpy as np
2
3 np.random.seed(40)
4
5 # Cluster 1:
6 mean1 = [-2, 2]
7 cov1 = [[3, 2], [2, 1]] # Diagonal orientation
8 cluster1 = np.random.multivariate_normal(mean1, cov1, 100)
9
1
10 # Cluster 2:
11 mean2 = [2, -2]
12 cov2 = [[1, -2], [-2, 3]] # Opposite diagonal orientation
13 cluster2 = np.random.multivariate_normal(mean2, cov2, 100)
14
15 # Cluster 3: narrow vertical with overlap
16 mean3 = [0, 0]
17 cov3 = [[0.3, 0], [0, 4]]
18 cluster3 = np.random.multivariate_normal(mean3, cov3, 100)
19
20 X_ellipses = np.vstack([cluster1, cluster2, cluster3])
What this produces: 300 points in 2D forming three elongated elliptical overlapping clusters. Each
cluster is stretched vertically with tight horizontal variance, creating non-spherical shapes. This is the challenging case where K-Means struggles because it assumes spherical clusters, while EM with full covariance
is highly likely to capture the elongated, elliptical structure of each cluster.
Part 1: K-Means Clustering [50 points]
• Implement K-Means clustering using sklearn.cluster.KMeans, which partitions data into K
clusters by minimizing the intra-cluster sum of squared (ICSS) distances.
• Apply K-Means to both the datasets and generate plots for K ∈ {2, 3, 4} using both initialization
strategies: init=’random’ and init=’k-means++’. Run multiple initializations for each
method and report results for the best run of each method. Use the best initialization for all visualizations and quantitative evaluations below.
• Visualization: [3 plots]
– Create one 2D scatter plot per K value (3 total). Use distinct colors for each cluster and mark
cluster centers with stars or X markers using model.cluster centers .
– For one K value (e.g., K = 3), create a side-by-side comparison showing random vs. k-means++
initialization to visualize the effect of initialization.
• Quantitative Evaluation:
– Compute the elbow curve: Plot intra-cluster sum of squared distances (ICSS) as a function of
K. This shows diminishing returns and helps identify the optimal K. To compute the elbow
method: loop through different K values and fit KMeans to each, storing kmeans.inertia
(the intra-cluster sum of squared distances). Then plot these inertia values against K and visually
identify where the curve “bends” or begins to flatten. Generate separate elbow curves for both
datasets.
– New topic: Explore and reportsilhouette scores(sklearn.metrics.silhouette score)
in a simple table for each K and initialization method. Higher scores indicate better-separated
clusters.
• Discussion:
– Which initialization strategy (random vs. k-means++) performs better? Why?
2
– Based on the elbow curve and silhouette scores, what is the optimal K?
– How well does K-Means work on spherical clusters? Is it sensitive to initialization?
Part 2: Gaussian Mixture Models using EM [50 points]
• Implement Expectation-Maximization (EM) using sklearn.mixture.GaussianMixture, which
models data as a mixture of K Gaussian distributions.
• Apply EM to the same datasets as K-means. Vary K ∈ {2, 3, 4} and covariance type:
– full: General elliptical covariance (non-spherical)
– diag: Diagonal covariance (axis-aligned ellipses)
For each configuration, run multiple initializations and report the best result. Use the best initialization
for all visualizations and quantitative evaluations.
• Visualization: [3 plots]
– Create one 2D scatter plot per K value (3 total), showing cluster assignments with distinct
colors.
– For one K value (e.g., K = 3), create a side-by-side comparison of full vs. diag covariance
types to show how covariance structure affects cluster shapes.
• Quantitative Evaluation:
– Report the Bayesian Information Criterion (BIC) (Use GaussianMixture.bic(X) to
compute BIC.) in a simple table for each K and covariance type. Lower BIC values indicate
better model fit while penalizing complexity. BIC is the primary criterion for selecting K in
GMMs.
– Report average log-likelihood using GaussianMixture.score(X) in the same table.
• Discussion:
– Based on BIC, what is the optimal K and covariance type?
– How do the full and diag covariance types differ in their ability to model the data?
Part 3: Comparison and Analysis [Included in report]
• In your report, briefly compare K-Means and EM on the same dataset:
– Which algorithm produces clearer cluster separations for this data?
– When would you choose K-Means over EM and vice versa?
– How do deterministic (K-Means) and probabilistic (EM) approaches differ conceptually?
3
What to Submit
Submit a single zip file containing:
• A PDF report (maximum 6 pages), which includes:
– Two simple tables: One summarizing K-Means results (K, silhouette scores, ICSS by init
method) and one summarizing GMM results (K, covariance type, BIC, log-likelihood).
– Key visualizations: All plots described above. Ensure plots are large, clear, and labeled with
axis titles and legends.
– Concise discussions: 1 paragraph per section answering the bulleted questions above.
• Clean, well-commented Python code (single notebook or .py file) using scikit-learn, matplotlib,
and numpy. Code should be readable and include comments explaining key steps.
• Optional: A short summary of any AI tools/prompts used to complete the project.
Resources
• https://scikit-learn.org/stable/modules/clustering.html#k-means
• https://scikit-learn.org/stable/modules/mixture.html
• https://numpy.org/
