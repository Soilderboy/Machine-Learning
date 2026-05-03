"""
Structure:

Dataset generation

class KMeansExperiment
    fit/run for K in {2,3,4}
    plot scatter (per K + side by side init comparison)
    plot elbow curve
    compute/print silhouette table

class GMMExperiment
    fit/run for K in {2,3,4} x covariance types
    plot scatter (per K + side by side cov comparison)
    compute/print BIC + log-likelihood table




"""
#Dataset 1: Spherical Clusters make_blobs
from sklearn.datasets import make_blobs
X_blobs, _ = make_blobs(n_samples=300, n_features=2, centers=3, random_state=40)

#Dataset 2: Non-Spherical Clusters (elliptical) 
import numpy as np
np.random.seed(40)

#Cluster 1:
mean1 = [-2, 2]
cov1 = [[3,2], [2,1]] #Diagonal orientation
cluster1= np.random.multivariate_normal(mean1, cov1, 100)

#Cluster 2:
mean2= [2,-2]
cov2 = [[1,-2], [-2,3]] #Opposite diagonal orientation
cluster2 = np.random.multivariate_normal(mean2,cov2,100)

#Cluster 3: narrow vertical with overlap
mean3 = [0,0]
cov3 = [[0.3, 0], [0,4]]
cluster3 = np.random.multivariate_normal(mean3,cov3, 100)

X_ellipses = np.vstack([cluster1, cluster2, cluster3])

# K Means Clustering
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
class KMeansExperiment:
    def __init__(self, X):
        self.X = X
        self.k_values = [2,3,4]
        self.results = {}
        self.init_methods = ['k-means++', 'random']
    
    def run(self):
        for init_method in self.init_methods:
            for k in self.k_values:
                model = KMeans(n_clusters=k, init=init_method, n_init=10, random_state=40)
                model.fit(self.X)
                self.results[(k, init_method)] = model

    def plot_clusters(self):
        """One scatter plot per K using the best init (lowest inertia) for each K."""
        fig, axes = plt.subplots(1, len(self.k_values), figsize=(15, 5))
        for i, k in enumerate(self.k_values):
            # pick whichever init method got the lower inertia for this K
            model_random = self.results[(k, 'random')]
            model_plus = self.results[(k, 'k-means++')]
            if model_random.inertia_ < model_plus.inertia_:
                best_model = model_random
            else:
                best_model = model_plus

            x_coords = self.X[:, 0]
            y_coords = self.X[:, 1]
            labels = best_model.labels_
            axes[i].scatter(x_coords, y_coords, c=labels, cmap='viridis', s=50)

            # plot the centroids as red X markers
            cx = best_model.cluster_centers_[:, 0]
            cy = best_model.cluster_centers_[:, 1]
            axes[i].scatter(cx, cy, c='red', marker='X', s=200, label='Centroids')

            axes[i].set_title(f'K={k}')
            axes[i].set_xlabel('Feature 1')
            axes[i].set_ylabel('Feature 2')
            axes[i].legend()
        plt.suptitle('K-Means Clustering')
        plt.tight_layout()
        plt.show()

    def plot_init_comparison(self, k=3):
        """Side-by-side comparison of random vs k-means++ for a given K."""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        for i, init in enumerate(self.init_methods):
            model = self.results[(k, init)]

            x_coords = self.X[:, 0]
            y_coords = self.X[:, 1]
            labels = model.labels_
            axes[i].scatter(x_coords, y_coords, c=labels, cmap='viridis', s=50)

            cx = model.cluster_centers_[:, 0]
            cy = model.cluster_centers_[:, 1]
            axes[i].scatter(cx, cy, c='red', marker='X', s=200, label='Centroids')

            axes[i].set_title(f'init={init}  |  Inertia={model.inertia_:.2f}')
            axes[i].set_xlabel('Feature 1')
            axes[i].set_ylabel('Feature 2')
            axes[i].legend()
        plt.suptitle(f'Initialization Comparison (K={k})')
        plt.tight_layout()
        plt.show()

    def plot_elbow_curve(self):
        """Plot ICSS vs K using the best model (lowest inertia) per K."""
        inertia_values = []
        for k in self.k_values:
            model_random = self.results[(k, 'random')]
            model_plus = self.results[(k, 'k-means++')]
            best_inertia = min(model_random.inertia_, model_plus.inertia_)
            inertia_values.append(best_inertia)
        plt.figure(figsize=(8, 5))
        plt.plot(self.k_values, inertia_values, marker='o')
        plt.title('Elbow Curve')
        plt.xlabel('Number of Clusters (K)')
        plt.ylabel('Inertia (ICSS)')
        plt.xticks(self.k_values)
        plt.grid()
        plt.show()
    
    def silhouette_table(self):
        """Print silhouette scores for each K and init method."""
        print(f"{'K':<6} {'Init':<12} {'Silhouette Score':<20} {'Inertia'}")
        print("-" * 52)
        for k in self.k_values:
            for init in self.init_methods:
                model = self.results[(k, init)]
                score = silhouette_score(self.X, model.labels_)
                print(f"{k:<6} {init:<12} {score:<20.4f} {model.inertia_:.2f}")

# GMM Clustering
from sklearn.mixture import GaussianMixture

class GMMExperiment:
    def __init__(self, X):
        self.X = X
        self.k_values = [2,3,4]
        #full means general elliptical covariance (non-spherical)
        #diag means diagonal covariance (axis-aligned ellipses)
        self.cov_types = ['full', 'diag']
        self.results = {}
    
    def run(self):
        for cov_type in self.cov_types:
            for k in self.k_values:
                model = GaussianMixture(n_components=k, covariance_type=cov_type, n_init=10, random_state=40)
                model.fit(self.X)
                self.results[(k, cov_type)] = model
    
    def plot_clusters(self):
        """One scatter plot per K using best cov type (lowest BIC) for each K."""
        fig, axes = plt.subplots(1, len(self.k_values), figsize=(15, 5))
        for i, k in enumerate(self.k_values):
            # pick whichever covariance type got the lower BIC for this K
            model_full = self.results[(k, 'full')]
            model_diag = self.results[(k, 'diag')]
            if model_full.bic(self.X) < model_diag.bic(self.X):
                best_model = model_full
            else:
                best_model = model_diag

            labels = best_model.predict(self.X)
            x_coords = self.X[:, 0]
            y_coords = self.X[:, 1]
            axes[i].scatter(x_coords, y_coords, c=labels, cmap='viridis', s=50)
            axes[i].set_title(f'K={k}')
            axes[i].set_xlabel('Feature 1')
            axes[i].set_ylabel('Feature 2')
        plt.suptitle('GMM Clustering')
        plt.tight_layout()
        plt.show()
    
    def plot_cov_comparison(self, k=3):
        """Side by side comparison of full vs diag covariance for given K"""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        for i, cov in enumerate(self.cov_types):
            model = self.results[(k, cov)]
            labels = model.predict(self.X)
            bic_score = model.bic(self.X)

            x_coords = self.X[:, 0]
            y_coords = self.X[:, 1]
            axes[i].scatter(x_coords, y_coords, c=labels, cmap='viridis', s=50)
            axes[i].set_title(f'covariance={cov} | BIC={bic_score:.2f}')
            axes[i].set_xlabel('Feature 1')
            axes[i].set_ylabel('Feature 2')
        plt.suptitle(f'Covariance Comparison (K={k})')
        plt.tight_layout()
        plt.show()

    def bic_loglikelihood_table(self):
        """Print BIC and log likelihood for each K and covariance type"""
        print(f"{'K':<6} {'Covariance':<12} {'BIC':<20} {'Log Likelihood'}")
        print("-" * 60)
        for k in self.k_values:
            for cov in self.cov_types:
                model = self.results[(k, cov)]
                print(f"{k:<6} {cov:<12} {model.bic(self.X):<20.2f} {model.score(self.X):.2f}")


if __name__ == "__main__":
    #run kmeans experiment on both datasets

    kmeans_blobs = KMeansExperiment(X_blobs)
    kmeans_blobs.run()
    kmeans_blobs.plot_clusters()
    kmeans_blobs.plot_init_comparison(k=3)
    kmeans_blobs.plot_elbow_curve()
    kmeans_blobs.silhouette_table()

    kmeans_ellipses = KMeansExperiment(X_ellipses)
    kmeans_ellipses.run()
    kmeans_ellipses.plot_clusters()
    kmeans_ellipses.plot_init_comparison(k=3)
    kmeans_ellipses.plot_elbow_curve()
    kmeans_ellipses.silhouette_table()
    

    #run GMM experiment on both datasets
    gmm_blobs = GMMExperiment(X_blobs)
    gmm_blobs.run()
    gmm_blobs.plot_clusters()
    gmm_blobs.plot_cov_comparison(k=3)
    gmm_blobs.bic_loglikelihood_table()

    gmm_ellipses = GMMExperiment(X_ellipses)
    gmm_ellipses.run()
    gmm_ellipses.plot_clusters()
    gmm_ellipses.plot_cov_comparison(k=3)
    gmm_ellipses.bic_loglikelihood_table()
