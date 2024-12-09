# Import necessary libraries
import numpy as np 
from sklearn.metrics import pairwise_distances

import zadu

MACHINE_EPSILON = np.finfo(np.float64).eps

class Metrics():
    """
    Class for computing various stress metrics between high-dimensional and low-dimensional data.
    """
    def __init__(self, X, Y, scaling_factors = np.linspace(MACHINE_EPSILON, 20, 100)):
        """
        Initialize the Metrics class with high-dimensional data X and low-dimensional data Y.
        Compute pairwise distances within X and Y.
        Also compute pairwise distances within Y for different scales in scaling_factors
        """
        self.X = X 
        self.Y = Y 

        self.dX = pairwise_distances(X)
        self.dY = pairwise_distances(Y)

        self.Y_batch = self.Y[None , :, :] * scaling_factors[ :, None, None]
        self.dY_batch = np.array([pairwise_distances(Y) for Y in self.Y_batch])

    def setY(self,Y):
        """
        Update low-dimensional data Y and compute pairwise distances within Y.
        """
        self.Y = Y 
        self.dY = pairwise_distances(Y)

    def compute_raw_stress(self):
        """
        Compute raw stress between pairwise distances of X and Y.
        """
        return np.sum(np.square(self.dX - self.dY)) / 2

    def compute_normalized_stress(self,alpha=1.0):  
        """
        Compute normalized stress between X and alpha*Y using zadu's stress measure.
        """      
        from zadu.measures import stress
        stressScore = stress.measure(self.X,alpha * self.Y,(self.dX, alpha * self.dY))
        return stressScore['stress']

    def compute_scale_normalized_stress(self,return_alpha=False):
        """
        Compute scale-normalized stress between pairwise distances of X and Y.
        Optimal scaling factor alpha is computed as well.
        """
        D_low_triu = self.dY[np.triu_indices(self.dY.shape[0], k=1)]
        D_high_triu = self.dX[np.triu_indices(self.dX.shape[0], k=1)]
        alpha = np.sum(D_low_triu * D_high_triu) / np.sum(np.square(D_low_triu))
        if return_alpha:
            return self.compute_normalized_stress(alpha), alpha
        return self.compute_normalized_stress(alpha)

    def compute_kruskal_stress(self):
        """
        Compute Kruskal's non-metric stress between pairwise distances of X and Y. Invariant to scale of Y.
        """

        dij = self.dX[np.triu_indices(self.dX.shape[0], k=1)]
        xij = self.dY[np.triu_indices(self.dY.shape[0], k=1)]

        # Find the indices of dij that when reordered, would sort it. Apply to both arrays
        sorted_indices = np.argsort(dij)
        dij = dij[sorted_indices]
        xij = xij[sorted_indices]

        from sklearn.isotonic import IsotonicRegression
        hij = IsotonicRegression().fit(dij, xij).predict(dij)

        raw_stress = np.sum(np.square(xij - hij))
        norm_factor = np.sum(np.square(xij))

        kruskal_stress = np.sqrt(raw_stress / norm_factor)
        return kruskal_stress

    def compute_shepard_correlation(self):
        """
        Compute Shepard's correlation between pairwise distances of X and Y using zadu's spearman_rho measure.
        Invariant to scale of Y.
        """
        from zadu.measures import spearman_rho
        shepardCorr = spearman_rho.measure(self.X,self.Y,(self.dX,self.dY))
        return shepardCorr['spearman_rho']
    
    def compute_kl_divergence(self, perplexity=30):
        """
        Compute joint probability distributions of X and Y à la t-SNE and compute the KL divergence between them

        Parameters
        ----------
        perplexity : Perplexity as described in Hinton and Roweis (2002) https://www.cs.toronto.edu/~hinton/absps/sne.pdf
        """
        # High-dimensional probability space
        n_samples, _ = self.X.shape
        conditional_P = self._conditional_probabilities(perplexity)
        P = (conditional_P + conditional_P.T) / (2 * n_samples)    

        # Low-dimensional probability space
        Q = np.where(self.dY != 0, (np.square(self.dY) + 1) ** -1, 0)
        Q = Q / Q.sum()

        # Clip at machine epsilon to fix precision errors
        epsilon = np.finfo(np.double).eps
        P = np.clip(P, epsilon, 1)
        Q = np.clip(Q, epsilon, 1)

        # KL Divergence
        log_P = np.where(P > 0, np.log2(P), 0)
        log_Q = np.where(Q > 0, np.log2(Q), 0)
        kl_divergence = (P * (log_P - log_Q)).sum()


        return kl_divergence
    
    def compute_kl_divergences(self, perplexity):
        """
        Compute joint probability distributions of X and Y à la t-SNE and compute the KL divergence between them

        Parameters
        ----------
        perplexity : float
            Perplexity as described in Hinton and Roweis (2002) https://www.cs.toronto.edu/~hinton/absps/sne.pdf
        scaling_factors : np.ndarray
            factors by which to scale the low-dimensional space
        """

        # High-dimensional probability space
        n_samples = self.X.shape[0]
        conditional_P = self._conditional_probabilities(perplexity)
        P = ((conditional_P + conditional_P.T) / (2 * n_samples))

        # Compute low-dimensional probability space for all Y_batch
        Q_batch = np.where(
            self.dY_batch != 0,
            (np.square(self.dY_batch) + 1) ** -1,
            0,
        )
        Q_batch = Q_batch / Q_batch.sum(axis=(1, 2), keepdims=True)

        # Clip at machine epsilon to fix precision errors
        P = np.clip(P, MACHINE_EPSILON, 1)
        Q_batch = np.clip(Q_batch, MACHINE_EPSILON, 1)

        # KL Divergence for all Y_batch
        log_P = np.log2(P)
        log_Q_batch = np.log2(Q_batch)

        kl_divergences = (P * (log_P - log_Q_batch)).sum(axis=(1, 2))
        return kl_divergences


    def _conditional_probabilities(self, perplexity, steps=40):
        """
        Calculate conditional probability matrix P corresponding to SNE algorithm
        
        P[i, j] = P (j | i)
        
        Parameters
        ----------
        perplexity : Perplexity as described in Hinton and Roweis (2002) https://www.cs.toronto.edu/~hinton/absps/sne.pdf
        steps : Number of steps for binary search for variances of Gaussian distributions
        """
        n_samples, _ = self.X.shape
        desired_entropy = np.full((n_samples, 1), np.log2(perplexity))
        beta = np.ones((n_samples, 1)) # (2 * var_i ** 2) in the exponent of the Gaussian distribution
        beta_min = beta_min = np.zeros((n_samples, 1), dtype=np.float64)
        beta_max = np.full((n_samples, 1), np.inf)

        # Binary Search
        for _ in range(steps):
            # Create conditional probability distributions
            P = np.where(self.dX != 0, np.exp(-1 * np.square(self.dX) / beta), 0)
            row_sums = np.maximum(P.sum(axis=1, keepdims=True), MACHINE_EPSILON)
            P = P / row_sums

            # Calculate entropy
            entropy = -1 * (P * np.log2(P, where=(P > MACHINE_EPSILON))).sum(axis=1, keepdims=True)
            entropy_diff = entropy - desired_entropy
    

            # Stop if variances sufficiently match perplexity
            if np.all(np.abs(entropy_diff) < 1e-5):
                break
            

            # Binary search update:

            should_increase_beta = entropy_diff < 0.0

            # Update beta_min and beta_max
            beta_min = np.where(should_increase_beta, beta, beta_min)
            beta_max = np.where(should_increase_beta, beta_max, beta)

            beta_max_is_inf = (beta_max == np.inf)
        
            # Update beta

            mask = should_increase_beta & beta_max_is_inf
            beta[mask] = beta[mask] * 2

            mask = should_increase_beta & ~beta_max_is_inf
            beta[mask] = (beta[mask] + beta_max[mask]) / 2.0

            mask = ~should_increase_beta
            beta[mask] = (beta[mask] + beta_min[mask]) / 2.0

        return P


if __name__ == "__main__":
    """
    Main function to compute and plot normalized stress for a range of scaling factors.
    """
    X = np.load('datasets/auto-mpg.npy')
    Y = np.load('embeddings/auto-mpg-TSNE-0.npy')

    M = Metrics(X,Y)
    
    scale_opt, alpha_opt = M.compute_scale_normalized_stress(return_alpha=True)

    # Compute normalized stress for a range of scaling factors
    rrange = np.linspace(0,100,2000)
    norm_stress_scores = list()
    for alpha in rrange:
        M.setY(alpha * Y)
        norm_stress_scores.append(M.compute_normalized_stress())

    # Plot normalized stress scores and highlight the optimal scaling factor
    import pylab as plt 
    plt.plot(rrange, norm_stress_scores)
    plt.scatter(alpha_opt, scale_opt)
    plt.show()