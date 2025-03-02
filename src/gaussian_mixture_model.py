# Import libraries
import numpy as np
import scipy

class GMM:
    def __init__(self, K, D):
        """
        Initialize Mixture of Gaussians with EM optimization.

        Args:
            K (int): Number of Gaussian components
            D (int): Dimensionality of the input data
        """
        self.K = K
        self.D = D

        np.random.seed(42) # Set seed for reproducibility

        # Initialize weights, means and variances
        self.weights = np.ones(self.K)
        self.means = np.random.randn(self.K, self.D) * 2
        self.variances = np.ones((self.K, self.D))

    def _log_multivariate_normal_pdf(self, X, mean, var):
        """
        Compute PDF of multivariate normal distribution for a given data matrix X and the parameters of the distribution.

        Args:
            X (np.ndarray): Input data shape (N, D)
            mean (np.ndarray): Mean of each dimension (D,)
            var (np.ndarray): Variance of each dimension (D,)

        Returns:
            prob (np.ndarray): Probabilities for each data point (N,)
        """
        diff = X - mean
        log_det = np.sum(np.log(var))
        quad = np.sum(diff * diff / var, axis=1)

        prob = 0.5 * (self.D * np.log(2 * np.pi) + log_det + quad)

        return prob
    
    def log_likelihood(self, X):
        """
        Compute log likelihood of the data

        Args:
            X (np.ndarray): Input data of shape (N, D)

        Return:
            ll (np.ndarray): log likelihood of shape ()
        """
        N = X.shape[0]
        log_probs = np.zeros((self.K, N))

        for k in range(self.K):
            log_probs[k] = self._log_multivariate_normal_pdf(X, self.means[k], self.variances[k])

        log_probs = log_probs + np.log(self.weights[k].reshape(-1, 1))

        log_likelihood = scipy.special.logsumexp(log_probs, axis=0)

        return np.sum(log_likelihood)
    
    def fit(self, X, max_iters=100, tol=1e-6):
        """
        Fit the model using EM algorithm

        Args:
            X (np.ndarray): Input data of shape (N, D)
            max_iters (int): Maximum numper of EM iterations
            tol (float): Convergence tolerance of log-likelihood

        Return:
            nll_history (list): nll losses at each iteration
        """
        N = X.shape[0]
        prev_ll = -np.inf
        nll_history = []

        for iter in range(max_iters):
            # E-step: Calculate responsibilities
            responsibilities = self._e_step(X)

            # M-step: Update parameters
            self._m_step(X, responsibilities)

            # Check convergence
            curr_ll = self.log_likelihood(X)
            nll_history.append(-curr_ll)
            if np.abs(curr_ll - prev_ll) < tol:
                print(f'Converged after {iter+1} iterations')
                break

            prev_ll = curr_ll

        return nll_history
    
    def _e_step(self, X):
        """
        Expectation step: compute responsibilities

        Args:
            X (np.ndarray): Input data of shape (N, D)

        Returns:
            responsibilities (np.ndarray): Responsibilities p(k|x): the probability that a given sample belongs to the k-th component of the mixture, of shape (N, K)
        """
        N = X.shape[0]
        log_resp = np.zeros((N, self.K)) #log(p(k|x))

        # Calculate log responsibilities
        for k in range(self.K):
            log_resp[:, k] = np.log(self.weights[k] + 1e-10) + self._log_multivariate_normal_pdf(X, self.means[k], self.variances[k])

        # Subtract max for numerical stability before exp
        log_resp_norm = log_resp - log_resp.max(axis=1, keepdims=True)
        responsibilities = np.exp(log_resp_norm)
        responsibilities /= responsibilities.sum(axis=1, keepdims=True) # Normalize

        return responsibilities
    
    def _m_step(self, X, responsibilities):
        """
        Maximization step: update parameters means, variances, and weights

        Args:
            X (np.ndarray): Input data of shape (N, D)
            responsibilities (np.ndarray): Responsibilities p(k|x): the probability that a given sample belongs to the k-th component of the mixture, of shape (N, K)
        """
        N = X.shape[0]
        Nk = responsibilities.sum(axis=0)

        # Update means
        #for k in range(self.K):
        #    self.means[k] = (responsibilities[:, k:k+1] * X).sum(axis=0) / Nk[k]
        self.means = (responsibilities.T @ X) / Nk[:, None]  # Shape: (K, D)

        # Update covariances
        #for k in range(self.K):
        #    diff = (X - self.means[k])**2
        #    self.variances[k] = np.sum(responsibilities[:, k:k+1] * diff, axis=0) / Nk[k]
        #
        #    # Add small diagonal term for numerical stability
        #    self.variances[k] += 1e-6
        squared_diff = (X[:, None, :] - self.means[None, :, :])**2 # Broadcasting and Element-wise square: (N, K, D)
        self.variances = (responsibilities[:, :, None] * squared_diff).sum(axis=0) / Nk[:, None]  # Shape: (K, D)
        self.variances += 1e-6

        # Update weights
        self.weights = Nk / N

    def sample(self, n_samples):
        """
        Generate samples form GMM

        Args:
            n_samples (int): Number of samples to be generated

        Return:
            samples (np.ndarray): Generated samples of shape (n_samples, D)
        """
        # Sample component indices
        component_indices = np.random.choice(self.K, size=n_samples, p=self.weights)

        # Generate samples from each selected component
        samples = np.zeros((n_samples, self.D))
        for i, k in enumerate(component_indices):
            samples[i] = self.means[k] + np.sqrt(self.variances[k] * np.random.randn(self.D))

        return samples

        

