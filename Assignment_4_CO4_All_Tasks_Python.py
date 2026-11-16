"""
SIMATS ENGINEERING
DEEP LEARNING - ASSIGNMENT 4 (CO4)
Complete Python practice and verification program.

TASK 1: Autoencoder diagnosis and repair
TASK 2: PCA and linear autoencoder
TASK 3: Factor Analysis EM and VAE architecture
TASK 4: Causal self-attention
TASK 5: Weighted DAG traversal, Gibbs sampling, covariance
TASK 6: Unsupervised anomaly detection

IMPORTANT:
The assignment brief requires language-independent pseudocode.
Use this Python file for practice/verification, not as a replacement
for the required pseudocode submission.

Dependencies:
    pip install numpy
Optional dependency for the VAE demonstration:
    pip install torch
"""

import numpy as np

np.set_printoptions(precision=4, suppress=True)


# ============================================================
# TASK 1 - AUTOENCODER DIAGNOSIS AND REPAIR
# ============================================================

def sigmoid(x):
    """Numerically stable sigmoid."""
    x = np.clip(x, -500, 500)
    return 1.0 / (1.0 + np.exp(-x))


def autoencoder_forward(X, W1, b1, W2, b2):
    """
    Encoder: input -> latent representation
    Decoder: latent representation -> reconstruction
    """
    hidden = sigmoid(X @ W1 + b1)
    reconstructed = sigmoid(hidden @ W2 + b2)
    return hidden, reconstructed


def reconstruction_loss(X, reconstructed):
    """Mean squared reconstruction error."""
    return float(np.mean((X - reconstructed) ** 2))


def task1():
    print("\n" + "=" * 70)
    print("TASK 1: AUTOENCODER DIAGNOSIS AND REPAIR")
    print("=" * 70)

    X = np.array([
        [0.1, 0.2, 0.8, 0.9],
        [0.2, 0.3, 0.7, 0.8],
        [0.9, 0.8, 0.2, 0.1],
        [0.8, 0.9, 0.3, 0.2],
    ], dtype=float)

    rng = np.random.default_rng(42)
    input_size, hidden_size = 4, 2

    W1 = rng.normal(0, 0.1, (input_size, hidden_size))
    b1 = np.zeros(hidden_size)
    W2 = rng.normal(0, 0.1, (hidden_size, input_size))
    b2 = np.zeros(input_size)

    hidden, reconstructed = autoencoder_forward(X, W1, b1, W2, b2)

    print("Input:\n", X)
    print("\nEncoded representation:\n", hidden)
    print("\nReconstructed output:\n", reconstructed)
    print("\nReconstruction MSE:", reconstruction_loss(X, reconstructed))

    print("\nRepair principles:")
    print("- Encoder maps the input to a lower-dimensional latent vector.")
    print("- Decoder maps the latent vector back to the original input size.")
    print("- Encoder and decoder matrix dimensions must be compatible.")
    print("- Reconstruction loss compares the input and reconstructed output.")
    print("- Training must update weights to reduce reconstruction loss.")


# ============================================================
# TASK 2 - PCA AND LINEAR AUTOENCODER
# ============================================================

def task2():
    print("\n" + "=" * 70)
    print("TASK 2: PCA AND LINEAR AUTOENCODER")
    print("=" * 70)

    labels = ["A", "B", "C", "D", "E"]
    X = np.array([
        [2, 80],
        [1, 35],
        [2, 60],
        [3, 40],
        [1, 25],
    ], dtype=float)

    mean = np.mean(X, axis=0)
    X_centered = X - mean
    covariance = np.cov(X_centered, rowvar=False)

    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    # Retain one principal component.
    components = eigenvectors[:, :1]
    reduced = X_centered @ components
    reconstructed = reduced @ components.T + mean

    print("Original data:")
    for label, row in zip(labels, X):
        print(f"{label}: {row}")

    print("\nFeature mean:", mean)
    print("\nCovariance matrix:\n", covariance)
    print("\nEigenvalues:", eigenvalues)
    print("\nFirst principal component:\n", components)
    print("\nOne-dimensional PCA representation:\n", reduced)
    print("\nPCA reconstruction:\n", reconstructed)

    print("\nLinear autoencoder comparison:")
    print("Input dimension = 2; latent dimension = 1; output dimension = 2.")
    print("Use linear activations and mean squared reconstruction loss.")
    print("A suitably trained linear autoencoder learns the PCA subspace")
    print("when trained with squared reconstruction error under standard conditions.")


# ============================================================
# TASK 3 - FACTOR ANALYSIS EM AND VAE
# ============================================================

def factor_analysis_one_em_iteration(X, W, mu, Psi):
    """
    One standard EM iteration for factor analysis.

    Model:
        x = W z + mu + epsilon
        z ~ N(0, I)
        epsilon ~ N(0, Psi)

    Psi is constrained to be diagonal.
    """
    n, p = X.shape
    k = W.shape[1]

    Psi_inv = np.linalg.inv(Psi)
    M = np.linalg.inv(np.eye(k) + W.T @ Psi_inv @ W) @ W.T @ Psi_inv

    X_centered = X - mu
    Ez = X_centered @ M.T

    posterior_cov = np.eye(k) - M @ W
    Ezz_sum = np.zeros((k, k))

    for i in range(n):
        zi = Ez[i].reshape(k, 1)
        Ezz_sum += posterior_cov + zi @ zi.T

    W_new = (X_centered.T @ Ez) @ np.linalg.inv(Ezz_sum)

    # Update mean and diagonal uniquenesses.
    mu_new = np.mean(X - Ez @ W_new.T, axis=0)
    residual = X - mu_new - Ez @ W_new.T
    Psi_new = np.diag(np.maximum(np.mean(residual ** 2, axis=0), 1e-8))

    return Ez, W_new, mu_new, Psi_new


def task3():
    print("\n" + "=" * 70)
    print("TASK 3: FACTOR ANALYSIS EM AND VAE")
    print("=" * 70)

    labels = ["P", "Q", "R", "S"]
    X = np.array([
        [2, 12],
        [3, 20],
        [4, 25],
        [5, 30],
    ], dtype=float)

    W = np.array([[1.0], [1.0]])
    mu = np.mean(X, axis=0)
    Psi = np.eye(2)

    print("Observations:\n", X)
    print("\nInitial W:\n", W)
    print("\nInitial mean:", mu)
    print("\nInitial Psi:\n", Psi)

    Ez, W_new, mu_new, Psi_new = factor_analysis_one_em_iteration(
        X, W, mu, Psi
    )

    print("\nE-step expected latent values:\n", Ez)
    print("\nUpdated loading matrix W:\n", W_new)
    print("\nUpdated mean mu:", mu_new)
    print("\nUpdated diagonal Psi:\n", Psi_new)

    print("\nVAE architecture:")
    print("Input -> Encoder -> latent mean and log-variance")
    print("      -> Reparameterization -> Decoder -> Reconstruction")
    print("Loss = reconstruction loss + KL divergence")

    try:
        import torch
        import torch.nn as nn

        class VAE(nn.Module):
            def __init__(self, input_dim=10, hidden_dim=6, latent_dim=2):
                super().__init__()
                self.encoder = nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.ReLU()
                )
                self.mean_layer = nn.Linear(hidden_dim, latent_dim)
                self.logvar_layer = nn.Linear(hidden_dim, latent_dim)
                self.decoder = nn.Sequential(
                    nn.Linear(latent_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, input_dim),
                    nn.Sigmoid()
                )

            def reparameterize(self, mean, logvar):
                std = torch.exp(0.5 * logvar)
                epsilon = torch.randn_like(std)
                return mean + epsilon * std

            def forward(self, x):
                h = self.encoder(x)
                mean = self.mean_layer(h)
                logvar = self.logvar_layer(h)
                z = self.reparameterize(mean, logvar)
                return self.decoder(z), mean, logvar

        model = VAE()
        sample = torch.rand(4, 10)
        reconstructed, latent_mean, latent_logvar = model(sample)
        print("\nPyTorch VAE demonstration:")
        print("Input shape:", tuple(sample.shape))
        print("Reconstruction shape:", tuple(reconstructed.shape))
        print("Latent mean shape:", tuple(latent_mean.shape))
    except ImportError:
        print("\nPyTorch not installed; architecture description shown only.")
        print("To run the VAE demonstration: pip install torch")


# ============================================================
# TASK 4 - CAUSAL SELF-ATTENTION
# ============================================================

def causal_attention(Q, K, V):
    """
    Scaled dot-product causal attention.
    Position i can attend only to positions j <= i.
    """
    dimension = Q.shape[1]
    scores = (Q @ K.T) / np.sqrt(dimension)

    # Mask future positions j > i.
    future_mask = np.triu(np.ones(scores.shape, dtype=bool), k=1)
    scores[future_mask] = -np.inf

    # Stable row-wise softmax.
    row_max = np.max(scores, axis=1, keepdims=True)
    exp_scores = np.exp(scores - row_max)
    weights = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

    return weights, weights @ V


def task4():
    print("\n" + "=" * 70)
    print("TASK 4: CAUSAL SELF-ATTENTION")
    print("=" * 70)

    Q = np.array([[1, 0], [0, 1], [1, 1]], dtype=float)
    K = np.array([[1, 0], [0, 1], [1, 1]], dtype=float)
    V = np.array([[10, 20], [30, 40], [50, 60]], dtype=float)

    weights, output = causal_attention(Q, K, V)

    print("Causal attention weights:\n", weights)
    print("\nAttention output:\n", output)

    # Regression/property test: all future attention weights must be zero.
    for i in range(weights.shape[0]):
        for j in range(weights.shape[1]):
            if j > i:
                assert np.isclose(weights[i, j], 0.0)

    print("\nRegression test passed: no position attends to a future token.")
    print("Correct allowed-position condition: j <= i.")
    print("The defective condition j != i wrongly blocks the current token")
    print("and does not correctly block every future token.")


# ============================================================
# TASK 5 - DAG TRAVERSAL, GIBBS SAMPLING, COVARIANCE
# ============================================================

DAG = {
    "A": [("B", 4), ("C", 7)],
    "B": [("C", 2), ("D", 6)],
    "C": [("D", 1), ("E", 8)],
    "D": [("E", 3)],
    "E": [],
}


def weighted_dag_walk(start="A", seed=42):
    """
    Illustrative weighted traversal of the supplied DAG.
    Edge weights are normalized among outgoing edges.

    This is not a full Bayesian-network ancestral sampler:
    the assignment supplies edge weights, but not node CPTs
    or conditional probability distributions.
    """
    rng = np.random.default_rng(seed)
    current = start
    path = [current]

    while DAG[current]:
        outgoing = DAG[current]
        children = [child for child, _ in outgoing]
        weights = np.array([weight for _, weight in outgoing], dtype=float)
        probabilities = weights / weights.sum()
        current = rng.choice(children, p=probabilities)
        path.append(current)

    return path


def gibbs_sampling(iterations=5000, burn_in=500, beta=0.8, seed=42):
    """
    Illustrative Gibbs sampler for a binary five-node Ising chain.
    The assignment does not specify a particular MRF topology
    or potential functions, so this is a clearly labelled example.
    """
    rng = np.random.default_rng(seed)
    number_of_nodes = 5
    state = rng.choice([-1, 1], size=number_of_nodes)
    samples = []

    for iteration in range(iterations):
        for i in range(number_of_nodes):
            neighbor_sum = 0
            if i > 0:
                neighbor_sum += state[i - 1]
            if i < number_of_nodes - 1:
                neighbor_sum += state[i + 1]

            p_positive = 1.0 / (1.0 + np.exp(-2.0 * beta * neighbor_sum))
            state[i] = 1 if rng.random() < p_positive else -1

        if iteration >= burn_in:
            samples.append(state.copy())

    return np.asarray(samples)


def covariance_matrix(samples):
    """Unbiased sample covariance between all column pairs."""
    samples = np.asarray(samples, dtype=float)
    centered = samples - np.mean(samples, axis=0)
    return centered.T @ centered / (samples.shape[0] - 1)


def task5():
    print("\n" + "=" * 70)
    print("TASK 5: PROBABILISTIC GRAPHICAL MODELS")
    print("=" * 70)

    print("Weighted DAG traversals (illustrative):")
    for seed in range(5):
        print(weighted_dag_walk(seed=seed))

    print("\nGibbs sampling for an illustrative binary MRF chain:")
    samples = gibbs_sampling()
    print("First 10 retained samples:\n", samples[:10])
    print("\nEstimated variable means:\n", np.mean(samples, axis=0))

    D_k = covariance_matrix(samples)
    print("\nAll-pairs covariance matrix D(k):\n", D_k)
    print("\nCovariance matrix shape:", D_k.shape)

    print("\nSource-data limitation:")
    print("The supplied DAG edge weights alone do not define Bayesian CPTs.")
    print("The MRF requires a specified graph and clique potentials.")
    print("The sampler above demonstrates the mechanics, not a unique")
    print("numerical answer implied by the incomplete model specification.")


# ============================================================
# TASK 6 - UNSUPERVISED ANOMALY DETECTION
# ============================================================

def task6():
    print("\n" + "=" * 70)
    print("TASK 6: UNSUPERVISED ANOMALY DETECTION")
    print("=" * 70)

    labels = ["J1", "J2", "J3", "J4", "J5", "J6", "J7", "J8"]
    X = np.array([
        [1, 3, 20],
        [2, 5, 50],
        [4, 6, 30],
        [6, 8, 40],
        [5, 9, 60],
        [8, 10, 30],
        [9, 11, 50],
        [11, 13, 40],
    ], dtype=float)

    print("Original dataset:")
    for label, row in zip(labels, X):
        print(f"{label}: {row}")

    # Standardize the three features.
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0, ddof=0)
    std[std == 0] = 1.0
    Z = (X - mean) / std

    # PCA reconstruction using two principal components.
    covariance = np.cov(Z, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    components = eigenvectors[:, order[:2]]

    reduced = Z @ components
    reconstructed = reduced @ components.T

    # Squared reconstruction error is used as an anomaly score.
    residual = Z - reconstructed
    scores = np.sum(residual ** 2, axis=1)

    # Demonstration threshold. With only eight samples this is
    # a small-sample heuristic, not a statistically validated cutoff.
    threshold = float(np.mean(scores) + 2.0 * np.std(scores))

    print("\nFeature means:", mean)
    print("Feature standard deviations:", std)
    print("\nStandardized data:\n", Z)
    print("\nRetained PCA components:\n", components)
    print("\nReconstruction errors:\n", residual)

    print("\nAnomaly scores and decisions:")
    print(f"{'Sample':<10}{'Score':<15}{'Threshold':<15}{'Decision'}")
    print("-" * 55)

    for label, score in zip(labels, scores):
        decision = "ANOMALY" if score > threshold else "NORMAL"
        print(f"{label:<10}{score:<15.6f}{threshold:<15.6f}{decision}")

    print("\nMethod: standardization + PCA + reconstruction error.")
    print("Learning type: unsupervised.")
    print("Score: sum of squared reconstruction residuals.")
    print("Threshold: mean score + 2 standard deviations.")
    print("This is a transparent demonstration score, not a negative")
    print("log-likelihood from a fitted probabilistic factor model.")


# ============================================================
# MAIN PROGRAM
# ============================================================

def main():
    print("=" * 70)
    print("SIMATS ENGINEERING - DEEP LEARNING - ASSIGNMENT 4 (CO4)")
    print("PYTHON PRACTICE AND VERIFICATION")
    print("=" * 70)

    task1()
    task2()
    task3()
    task4()
    task5()
    task6()

    print("\n" + "=" * 70)
    print("ALL SIX TASK DEMONSTRATIONS FINISHED")
    print("=" * 70)


if __name__ == "__main__":
    main()
