"""
SIMATS ENGINEERING - DEEP LEARNING
ASSIGNMENT 5 (CO5): GENERATIVE MODELLING, SAMPLING AND APPROXIMATION

PART B computational practice file:
 B1: Instrumented RBM training: CD-1 and Persistent CD (PCD)
 B1 validation: exact tiny model, under-capacity model, boundary case
 B2: Same-size easy and difficult synthetic binary datasets
 B3: Small exact-vs-approximation energy-model comparison and scaling counts
 B4: Optional PyTorch VAE, generated samples, metrics and challenge tests
 A2: Exact partition function vs Gibbs-sampling estimate

IMPORTANT:
- Part A of the assignment requires language-independent pseudocode.
  This Python program is for Part B practice/evidence, not a replacement.
- The brief requires faculty-approved open-source data. Synthetic datasets
  below are clearly labelled demonstrations. Replace them with the approved
  dataset before submitting experimental results.
- Numerical results depend on random seeds, hyperparameters and runtime.

Install:
    pip install numpy matplotlib
Optional for B4 VAE:
    pip install torch

Run:
    python Assignment_5_CO5_Generative_Models.py
"""

import time
import itertools
import numpy as np

SEED = 42
rng_global = np.random.default_rng(SEED)
np.set_printoptions(precision=4, suppress=True)


# ---------------------------------------------------------------------
# GENERAL UTILITIES
# ---------------------------------------------------------------------

def sigmoid(x):
    x = np.clip(x, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-x))


def bernoulli_sample(probabilities, rng):
    return (rng.random(probabilities.shape) < probabilities).astype(np.float32)


def binary_patterns(n_units):
    """Enumerate all binary vectors of length n_units."""
    return np.array(list(itertools.product([0.0, 1.0], repeat=n_units)),
                    dtype=np.float32)


def save_training_plots(history, prefix):
    """Save loss/error and weight-norm curves if matplotlib is available."""
    try:
        import matplotlib.pyplot as plt
        if "reconstruction_error" in history:
            plt.figure()
            plt.plot(history["reconstruction_error"])
            plt.xlabel("Epoch")
            plt.ylabel("Reconstruction error")
            plt.title(prefix + " reconstruction error")
            plt.tight_layout()
            plt.savefig(prefix + "_reconstruction.png", dpi=150)
            plt.close()

        if "weight_norm" in history:
            plt.figure()
            plt.plot(history["weight_norm"])
            plt.xlabel("Epoch")
            plt.ylabel("Weight norm")
            plt.title(prefix + " weight norm")
            plt.tight_layout()
            plt.savefig(prefix + "_weight_norm.png", dpi=150)
            plt.close()
    except ImportError:
        print("matplotlib unavailable; plots skipped. Install with pip install matplotlib")


# ---------------------------------------------------------------------
# RBM IMPLEMENTATION
# ---------------------------------------------------------------------

class RBM:
    """
    Binary Restricted Boltzmann Machine.

    Energy:
      E(v,h) = -a^T v - b^T h - v^T W h

    Conditional probabilities:
      P(h_j=1 | v) = sigmoid(b_j + sum_i v_i W_ij)
      P(v_i=1 | h) = sigmoid(a_i + sum_j W_ij h_j)

    Supports CD-k and persistent CD (PCD).
    """

    def __init__(self, n_visible, n_hidden, seed=SEED):
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.rng = np.random.default_rng(seed)
        self.W = self.rng.normal(0, 0.01, (n_visible, n_hidden)).astype(np.float32)
        self.vbias = np.zeros(n_visible, dtype=np.float32)
        self.hbias = np.zeros(n_hidden, dtype=np.float32)
        self.persistent_v = None

    def prop_hidden(self, v):
        return sigmoid(v @ self.W + self.hbias)

    def prop_visible(self, h):
        return sigmoid(h @ self.W.T + self.vbias)

    def sample_hidden(self, v):
        p = self.prop_hidden(v)
        return p, bernoulli_sample(p, self.rng)

    def sample_visible(self, h):
        p = self.prop_visible(h)
        return p, bernoulli_sample(p, self.rng)

    def free_energy(self, v):
        vbias_term = v @ self.vbias
        hidden_term = np.sum(np.logaddexp(0, v @ self.W + self.hbias), axis=1)
        return -(vbias_term + hidden_term)

    def energy(self, v, h):
        return -(v @ self.vbias + h @ self.hbias +
                 np.sum((v @ self.W) * h, axis=1))

    def gibbs_from_visible(self, v_start, k=1):
        """Run k full alternating Gibbs transitions."""
        vk = np.asarray(v_start, dtype=np.float32).copy()
        for _ in range(k):
            _, hk = self.sample_hidden(vk)
            _, vk = self.sample_visible(hk)
        phk = self.prop_hidden(vk)
        return vk, phk

    def train(self, data, epochs=30, batch_size=16, lr=0.05,
              method="cd", k=1, record=True):
        data = np.asarray(data, dtype=np.float32)
        if data.ndim != 2 or data.shape[1] != self.n_visible:
            raise ValueError("Training data must have shape (N, n_visible).")
        if not np.all((data == 0) | (data == 1)):
            raise ValueError("RBM input must be binary (0/1).")
        if method not in ("cd", "pcd"):
            raise ValueError("method must be 'cd' or 'pcd'.")

        if method == "pcd" and self.persistent_v is None:
            self.persistent_v = bernoulli_sample(
                np.full((batch_size, self.n_visible), 0.5), self.rng
            )

        history = {
            "reconstruction_error": [],
            "pseudo_log_likelihood": [],
            "hidden_activation": [],
            "gibbs_steps": [],
            "weight_norm": [],
            "epoch_seconds": [],
            "chain_flip_fraction": [],
        }

        for epoch in range(epochs):
            start = time.perf_counter()
            order = self.rng.permutation(len(data))
            epoch_errors, epoch_hidden, flip_values = [], [], []

            for start_idx in range(0, len(data), batch_size):
                batch = data[order[start_idx:start_idx + batch_size]]
                if len(batch) == 0:
                    continue

                ph0 = self.prop_hidden(batch)
                h0 = bernoulli_sample(ph0, self.rng)

                if method == "cd":
                    chain_start = batch
                else:
                    if self.persistent_v.shape[0] != len(batch):
                        self.persistent_v = bernoulli_sample(
                            np.full((len(batch), self.n_visible), 0.5), self.rng
                        )
                    chain_start = self.persistent_v.copy()

                vk, phk = self.gibbs_from_visible(chain_start, k=k)
                if method == "pcd":
                    flip_values.append(float(np.mean(vk != chain_start)))
                    self.persistent_v = vk.copy()

                # Stochastic-gradient approximation to log-likelihood gradient.
                positive = batch.T @ ph0
                negative = vk.T @ phk
                scale = 1.0 / len(batch)

                self.W += lr * (positive - negative) * scale
                self.vbias += lr * np.mean(batch - vk, axis=0)
                self.hbias += lr * np.mean(ph0 - phk, axis=0)

                reconstruction = self.prop_visible(ph0)
                epoch_errors.append(float(np.mean((batch - reconstruction) ** 2)))
                epoch_hidden.append(float(np.mean(ph0)))

            history["reconstruction_error"].append(float(np.mean(epoch_errors)))
            history["hidden_activation"].append(float(np.mean(epoch_hidden)))
            history["pseudo_log_likelihood"].append(self.pseudo_log_likelihood(data))
            history["gibbs_steps"].append(int(k))
            history["weight_norm"].append(float(np.linalg.norm(self.W)))
            history["epoch_seconds"].append(time.perf_counter() - start)
            history["chain_flip_fraction"].append(
                float(np.mean(flip_values)) if flip_values else float("nan")
            )

            if record:
                print(
                    f"{method.upper():3s} epoch {epoch+1:03d}/{epochs} | "
                    f"recon={history['reconstruction_error'][-1]:.5f} | "
                    f"PLL={history['pseudo_log_likelihood'][-1]:.5f} | "
                    f"hidden={history['hidden_activation'][-1]:.3f} | "
                    f"||W||={history['weight_norm'][-1]:.3f} | "
                    f"time={history['epoch_seconds'][-1]:.3f}s"
                )
        return history

    def pseudo_log_likelihood(self, data):
        """
        Approximate pseudo-log-likelihood:
        mean_i [D * log sigmoid(FE(v_flip_i)-FE(v))].
        One randomly selected bit is flipped per row.
        """
        data = np.asarray(data, dtype=np.float32)
        if len(data) == 0:
            return float("nan")
        indices = self.rng.integers(0, self.n_visible, size=len(data))
        flipped = data.copy()
        flipped[np.arange(len(data)), indices] = 1.0 - flipped[
            np.arange(len(data)), indices
        ]
        delta = self.free_energy(flipped) - self.free_energy(data)
        pll = self.n_visible * (-np.logaddexp(0, -delta))
        return float(np.mean(pll))

    def generate(self, n_samples=16, burn_in=100, thinning=10):
        """Generate visible samples using alternating Gibbs transitions."""
        v = bernoulli_sample(
            np.full((n_samples, self.n_visible), 0.5), self.rng
        )
        for _ in range(burn_in):
            v, _ = self.gibbs_from_visible(v, k=1)
        samples = []
        for _ in range(n_samples):
            v, _ = self.gibbs_from_visible(v, k=thinning)
            samples.append(v.copy())
        return np.vstack(samples)


# ---------------------------------------------------------------------
# A2: EXACT PARTITION FUNCTION VS GIBBS APPROXIMATION
# ---------------------------------------------------------------------

def task_a2_partition_function():
    print("\n" + "=" * 72)
    print("A2 DEMONSTRATION: EXACT Z VS GIBBS-SAMPLE ESTIMATE")
    print("=" * 72)

    biases = np.array([0.4, 0.7, 0.9, 1.2, 1.5], dtype=float)
    target = np.array([1, 0, 1, 0, 1], dtype=float)
    number_of_samples = 20

    states = binary_patterns(5)
    unnormalized = np.exp(states @ biases)
    Z = float(np.sum(unnormalized))
    exact_probability = float(np.exp(target @ biases) / Z)

    # Independent Gibbs chains for a model with no hidden units:
    # each visible bit is Bernoulli(sigmoid(bias_i)).
    rng = np.random.default_rng(SEED)
    p = sigmoid(biases)
    draws = (rng.random((number_of_samples, 5)) < p).astype(int)
    estimate = float(np.mean(np.all(draws == target, axis=1)))

    print("Biases:", biases)
    print("Target vector:", target.astype(int))
    print("Exact partition function Z:", Z)
    print("Exact P(target):", exact_probability)
    print(f"Empirical frequency from {number_of_samples} independent Gibbs samples:", estimate)
    print("Absolute estimation error:", abs(estimate - exact_probability))
    print("More samples generally reduce Monte Carlo variance.")
    print("More units increase the number of configurations exponentially (2^n).")
    print("This particular no-hidden-unit model has independent visible units,")
    print("so direct independent sampling is possible; a general BM/RBM can mix")
    print("poorly, especially when its probability mass is split across modes.")


# ---------------------------------------------------------------------
# DATASET DEMONSTRATIONS
# ---------------------------------------------------------------------

def make_easy_dataset(n=400, seed=1):
    """Single simple binary pattern with light bit noise."""
    rng = np.random.default_rng(seed)
    prototype = np.array([1, 1, 1, 1, 0, 0, 0, 0], dtype=np.float32)
    X = np.tile(prototype, (n, 1))
    flips = rng.random(X.shape) < 0.05
    return np.logical_xor(X.astype(bool), flips).astype(np.float32)


def make_hard_dataset(n=400, seed=2):
    """Mixture of well-separated prototypes; more modes to learn."""
    rng = np.random.default_rng(seed)
    prototypes = np.array([
        [1, 1, 1, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 1, 1, 1],
        [1, 0, 1, 0, 1, 0, 1, 0],
        [0, 1, 0, 1, 0, 1, 0, 1],
    ], dtype=np.float32)
    chosen = rng.integers(0, len(prototypes), size=n)
    X = prototypes[chosen].copy()
    flips = rng.random(X.shape) < 0.05
    return np.logical_xor(X.astype(bool), flips).astype(np.float32)


def split_data(X, fraction=0.8, seed=123):
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(X))
    cut = int(len(X) * fraction)
    return X[order[:cut]], X[order[cut:]]


# ---------------------------------------------------------------------
# B1 AND B2: TRAIN AND INSTRUMENT RBMS
# ---------------------------------------------------------------------

def run_rbm_experiment(name, X, hidden_units=4, epochs=25):
    print("\n" + "-" * 72)
    print(name)
    print("-" * 72)
    train, test = split_data(X)

    print("Train shape:", train.shape, "| Test shape:", test.shape)
    print("Visible units:", X.shape[1], "| Hidden units:", hidden_units)

    cd_model = RBM(X.shape[1], hidden_units, seed=SEED)
    print("\nTraining CD-1:")
    cd_history = cd_model.train(
        train, epochs=epochs, batch_size=32, lr=0.05,
        method="cd", k=1
    )

    pcd_model = RBM(X.shape[1], hidden_units, seed=SEED)
    print("\nTraining PCD-5:")
    pcd_history = pcd_model.train(
        train, epochs=epochs, batch_size=32, lr=0.05,
        method="pcd", k=5
    )

    print("\nComparison (last epoch):")
    print("CD-1 test pseudo-LL:", cd_model.pseudo_log_likelihood(test))
    print("PCD-5 test pseudo-LL:", pcd_model.pseudo_log_likelihood(test))
    print("CD-1 reconstruction:", cd_history["reconstruction_error"][-1])
    print("PCD-5 reconstruction:", pcd_history["reconstruction_error"][-1])
    print("CD-1 time/epoch:", np.mean(cd_history["epoch_seconds"]))
    print("PCD-5 time/epoch:", np.mean(pcd_history["epoch_seconds"]))
    print("PCD-5 mean chain flip fraction:",
          np.nanmean(pcd_history["chain_flip_fraction"]))

    print("\nGenerated samples (PCD model):")
    print(pcd_model.generate(n_samples=8, burn_in=100, thinning=10).astype(int))

    safe_name = name.lower().replace(" ", "_").replace("/", "_")
    save_training_plots(cd_history, safe_name + "_cd1")
    save_training_plots(pcd_history, safe_name + "_pcd5")
    return cd_model, pcd_model, cd_history, pcd_history


def task_b1_b2():
    print("\n" + "=" * 72)
    print("B1/B2: RBM IMPLEMENTATION, INSTRUMENTATION AND ADVERSARIAL DATA")
    print("=" * 72)

    easy = make_easy_dataset()
    hard = make_hard_dataset()

    print("B2 prediction: the multi-mode dataset is expected to be harder")
    print("because the negative-phase chain may remain in one mode and fail")
    print("to visit other separated modes within a limited number of Gibbs steps.")

    run_rbm_experiment("Easy single-pattern dataset", easy, hidden_units=4)
    run_rbm_experiment("Hard multi-mode dataset", hard, hidden_units=4)

    # Boundary case: one visible unit, constant training data.
    print("\nBoundary test: one visible unit and constant observations.")
    boundary = np.zeros((40, 1), dtype=np.float32)
    model = RBM(n_visible=1, n_hidden=1, seed=7)
    history = model.train(boundary, epochs=5, batch_size=10,
                          lr=0.03, method="cd", k=1)
    print("Boundary test completed. Final reconstruction error:",
          history["reconstruction_error"][-1])

    # Exact tiny-model validation: enumerate all v,h states for a 2x1 RBM.
    print("\nSmall-model exact validation (2 visible, 1 hidden):")
    tiny = RBM(2, 1, seed=9)
    tiny.W[:] = np.array([[0.2], [-0.1]])
    tiny.vbias[:] = np.array([0.1, -0.2])
    tiny.hbias[:] = np.array([0.05])
    v_states = binary_patterns(2)
    h_states = binary_patterns(1)
    all_energies = []
    for v in v_states:
        for h in h_states:
            all_energies.append(tiny.energy(v[None, :], h[None, :])[0])
    exact_Z = float(np.sum(np.exp(-np.array(all_energies))))
    print("Enumerated configurations:", len(all_energies))
    print("Exact Z:", exact_Z)
    print("This test checks energy and enumeration on a model small enough")
    print("to verify by hand. The under-capacity test is the 4-hidden-unit")
    print("model on the multi-mode data; it may fail to capture all modes.")


# ---------------------------------------------------------------------
# B3: SMALL EXACT MODEL AND GREEDY LAYER-WISE APPROXIMATION DEMO
# ---------------------------------------------------------------------

def exact_binary_ebm_fit(data, max_units=8, steps=100, lr=0.05):
    """
    Tiny fully-visible binary pairwise energy model.
    Exact likelihood and partition function are computed by enumeration.
    Used as a tractability demonstration, not a full DBN implementation.
    """
    data = np.asarray(data, dtype=float)
    n, d = data.shape
    if d > max_units:
        raise ValueError("Exact enumeration restricted to a small number of units.")

    states = binary_patterns(d).astype(float)
    bias = np.zeros(d)
    pair = np.zeros((d, d))

    for _ in range(steps):
        logits = states @ bias + 0.5 * np.sum((states @ pair) * states, axis=1)
        logits -= np.max(logits)
        probs = np.exp(logits)
        probs /= probs.sum()

        model_mean = probs @ states
        data_mean = np.mean(data, axis=0)
        bias += lr * (data_mean - model_mean)

        model_second = (states.T * probs) @ states
        data_second = (data.T @ data) / n
        pair += lr * (data_second - model_second)
        pair = 0.5 * (pair + pair.T)
        np.fill_diagonal(pair, 0.0)

    logits = states @ bias + 0.5 * np.sum((states @ pair) * states, axis=1)
    max_logit = np.max(logits)
    logZ = max_logit + np.log(np.sum(np.exp(logits - max_logit)))
    data_logits = data @ bias + 0.5 * np.sum((data @ pair) * data, axis=1)
    avg_log_likelihood = float(np.mean(data_logits - logZ))
    return avg_log_likelihood, 2 ** d


def greedy_layerwise_demo(data, hidden_sizes=(3, 2), epochs=15):
    """
    Greedy RBM stack demonstration:
    train an RBM on data, transform to hidden probabilities,
    then train the next RBM on those activations binarised.
    This is a pretraining procedure, not joint exact DBN maximum likelihood.
    """
    representation = np.asarray(data, dtype=np.float32)
    total_gibbs_steps = 0
    layer_models = []

    for layer_index, hidden_size in enumerate(hidden_sizes):
        visible_size = representation.shape[1]
        rbm = RBM(visible_size, hidden_size, seed=SEED + layer_index)
        rbm.train(representation, epochs=epochs, batch_size=16,
                  lr=0.04, method="cd", k=1, record=False)
        probabilities = rbm.prop_hidden(representation)
        representation = (probabilities >= 0.5).astype(np.float32)
        total_gibbs_steps += epochs * int(np.ceil(len(data) / 16))
        layer_models.append(rbm)

    # A simple diagnostic score of the learned representation.
    # Not a normalized DBN log-likelihood; label it explicitly.
    activation_rate = float(np.mean(representation))
    return activation_rate, total_gibbs_steps, layer_models


def task_b3():
    print("\n" + "=" * 72)
    print("B3: EXACT ENUMERATION VS GREEDY LAYER-WISE PRETRAINING")
    print("=" * 72)

    datasets = [
        make_easy_dataset(n=80, seed=11)[:, :4],
        make_hard_dataset(n=80, seed=12)[:, :4],
        make_easy_dataset(n=80, seed=13)[:, :4] ^ 0,  # third reproducible case
    ]

    print("Three small binary datasets (4 visible units each):")
    for index, data in enumerate(datasets, 1):
        exact_ll, configurations = exact_binary_ebm_fit(data, steps=80)
        greedy_score, gibbs_steps, _ = greedy_layerwise_demo(
            data, hidden_sizes=(3, 2), epochs=10
        )
        print(f"\nDataset {index}:")
        print("  Exact tiny EBM average log-likelihood:", exact_ll)
        print("  Exact configurations enumerated:", configurations)
        print("  Greedy final activation-rate diagnostic:", greedy_score)
        print("  Greedy Gibbs steps (count):", gibbs_steps)

    # Deliberate counterexample illustration: greedy pretraining optimizes
    # local layer objectives, not the joint global likelihood.
    print("\nCounterexample explanation:")
    print("Greedy layer-wise pretraining can lock the first layer into a")
    print("representation that is locally useful but discards information")
    print("needed by the joint multi-layer optimum. Therefore local layer")
    print("training does not guarantee the exact joint maximum-likelihood solution.")

    print("\nScaling operation counts for exact binary enumeration:")
    for units in [4, 8, 12, 16, 20]:
        print(f"  {units:2d} units -> {2**units:>10,d} configurations")

    print("\nImportant measurement note:")
    print("The exact model above is a small fully-visible EBM used to show")
    print("enumeration and exact normalization. The greedy stack is a DBN-style")
    print("pretraining demonstration. Its activation diagnostic is NOT a DBN")
    print("log-likelihood, so do not place it in a log-likelihood comparison table.")
    print("A fully comparable DBN likelihood requires a specified DBN joint")
    print("distribution and a consistent likelihood estimator.")


# ---------------------------------------------------------------------
# B4: OPTIONAL DIRECTED GENERATIVE NETWORK (VAE)
# ---------------------------------------------------------------------

def task_b4():
    print("\n" + "=" * 72)
    print("B4: DIRECTED GENERATIVE MODEL - VAE")
    print("=" * 72)

    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError:
        print("PyTorch is not installed. Install using: pip install torch")
        print("B4 architecture: encoder -> mean/log-variance -> latent z -> decoder.")
        return

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # Synthetic binary tabular demonstration; replace with faculty-approved data.
    X_np = make_hard_dataset(n=1200, seed=21)
    X_tensor = torch.tensor(X_np, dtype=torch.float32)
    loader = DataLoader(TensorDataset(X_tensor), batch_size=64, shuffle=True)

    class TabularVAE(nn.Module):
        def __init__(self, input_dim, hidden_dim=32, latent_dim=4):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU()
            )
            self.fc_mu = nn.Linear(hidden_dim, latent_dim)
            self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
            self.decoder = nn.Sequential(
                nn.Linear(latent_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, input_dim),
                nn.Sigmoid()
            )

        def reparameterize(self, mu, logvar):
            std = torch.exp(0.5 * logvar)
            return mu + torch.randn_like(std) * std

        def forward(self, x):
            h = self.encoder(x)
            mu, logvar = self.fc_mu(h), self.fc_logvar(h)
            z = self.reparameterize(mu, logvar)
            return self.decoder(z), mu, logvar

    model = TabularVAE(input_dim=X_np.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    epochs = 30
    losses = []

    for epoch in range(epochs):
        model.train()
        total = 0.0
        for (batch,) in loader:
            optimizer.zero_grad()
            recon, mu, logvar = model(batch)
            reconstruction = F.binary_cross_entropy(
                recon, batch, reduction="sum"
            )
            kl = -0.5 * torch.sum(
                1 + logvar - mu.pow(2) - logvar.exp()
            )
            loss = (reconstruction + kl) / len(batch)
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * len(batch)

        epoch_loss = total / len(X_tensor)
        losses.append(epoch_loss)
        print(f"VAE epoch {epoch+1:02d}/{epochs} | negative-ELBO loss={epoch_loss:.5f}")

    model.eval()
    with torch.no_grad():
        generated = model.decoder(torch.randn(1000, 4)).numpy()
        generated_binary = (generated >= 0.5).astype(np.float32)

    # Simple distributional checks for binary tabular features.
    real_marginals = X_np.mean(axis=0)
    generated_marginals = generated_binary.mean(axis=0)
    marginal_abs_error = float(np.mean(np.abs(real_marginals - generated_marginals)))

    real_unique = len(np.unique(X_np, axis=0))
    generated_unique = len(np.unique(generated_binary, axis=0))
    print("\nReal feature-wise one-rates:", real_marginals)
    print("Generated feature-wise one-rates:", generated_marginals)
    print("Mean absolute marginal difference:", marginal_abs_error)
    print("Unique real patterns:", real_unique)
    print("Unique generated patterns:", generated_unique)
    print("\nGenerated sample examples:")
    print(generated_binary[:12].astype(int))

    # Challenge test: compare diversity and nearest-neighbour overlap.
    # Exact record matches are a simple memorisation warning, not a privacy proof.
    real_rows = {tuple(row.astype(int)) for row in X_np}
    generated_rows = [tuple(row.astype(int)) for row in generated_binary]
    exact_match_rate = np.mean([row in real_rows for row in generated_rows])
    print("Generated samples exactly matching a real training pattern:",
          float(exact_match_rate))
    print("Challenge interpretation: high match may reflect common discrete")
    print("patterns or memorisation; inspect nearest neighbours and rare modes.")

    try:
        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot(losses)
        plt.xlabel("Epoch")
        plt.ylabel("Negative ELBO loss")
        plt.title("VAE training objective")
        plt.tight_layout()
        plt.savefig("co5_vae_training_curve.png", dpi=150)
        plt.close()
        print("Saved VAE curve: co5_vae_training_curve.png")
    except ImportError:
        print("Install matplotlib to save VAE training curve.")


# ---------------------------------------------------------------------
# FINAL ENGINEERING DECISION NOTE
# ---------------------------------------------------------------------

def engineering_decision_note():
    print("\n" + "=" * 72)
    print("B4 ENGINEERING DECISION NOTE")
    print("=" * 72)
    print("""
1. RBM/DBM production limitation:
   Undirected models commonly require iterative MCMC transitions.
   Sampling millions of records can therefore require many sequential
   Gibbs updates and careful monitoring of mixing.

2. Directed VAE advantage:
   Once trained, a VAE can draw a latent vector from its prior and pass it
   through the decoder in one forward pass. This is convenient for high-
   throughput generation.

3. Guarantees and limitations:
   A VAE optimizes a variational lower bound (ELBO), not generally the exact
   data log-likelihood. It does not guarantee privacy, perfect mode coverage,
   or absence of memorisation.

4. When tractable likelihood matters:
   Prefer a model with tractable likelihood when calibrated density estimates,
   anomaly scoring, or principled likelihood comparison is central.

5. Deployment monitoring:
   Monitor held-out likelihood/ELBO, feature and category distributions,
   sample-quality metrics, rare-category coverage, nearest-neighbour overlap,
   duplicate rates, latency and throughput. Synthetic data is not automatically
   private; privacy evaluation is a separate requirement.
""")


def main():
    print("=" * 72)
    print("SIMATS ENGINEERING - DEEP LEARNING - ASSIGNMENT 5 (CO5)")
    print("GENERATIVE MODELLING, SAMPLING AND APPROXIMATION ENGINEERING")
    print("=" * 72)

    task_a2_partition_function()
    task_b1_b2()
    task_b3()
    task_b4()
    engineering_decision_note()

    print("\nAll available demonstrations have finished.")
    print("Replace synthetic data with faculty-approved open-source data")
    print("before treating results as final assignment evidence.")


if __name__ == "__main__":
    main()
