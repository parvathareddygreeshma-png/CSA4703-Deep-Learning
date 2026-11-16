"""
Deep Learning Assignment 1 – HIGGS experiments
Install: pip install numpy pandas scikit-learn matplotlib scipy
Run: python higgs_assignment_experiments.py

The script downloads the official UCI compressed CSV if it is not present.
The compressed source is large (about 2.6 GB). Ensure sufficient disk space.
It uses the first 1,000,000 rows for the required size ladder and fixed seed 42.
"""
import os, time, gzip, urllib.request
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import SGDClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, log_loss)
from scipy.special import expit

SEED = 42
URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00280/HIGGS.csv.gz"
FILE = "HIGGS.csv.gz"
MAX_ROWS = 1_000_000
SIZES = [1_000, 10_000, 100_000, 1_000_000]
BATCH = 512
EPOCHS = 15

def obtain_data():
    if not os.path.exists(FILE):
        print("Downloading the official UCI HIGGS file (~2.6 GB compressed)...")
        urllib.request.urlretrieve(URL, FILE)
    chunks, count = [], 0
    for chunk in pd.read_csv(FILE, header=None, compression="gzip",
                             chunksize=100_000):
        need = MAX_ROWS - count
        if need <= 0: break
        chunk = chunk.iloc[:need]
        chunks.append(chunk)
        count += len(chunk)
        print(f"Loaded {count:,} rows")
        if count >= MAX_ROWS: break
    data = pd.concat(chunks, ignore_index=True)
    y = data.iloc[:, 0].astype("int32").to_numpy()
    X = data.iloc[:, 1:].astype("float32").to_numpy()
    return X, y

def metrics(model, X, y, name, n, train_time, val_loss):
    t0 = time.perf_counter()
    pred = model.predict(X)
    elapsed = time.perf_counter() - t0
    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(X)[:, 1]
    elif hasattr(model, "decision_function"):
        prob = expit(model.decision_function(X))
    else:
        prob = pred.astype(float)
    return {
        "model": name, "n": n, "train_seconds": train_time,
        "validation_loss": val_loss,
        "test_accuracy": accuracy_score(y, pred),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        "test_log_loss": log_loss(y, np.clip(prob, 1e-7, 1-1e-7)),
        "prediction_seconds": elapsed,
        "prediction_ms_per_record": 1000*elapsed/len(y)
    }

def check_logistic_gradient():
    rng = np.random.default_rng(SEED)
    X = rng.normal(size=(12, 5))
    y = rng.integers(0, 2, size=12).astype(float)
    w = rng.normal(size=5); b = 0.13; lam = 0.01
    z = X @ w + b
    p = expit(z)
    gw = X.T @ (p-y)/len(y) + lam*w
    gb = np.mean(p-y)
    theta = np.r_[w, b]
    def loss(th):
        zz = X @ th[:-1] + th[-1]
        pp = np.clip(expit(zz), 1e-12, 1-1e-12)
        return -np.mean(y*np.log(pp)+(1-y)*np.log(1-pp)) + lam/2*np.sum(th[:-1]**2)
    eps = 1e-6
    gn = np.array([(loss(theta+eps*np.eye(6)[i])-loss(theta-eps*np.eye(6)[i]))/(2*eps)
                   for i in range(6)])
    ga = np.r_[gw, gb]
    print("Gradient check max absolute difference:", np.max(np.abs(ga-gn)))

def main():
    check_logistic_gradient()
    X_all, y_all = obtain_data()
    rows = []
    for n in SIZES:
        X, y = X_all[:n], y_all[:n]
        Xtr, Xtemp, ytr, ytemp = train_test_split(
            X, y, test_size=0.30, random_state=SEED, stratify=y)
        Xval, Xte, yval, yte = train_test_split(
            Xtemp, ytemp, test_size=0.50, random_state=SEED, stratify=ytemp)

        # Gaussian Naive Bayes: unscaled original numeric features
        model = GaussianNB()
        t0 = time.perf_counter(); model.fit(Xtr, ytr)
        tr_time = time.perf_counter()-t0
        val_prob = np.clip(model.predict_proba(Xval)[:,1], 1e-7, 1-1e-7)
        rows.append(metrics(model, Xte, yte, "GaussianNB", n, tr_time,
                            log_loss(yval, val_prob)))

        # Fit scaler only on training data; apply to val and test
        scaler = StandardScaler()
        t0 = time.perf_counter()
        Xtr_s = scaler.fit_transform(Xtr)
        Xval_s = scaler.transform(Xval)
        Xte_s = scaler.transform(Xte)
        preprocessing_time = time.perf_counter()-t0

        # Logistic regression via SGD with log loss
        lr = SGDClassifier(loss="log_loss", penalty="l2", alpha=1e-4,
                           learning_rate="optimal", random_state=SEED,
                           max_iter=1000, tol=1e-3)
        t0 = time.perf_counter(); lr.fit(Xtr_s, ytr)
        tr_time = time.perf_counter()-t0
        val_prob = np.clip(expit(lr.decision_function(Xval_s)), 1e-7, 1-1e-7)
        rows.append(metrics(lr, Xte_s, yte, "LogisticRegression", n, tr_time,
                            log_loss(yval, val_prob)))

        # MLP: partial_fit gives explicit mini-batch training
        mlp = MLPClassifier(hidden_layer_sizes=(128,64,32), activation="relu",
                            solver="adam", alpha=1e-4, batch_size=BATCH,
                            learning_rate_init=1e-3, max_iter=1,
                            random_state=SEED, shuffle=False)
        t0 = time.perf_counter()
        rng = np.random.default_rng(SEED)
        for epoch in range(EPOCHS):
            order = rng.permutation(len(Xtr_s))
            for start in range(0, len(order), BATCH):
                ids = order[start:start+BATCH]
                if epoch == 0 and start == 0:
                    mlp.partial_fit(Xtr_s[ids], ytr[ids], classes=np.array([0,1]))
                else:
                    mlp.partial_fit(Xtr_s[ids], ytr[ids])
            # validation loss is evaluated after each epoch by the final epoch below
        tr_time = time.perf_counter()-t0
        val_prob = np.clip(mlp.predict_proba(Xval_s)[:,1], 1e-7, 1-1e-7)
        rows.append(metrics(mlp, Xte_s, yte, "MLP", n, tr_time,
                            log_loss(yval, val_prob)))
        print(f"Completed n={n:,}; scaler time={preprocessing_time:.2f}s")

    result = pd.DataFrame(rows)
    result.to_csv("results.csv", index=False)
    print(result.to_string(index=False))

    for metric, ylabel, filename in [
        ("test_accuracy", "Test accuracy", "accuracy_vs_size.png"),
        ("train_seconds", "Training time (seconds)", "training_time_vs_size.png"),
        ("prediction_ms_per_record", "Prediction ms per record", "latency_vs_size.png")
    ]:
        plt.figure()
        for name, group in result.groupby("model"):
            plt.plot(group["n"], group[metric], marker="o", label=name)
        plt.xscale("log"); plt.xlabel("Number of records"); plt.ylabel(ylabel)
        plt.title(ylabel + " vs dataset size"); plt.legend(); plt.grid(True)
        plt.tight_layout(); plt.savefig(filename, dpi=160); plt.close()
    print("Saved results.csv and comparison plots.")

if __name__ == "__main__":
    main()
