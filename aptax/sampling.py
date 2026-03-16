import numpy as np


def sample_from_scm(n_samples, n_features):
    # 1. Sample a random DAG structure
    # (Conceptual: nodes are ordered to ensure acyclicity)
    num_nodes = n_features + np.random.randint(1, 10)
    adj_matrix = np.tril(np.random.rand(num_nodes, num_nodes) > 0.8, -1)

    # 2. Sample noise and propagate through functions
    z = np.zeros((n_samples, num_nodes))
    for i in range(num_nodes):
        parents = z[:, adj_matrix[i]]
        noise = np.random.normal(0, 1, size=(n_samples,))

        # Apply a random non-linear function f_i
        if parents.shape[1] > 0:
            # Simple weighted sum with a non-linearity (e.g., Tanh or ReLU)
            weights = np.random.randn(parents.shape[1])
            z[:, i] = np.tanh(parents @ weights) + noise
        else:
            z[:, i] = noise

    # 3. Select which nodes are features (X) and which is the target (y)
    indices = np.random.permutation(num_nodes)
    x_indices = indices[:n_features]
    y_index = indices[n_features]

    return z[:, x_indices], z[:, y_index]


def sample_from_bnn(n_samples, n_features):
    # 1. Sample random inputs
    X = np.random.normal(0, 1, size=(n_samples, n_features))

    # 2. Define a simple random architecture (e.g., 2-layer MLP)
    hidden_dim = np.random.randint(16, 128)
    W1 = np.random.randn(n_features, hidden_dim)
    W2 = np.random.randn(hidden_dim, 1)

    # 3. Forward pass to get continuous targets
    # (Conceptual: TabPFN uses more complex variations and noise)
    hidden = np.maximum(0, X @ W1)  # ReLU
    y_continuous = (hidden @ W2).flatten()

    return X, y_continuous
