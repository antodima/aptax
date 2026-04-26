import jax
import jax.numpy as jnp


def sample_from_scm(key, n_samples, n_features):
    k1, k2, k3, k4 = jax.random.split(key, 4)
    # 1. Sample a random DAG structure
    # (Conceptual: nodes are ordered to ensure acyclicity)
    num_nodes = n_features + int(jax.random.randint(k1, (), 1, 10))
    adj_matrix = jnp.tril(jax.random.uniform(k2, (num_nodes, num_nodes)) > 0.8, -1)

    # 2. Sample noise and propagate through functions
    z = jnp.zeros((n_samples, num_nodes))
    for i in range(num_nodes):
        k_iter = jax.random.fold_in(k3, i)
        k_noise, k_weights = jax.random.split(k_iter)

        mask = adj_matrix[i]
        parents = z[:, mask]
        noise = jax.random.normal(k_noise, (n_samples,))

        # Apply a random non-linear function f_i
        if parents.shape[1] > 0:
            # Simple weighted sum with a non-linearity (e.g., Tanh or ReLU)
            weights = jax.random.normal(k_weights, (parents.shape[1],))
            z = z.at[:, i].set(jnp.tanh(parents @ weights) + noise)
        else:
            z = z.at[:, i].set(noise)

    # 3. Select which nodes are features (X) and which is the target (y)
    indices = jax.random.permutation(k4, num_nodes)
    x_indices = indices[:n_features]
    y_index = indices[n_features]

    return z[:, x_indices], z[:, y_index]


def sample_from_bnn(key, n_samples, n_features):
    k1, k2, k3, k4 = jax.random.split(key, 4)
    # 1. Sample random inputs
    X = jax.random.normal(k1, (n_samples, n_features))

    # 2. Define a simple random architecture (e.g., 2-layer MLP)
    hidden_dim = int(jax.random.randint(k2, (), 16, 128))
    W1 = jax.random.normal(k3, (n_features, hidden_dim))
    W2 = jax.random.normal(k4, (hidden_dim, 1))

    # 3. Forward pass to get continuous targets
    # (Conceptual: TabPFN uses more complex variations and noise)
    hidden = jnp.maximum(0, X @ W1)  # ReLU
    y_continuous = (hidden @ W2).flatten()

    return X, y_continuous
