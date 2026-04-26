import flax.nnx as nnx
import jax
import jax.numpy as jnp


def causal_attention_mask(seq_len):
    """Create a causal attention mask. The mask is a lower triangular matrix
    of shape (seq_len, seq_len) where the entries above the diagonal are -inf
    and the entries on and below the diagonal are 0.
    """
    return jnp.tril(jnp.ones((seq_len, seq_len)))


def scaled_dot_product(q, k, v, mask=None, dropout_fn=None):
    d_k = q.shape[-1]
    # Calculate scores: (batch, heads, seq, seq)
    # Using swapaxes is rank-agnostic for the leading dims
    attn_logits = jnp.matmul(q, k.swapaxes(-2, -1)) / jnp.sqrt(d_k)
    if mask is not None:
        # Assuming mask is boolean: True to keep, False to mask
        attn_logits = jnp.where(mask, attn_logits, -1e9)

    attention = jax.nn.softmax(attn_logits, axis=-1)
    # Optional dropout on attention weights
    if dropout_fn is not None:
        attention = dropout_fn(attention)

    values = jnp.matmul(attention, v)
    return values, attention


class MultiHeadAttention(nnx.Module):
    def __init__(self, input_dim, embed_dim, num_heads, *, rngs):
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")

        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.qkv_layer = nnx.Linear(input_dim, 3 * embed_dim, rngs=rngs)
        self.linear_layer = nnx.Linear(embed_dim, embed_dim, rngs=rngs)
        # Suggestion: Add dropout
        self.dropout = nnx.Dropout(0.1, rngs=rngs)

    def __call__(self, x, mask=None, *, deterministic=False):
        batch_size, seq_len, input_dim = x.shape
        # Project and split heads
        qkv = self.qkv_layer(x)
        # Using -1 for the last dim is often cleaner
        qkv = qkv.reshape(batch_size, seq_len, self.num_heads, 3, self.head_dim)
        # Transpose to (3, batch, heads, seq, head_dim)
        qkv = qkv.transpose((3, 0, 2, 1, 4))
        q, k, v = qkv[0], qkv[1], qkv[2]
        # Scaled dot-product attention
        # Note: Ensure your scaled_dot_product handles the 1/sqrt(dk)
        values, attn_weights = scaled_dot_product(
            q, k, v, mask=mask, dropout_fn=self.dropout if not deterministic else None
        )
        # Recombine heads: (b, s, h, d_h) -> (b, s, d)
        values = values.transpose((0, 2, 1, 3)).reshape(
            batch_size, seq_len, self.embed_dim
        )

        return self.linear_layer(values)


class TransformerBlock(nnx.Module):
    def __init__(self, input_dim, embed_dim, num_heads, *, rngs):
        self.attn = MultiHeadAttention(
            input_dim=input_dim,
            embed_dim=embed_dim,
            num_heads=num_heads,
            rngs=rngs,
        )

    def __call__(self, x, mask=None):
        attn_out = self.attn(x, mask=mask)
        x = x + attn_out
        return x


class TokenAndPositionEmbedding(nnx.Module):
    def __init__(self, max_seq_len, vocab_size, embed_dim, *, rngs):
        self.token_emb = nnx.Embed(vocab_size, embed_dim, rngs=rngs)
        self.pos_emb = nnx.Embed(max_seq_len, embed_dim, rngs=rngs)

    def __call__(self, x):
        seq_len = x.shape[1]
        positions = jnp.atleast_2d(jnp.arange(seq_len))
        return self.token_emb(x) + self.pos_emb(positions)


###############################################################################
# MiniGPT
###############################################################################


class MiniGPT(nnx.Module):
    def __init__(
        self,
        max_seq_len,
        vocab_size,
        embed_dim,
        num_heads,
        num_transformer_blocks,
        *,
        rngs,
    ):
        self.max_seq_len = max_seq_len
        self.embedding = TokenAndPositionEmbedding(
            max_seq_len=max_seq_len,
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            rngs=rngs,
        )
        self.transformer_blocks = nnx.data(
            [
                TransformerBlock(
                    input_dim=embed_dim,
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    rngs=rngs,
                )
                for _ in range(num_transformer_blocks)
            ]
        )
        self.dropout = nnx.Dropout(0.1, rngs=rngs)
        self.output_layer = nnx.Linear(embed_dim, vocab_size, use_bias=False, rngs=rngs)

    def __call__(self, token_ids, *, deterministic=False):
        seq_len = token_ids.shape[1]
        mask = causal_attention_mask(seq_len)
        x = self.embedding(token_ids)
        x = self.dropout(x, deterministic=deterministic)
        for block in self.transformer_blocks:
            x = block(x, mask=mask)

        logits = self.output_layer(x)
        return logits


###############################################################################
# MiniTabPFN
###############################################################################


class FeatureEncoder(nnx.Module):
    def __init__(
        self,
        embed_dim,
        *,
        rngs,
    ):
        self.linear = nnx.Linear(1, embed_dim, rngs=rngs)

    def __call__(self, x, train_test_split_index):
        x = jnp.atleast_2d(x)
        if x.ndim == 2:
            x = x[..., None]

        # Create a boolean mask for the training part (static shape)
        # mask shape: (1, seq_len, 1)
        mask = jnp.arange(x.shape[1])[None, :, None] < train_test_split_index

        # Compute stats using the 'where' argument to ignore test data
        # This avoids dynamic slicing and the resulting IndexError
        mean = jnp.mean(x, axis=1, where=mask, keepdims=True)
        std = jnp.std(x, axis=1, where=mask, keepdims=True)

        # Handle potential NaNs if split index is 0 or all values are the same
        mean = jnp.where(jnp.isnan(mean), 0.0, mean)
        std = jnp.where(jnp.isnan(std) | (std == 0), 1.0, std)

        x = (x - mean) / (std + 1e-6)
        x = jnp.clip(x, min=-100, max=100)
        x = jnp.expand_dims(x, -1)
        return self.linear(x)


class TargetEncoder(nnx.Module):
    def __init__(
        self,
        embed_dim,
        *,
        rngs,
    ):
        self.linear = nnx.Linear(1, embed_dim, rngs=rngs)

    def __call__(self, y, num_rows, *, deterministic=False):
        mean = jnp.mean(y, axis=1, keepdims=True)
        padding = mean.repeat(num_rows - y.shape[1], 1)
        y = jnp.concatenate([y, padding], axis=1)
        y = jnp.expand_dims(y, -1)
        return self.linear(y)


class MiniTabPFN(nnx.Module):
    def __init__(
        self,
        embed_dim,
        output_dim,
        num_heads,
        num_transformer_blocks,
        *,
        rngs,
    ):
        self.feature_encoder = FeatureEncoder(
            embed_dim=embed_dim,
            rngs=rngs,
        )
        self.target_encoder = TargetEncoder(
            embed_dim=embed_dim,
            rngs=rngs,
        )
        self.transformer_blocks = nnx.data(
            [
                TransformerBlock(
                    input_dim=embed_dim,
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    rngs=rngs,
                )
                for _ in range(num_transformer_blocks)
            ]
        )
        self.linear1 = nnx.Linear(embed_dim, embed_dim, rngs=rngs)
        self.linear2 = nnx.Linear(embed_dim, output_dim, rngs=rngs)

    def __call__(self, x, y, train_test_split_index):
        x = self.feature_encoder(x, train_test_split_index)
        num_rows = x.shape[1]
        y = self.target_encoder(y, num_rows)
        z = jnp.concatenate([x, y], axis=2)

        batch_size, rows_size, col_size, embedding_size = z.shape
        z = z.reshape(batch_size * rows_size, col_size, embedding_size)
        for block in self.transformer_blocks:
            z = block(z)

        output = z.reshape(batch_size, rows_size, col_size, embedding_size)
        # output = z[:, :, -1:, :]
        output = self.linear2(nnx.gelu(self.linear1(output)))
        return output[:, :, -1]
