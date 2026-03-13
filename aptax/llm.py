import flax.nnx as nnx
import jax
import jax.numpy as jnp


def causal_attention_mask(seq_len):
    """Create a causal attention mask. The mask is a lower triangular matrix
    of shape (seq_len, seq_len) where the entries above the diagonal are -inf
    and the entries on and below the diagonal are 0.
    """
    return jnp.tril(jnp.ones((seq_len, seq_len)))


def scaled_dot_product(q, k, v, mask=None):
    """Compute scaled dot-product attention.

    Args:
    q: (batch, heads, seq_len, head_dim)
    k: (batch, heads, seq_len, head_dim)
    v: (batch, heads, seq_len, head_dim)
    mask: (seq_len, seq_len) or None

    Returns:
    values: (batch, heads, seq_len, head_dim)
    attention: (batch, heads, seq_len, seq_len)
    """
    d_k = q.shape[-1]
    # (batch, heads, seq_len, head_dim) @ (batch, heads, head_dim, seq_len) --> (batch, heads, seq_len, seq_len)
    scaled = jnp.matmul(q, k.transpose(0, 1, 3, 2)) / jnp.sqrt(d_k)
    if mask is not None:
        scaled += mask
    attention = jax.nn.softmax(scaled, axis=-1)
    # (batch, heads, seq_len, seq_len) @ (batch, heads, seq_len, head_dim) --> (batch, heads, seq_len, head_dim)
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

    def __call__(self, x, mask=None):
        batch_size, seq_len, input_dim = x.shape
        # (batch_size, seq_len, 3 * embed_dim)
        qkv = self.qkv_layer(x)
        # (batch_size, seq_len, num_heads, 3 * head_dim)
        qkv = qkv.reshape(batch_size, seq_len, self.num_heads, 3 * self.head_dim)
        # (batch_size, num_heads, seq_len, 3 * head_dim)
        qkv = jnp.permute_dims(qkv, (0, 2, 1, 3))
        # q (batch_size, num_heads, seq_len, head_dim)
        # k (batch_size, num_heads, seq_len, head_dim)
        # v (batch_size, num_heads, seq_len, head_dim)
        q, k, v = jnp.split(qkv, 3, axis=-1)
        # values (batch_size, num_heads, seq_len, head_dim)
        # attn (batch_size, num_heads, seq_len, seq_len)
        values, attn = scaled_dot_product(q, k, v, mask=mask)
        # (batch_size, seq_len, num_heads, head_dim)
        values = jnp.permute_dims(values, (0, 2, 1, 3))
        # (batch_size, seq_len, num_heads * head_dim)
        values = values.reshape(batch_size, seq_len, self.num_heads * self.head_dim)
        # (batch_size, seq_len, num_heads * head_dim)
        out = self.linear_layer(values)

        return out


class TransformerBlock(nnx.Module):
    def __init__(self, input_dim, embed_dim, num_heads, *, rngs):
        self.attn = nnx.MultiHeadAttention(
            num_heads=num_heads,
            in_features=input_dim,
            qkv_features=embed_dim,
            out_features=embed_dim,
            decode=False,
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


class MiniGPT(nnx.Module):
    def __init__(
        self,
        max_seq_len,
        vocab_size,
        embed_dim,
        num_heads,
        ff_dim,
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

    def __call__(self, token_ids):
        seq_len = token_ids.shape[1]
        mask = causal_attention_mask(seq_len)
        x = self.embedding(token_ids)
        x = self.dropout(x)
        for block in self.transformer_blocks:
            x = block(x, mask=mask)

        logits = self.output_layer(x)
        return logits
