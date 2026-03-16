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
        # self.attn = nnx.MultiHeadAttention(
        #     num_heads=num_heads,
        #     in_features=input_dim,
        #     qkv_features=embed_dim,
        #     out_features=embed_dim,
        #     decode=False,
        #     rngs=rngs,
        # )
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

    def __call__(self, token_ids):
        seq_len = token_ids.shape[1]
        mask = causal_attention_mask(seq_len)
        x = self.embedding(token_ids)
        x = self.dropout(x)
        for block in self.transformer_blocks:
            x = block(x, mask=mask)

        logits = self.output_layer(x)
        return logits


class MiniQA(nnx.Module):
    def __init__(self, base_model: nnx.Module, hidden_dim: int, rngs: nnx.Rngs):
        self.base_model = base_model
        self.qa_head = nnx.Linear(in_features=hidden_dim, out_features=2, rngs=rngs)

    def __call__(self, input_ids):
        hidden_states = self.base_model(input_ids)
        logits = self.qa_head(hidden_states)
        start_logits, end_logits = jnp.split(logits, 2, axis=-1)
        start_logits = jnp.squeeze(start_logits, axis=-1)
        end_logits = jnp.squeeze(end_logits, axis=-1)

        return start_logits, end_logits
