import numpy as np

def softmax(x, axis=-1):
    """
    Numerically stable softmax implementation.
    """
    x_max = np.max(x, axis=axis, keepdims=True)
    exps = np.exp(x - x_max)
    return exps / np.sum(exps, axis=axis, keepdims=True)


def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Implements Scaled Dot-Product Attention using only NumPy.

    Args:
        Q: Query matrix  (..., seq_len_q, d_k)
        K: Key matrix    (..., seq_len_k, d_k)
        V: Value matrix  (..., seq_len_v, d_v)
        mask: optional mask that prevents attention to certain positions.

    Returns:
        output: attention output (..., seq_len_q, d_v)
        attention_weights: attention weights (..., seq_len_q, seq_len_k)
    """

    # --- Step 1: Compute raw attention scores (QKᵀ) ---
    scores = np.matmul(Q, np.swapaxes(K, -1, -2))

    # --- Step 2: Scale by sqrt(d_k) ---
    d_k = Q.shape[-1]
    scores = scores / np.sqrt(d_k)

    # --- Step 3: Apply mask (if any) ---
    if mask is not None:
        scores = np.where(mask, -1e9, scores)

    # --- Step 4: Softmax → attention weights ---
    attention_weights = softmax(scores, axis=-1)

    # --- Step 5: Multiply weights by V ---
    output = np.matmul(attention_weights, V)

    return output, attention_weights
