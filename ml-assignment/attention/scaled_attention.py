import numpy as np

def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Implements Scaled Dot-Product Attention using only NumPy.

    Args:
        Q: Query matrix of shape (..., seq_len_q, depth)
        K: Key matrix of shape (..., seq_len_k, depth)
        V: Value matrix of shape (..., seq_len_k, depth_v)
        mask: Optional mask broadcastable to (..., seq_len_q, seq_len_k)

    Returns:
        output: Attention output (..., seq_len_q, depth_v)
        attention_weights: Softmax(QK^T / sqrt(d_k)) (..., seq_len_q, seq_len_k)
    """

    # 1. Compute raw attention scores QK^T
    scores = np.matmul(Q, K.transpose(0, 2, 1))

    # 2. Scale by sqrt(dk)
    dk = Q.shape[-1]
    scaled_scores = scores / np.sqrt(dk)

    # 3. Apply mask (if provided)
    if mask is not None:
        scaled_scores = scaled_scores + (mask * -1e9)

    # 4. Softmax along last axis
    exp_scores = np.exp(scaled_scores - np.max(scaled_scores, axis=-1, keepdims=True))
    attention_weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)

    # 5. Weighted sum: attention_weights @ V
    output = np.matmul(attention_weights, V)

    return output, attention_weights

