import numpy as np

def softmax(x, axis=-1):
    """Provided: Softmax function."""
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

def layer_norm(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Apply layer normalization.
    """
    u = np.mean(x, axis = -1, keepdims=True)
    var = np.var(x, axis = -1, keepdims= True)
    return gamma * ((x - u) / np.sqrt(var + eps)) + beta
    

def multi_head_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                         W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                         W_o: np.ndarray, num_heads: int) -> np.ndarray:
    """
    Multi-head attention.
    """
    batch , seq, d_model = Q.shape
    d_head = d_model // num_heads

    q = Q @ W_q
    k = K @ W_k
    v = V @ W_v

    q = q.reshape(batch, seq, num_heads, d_head)
    k = k.reshape(batch, seq, num_heads, d_head)
    v = v.reshape(batch, seq, num_heads, d_head)

    q = q.transpose(0,2,1,3)
    k = k.transpose(0,2,1,3)
    v = v.transpose(0,2,1,3)

    attention_score = softmax((q @ k.transpose(0,1,3,2)) / np.sqrt(d_head) ) @ v

    multi_head = attention_score.transpose(0,2,1,3).reshape(batch, seq, d_model) @ W_o
    return multi_head

def feed_forward(x: np.ndarray, W1: np.ndarray, b1: np.ndarray,
                 W2: np.ndarray, b2: np.ndarray) -> np.ndarray:
    """
    Position-wise feed-forward network.
    """
    return np.maximum(0, x @ W1 + b1) @ W2 + b2
    

def encoder_block(x: np.ndarray, W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                  W_o: np.ndarray, W1: np.ndarray, b1: np.ndarray, W2: np.ndarray,
                  b2: np.ndarray, gamma1: np.ndarray, beta1: np.ndarray,
                  gamma2: np.ndarray, beta2: np.ndarray, num_heads: int) -> np.ndarray:
    """
    Complete encoder block: MHA + FFN with residuals and layer norms.
    """
    vecto_multi_head = multi_head_attention(x,x,x,
                         W_q, W_k, W_v,
                         W_o, num_heads)
    vecto_layer_norm1 = layer_norm(x + vecto_multi_head, gamma1, beta1)
    
    vecto_ff = feed_forward(vecto_layer_norm1, W1, b1,
                 W2, b2)

    vecto_layer_norm2 = layer_norm(vecto_layer_norm1 + vecto_ff, gamma2, beta2)

    return vecto_layer_norm2