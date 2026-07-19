import numpy as np

def softmax(x, axis=-1):
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

def multi_head_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                         W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                         W_o: np.ndarray, num_heads: int) -> np.ndarray:
    """
    Compute multi-head attention.
    """
    batch, seq, d_model = Q.shape
    d_head = d_model // num_heads

    q = Q @ W_q
    k = K @ W_k
    v = V @ W_v
    
    q = q.reshape(batch, seq, num_heads, d_head)
    k = k.reshape(batch, seq, num_heads, d_head)
    v = v.reshape(batch, seq, num_heads, d_head)

    q =  q.transpose(0,2,1,3)
    k =  k.transpose(0,2,1,3)
    v =  v.transpose(0,2,1,3)
   
    attention_softmax = softmax((q @ k.transpose(0,1,3,2)) / np.sqrt(d_head)) @ v
    attention_softmax = attention_softmax.transpose(0,2,1,3)
    attention_softmax = attention_softmax.reshape(batch, seq,d_model)
    return attention_softmax @ W_o
    