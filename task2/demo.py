import numpy as np
from attention import scaled_dot_product_attention

def demo():
    # Example matrices
    Q = np.array([[[1., 0., 0.]]])      # shape (1, 1, 3)
    K = np.array([[[1., 0., 0.]]])      # shape (1, 1, 3)
    V = np.array([[[5., 10.]]])         # shape (1, 1, 2)

    output, attn = scaled_dot_product_attention(Q, K, V)

    print("Query:\n", Q)
    print("Key:\n", K)
    print("Value:\n", V)
    print("Attention Weights:\n", attn)
    print("Output:\n", output)

if __name__ == "__main__":
    demo()
