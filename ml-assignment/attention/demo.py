import numpy as np
from scaled_attention import scaled_dot_product_attention

def main():
    # Example Q, K, V (batch=1, seq_len=3, depth=4)
    Q = np.array([[
        [1.0, 0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0, 1.0],
        [1.0, 1.0, 1.0, 1.0]
    ]])

    K = np.array([[
        [1.0, 0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0, 1.0],
        [1.0, 1.0, 1.0, 1.0]
    ]])

    V = np.array([[
        [10, 0, 10, 0],
        [0, 10, 0, 10],
        [5, 5, 5, 5]
    ]])

    output, weights = scaled_dot_product_attention(Q, K, V)

    print("Attention Weights:\n", weights)
    print("\nOutput:\n", output)

if __name__ == "__main__":
    main()
