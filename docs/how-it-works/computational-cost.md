# Computational Cost

`skillinfer` is designed to be fast. No neural networks, no training loops, no GPU. Just matrix algebra.

## Complexity

| Operation | Time | Memory |
|-----------|------|--------|
| **Build population** (`from_dataframe`) | $O(N \cdot K^2)$ | $O(K^2)$ |
| **Single observation** (`observe`) | $O(K^2)$ | $O(K^2)$ |
| **Batch observation** (`observe_many`) | $O(n \cdot K^2)$ | $O(K^2)$ |
| **Query mean/std** | $O(K)$ | — |
| **Most uncertain** | $O(K \log K)$ | — |
| **Similarity** | $O(K)$ | — |

Where N = number of entities, K = number of features, n = number of observations.

## Building the population

The one-time cost is dominated by covariance estimation:

- **Ledoit-Wolf**: $O(N \cdot K^2)$ — fits the shrinkage estimator
- **Sample covariance**: $O(N \cdot K^2)$ — computes `np.cov`

For typical use cases:

| Dataset | N | K | Build time |
|---------|---|---|------------|
| LLM benchmarks | 4,576 | 6 | < 10 ms |
| O\*NET | 894 | 120 | < 100 ms |
| Large population | 10,000 | 500 | < 1 s |

## Per-observation cost

Each `observe()` call performs:

1. One column extraction: $O(K)$
2. One division: $O(K)$
3. One vector-scalar multiply: $O(K)$
4. One outer product and subtraction: $O(K^2)$

The dominant cost is the rank-1 covariance update (outer product), which is $O(K^2)$.

For K = 120 features (O\*NET), each observation takes approximately **0.1 ms**. For K = 6 features (LLM benchmarks), it's effectively instantaneous.

## Scaling

The method scales comfortably to K = 1,000+ features. The bottleneck is memory for the $K \times K$ covariance matrix:

| K | Covariance memory | Per-observation time |
|---|-------------------|---------------------|
| 6 | 288 B | ~1 μs |
| 120 | 115 KB | ~0.1 ms |
| 500 | 2 MB | ~1 ms |
| 1,000 | 8 MB | ~5 ms |
| 5,000 | 200 MB | ~100 ms |

## Comparison with alternatives

| Method | Per-prediction cost | Setup cost | GPU needed |
|--------|-------------------|------------|------------|
| **skillinfer** | $O(K^2)$ | $O(N K^2)$ | No |
| Neural collaborative filtering | $O(d)$ forward pass | Hours of training | Usually |
| Matrix factorization (SVD) | $O(K \cdot r)$ | $O(N K r)$ | No |
| Gaussian Process | $O(N^3)$ | $O(N^3)$ | No |

`skillinfer` has no training loop — the "setup" is a single covariance estimation. Predictions are matrix-vector products, not forward passes through a network.
