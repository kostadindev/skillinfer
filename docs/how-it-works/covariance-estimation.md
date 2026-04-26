# Covariance Estimation

The quality of `skillinfer`'s predictions depends on how well the covariance matrix captures the true relationships between features. This page explains how the covariance is estimated and why regularization matters.

## The problem

Given N entities and K features, we want to estimate the K x K covariance matrix $\Sigma$. The sample covariance is:

$$
\hat{\Sigma} = \frac{1}{N-1} \sum_{i=1}^{N} (x_i - \bar{x})(x_i - \bar{x})^T
$$

This works well when $N \gg K$, but becomes **ill-conditioned** when K approaches N:

- With K = 120 features and N = 894 entities (O\*NET), the sample covariance has $\frac{120 \times 121}{2} = 7{,}260$ free parameters estimated from 894 samples
- Small eigenvalues get pushed toward zero, making the matrix nearly singular
- The Kalman gain involves dividing by $\Sigma_{j,j}$, so numerical instability propagates to predictions

## Ledoit-Wolf shrinkage

`skillinfer` uses [Ledoit-Wolf shrinkage](https://scikit-learn.org/stable/modules/covariance.html#shrunk-covariance) (the default and recommended method). The shrinkage estimator is:

$$
\hat{\Sigma}_{\text{shrunk}} = (1 - \alpha) \hat{\Sigma} + \alpha \cdot \mu \cdot I
$$

where:

- $\hat{\Sigma}$ is the sample covariance
- $\mu = \text{tr}(\hat{\Sigma}) / K$ is the average variance
- $\alpha \in [0, 1]$ is the **shrinkage coefficient** (chosen automatically)
- $I$ is the identity matrix

The shrinkage coefficient $\alpha$ is computed analytically to minimize the expected loss (Frobenius norm between the estimator and the true covariance).

### What shrinkage does

- **Pulls small eigenvalues up** (prevents near-singularity)
- **Pulls large eigenvalues down** (reduces overfitting to sampling noise)
- **Preserves the overall correlation structure** while improving conditioning

### How much shrinkage?

The automatically chosen $\alpha$ depends on the ratio of features to entities:

| Scenario | $K / N$ | Typical $\alpha$ | Effect |
|----------|---------|-------------------|--------|
| LLM benchmarks | 6 / 4576 | 0.0006 | Almost no shrinkage needed |
| O\*NET | 120 / 894 | 0.024 | Moderate shrinkage |
| Small dataset | 50 / 100 | 0.15+ | Heavy shrinkage |

You can check the shrinkage coefficient via `pop.shrinkage`.

## Sample covariance (alternative)

For cases where you want the unregularized estimate:

```python
pop = skillinfer.Population.from_dataframe(df, covariance="sample")
```

This adds a small ridge ($10^{-6} \cdot I$) for numerical stability, but does not apply Ledoit-Wolf shrinkage. Use this only when $N \gg K$ and you're confident the sample covariance is well-conditioned.

## Condition number

The condition number measures how numerically stable the covariance matrix is:

```python
pop.condition_number()  # lower = more stable
```

$$
\kappa(\Sigma) = \frac{\lambda_{\max}}{\lambda_{\min}}
$$

- **< 100**: well-conditioned, no concerns
- **100 – 1000**: moderate, Ledoit-Wolf handles this well
- **> 1000**: high, consider whether you have enough entities for the number of features

## Correlation matrix

The correlation matrix is derived from the covariance:

$$
\text{Corr}_{i,j} = \frac{\Sigma_{i,j}}{\sqrt{\Sigma_{i,i}} \cdot \sqrt{\Sigma_{j,j}}}
$$

Access it via `pop.correlation`. Use `pop.top_correlations(k=20)` to see the strongest feature-feature relationships.

## PCA

PCA reveals the effective dimensionality of the feature space:

```python
pca = pop.pca(n_components=10)
print(pca["cumulative"])  # [0.71, 0.82, 0.89, ...]
```

If 3 components explain 90% of variance, the 120-feature space is effectively ~3-dimensional. This is good — it means there's strong covariance structure for the Kalman filter to exploit.

## Implementation

The covariance estimation internals are in `skillinfer/_covariance.py`:

```python
from sklearn.covariance import LedoitWolf

def ledoit_wolf_covariance(R):
    lw = LedoitWolf().fit(R)
    return lw.covariance_, lw.shrinkage_

def sample_covariance(R):
    Sigma = np.cov(R, rowvar=False)
    Sigma += np.eye(Sigma.shape[0]) * 1e-6  # ridge for stability
    return Sigma
```

The heavy lifting is done by scikit-learn's `LedoitWolf` estimator, which implements the analytical shrinkage formula from [Ledoit & Wolf (2004)](https://doi.org/10.1016/j.jmva.2004.02.003).
