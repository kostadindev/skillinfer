# The Kalman Update

The core of `skillinfer` is a single equation: the multivariate Gaussian conditioning rule. When you call `profile.observe(j, y)`, this is what happens.

## The update equations

Let $J$ be the indices of all observed features and $\mathbf{y}_J$ the observed values. Given the prior $\mathcal{N}(\mu_0, \Sigma_0)$ — usually the population mean and covariance — the posterior mean is computed in one shot:

$$
M = \Sigma_{J,J} + \sigma^2_{\text{noise}} I
$$

$$
\boldsymbol{\alpha} = M^{-1} (\mathbf{y}_J - \mu_{0,J})
$$

$$
\boldsymbol{\delta} = \Sigma_{:,J} \boldsymbol{\alpha}
$$

$$
\mu_{\text{post}} = \mu_0 + \boldsymbol{\delta}
$$

$$
\Sigma_{\text{post}} = \Sigma_0 - \Sigma_{:,J} M^{-1} \Sigma_{J,:}
$$

where $\sigma^2_{\text{noise}}$ is the observation noise variance.

## What each part does

### Innovation

$\mathbf{y}_J - \mu_{0,J}$ is the difference between observations and prior expectations. Zero innovation leaves everything unchanged.

### Cross-covariance transfer

$\Sigma_{:,J}$ is the cross-covariance between every feature and the observed features. It controls how the innovation propagates:

- $\Sigma_{i,J} > 0$: feature $i$ moves in the **same direction** as the innovation
- $\Sigma_{i,J} < 0$: feature $i$ moves in the **opposite direction**
- $\Sigma_{i,J} \approx 0$: feature $i$ is **unaffected**

### Posterior covariance

The Schur-complement term $\Sigma_{:,J} M^{-1} \Sigma_{J,:}$ is subtracted from the prior covariance, shrinking uncertainty for both observed features and any features correlated with them.

## In code

The implementation in `skillinfer/_kalman.py`:

```python
def condition(prior_mean, pop_cov, obs_indices, obs_values, obs_noise):
    J = np.asarray(obs_indices, dtype=int)
    y = np.asarray(obs_values, dtype=float)

    S_J = pop_cov[:, J]                        # (K, n_obs)
    S_JJ = pop_cov[np.ix_(J, J)]               # (n_obs, n_obs)
    M = S_JJ + obs_noise ** 2 * np.eye(len(J))

    innovation = y - prior_mean[J]
    alpha = np.linalg.solve(M, innovation)
    delta = S_J @ alpha

    return prior_mean + delta
```

The population covariance is treated as a read-only reference — it is never mutated. Each `Profile` recomputes its posterior from $\Sigma_0$ and the current set of observations, so the order of `observe()` calls does not matter and the math is exact.

Predictions are clipped to $[0, 1]$ when reported (`profile.predict()`, `profile.mean()`) to match the population's natural scale.

## Exactness

The Gaussian conditioning rule is the **exact Bayesian posterior** when:

1. The prior is Gaussian: $p(\mathbf{x}) = \mathcal{N}(\mu_0, \Sigma_0)$
2. Each observation is linear in the state with Gaussian noise: $y_j = x_j + \epsilon$, where $\epsilon \sim \mathcal{N}(0, \sigma^2_{\text{noise}})$

Under these conditions the posterior is also Gaussian and `condition()` computes its exact mean. No iteration, no sampling, no variational bounds.

For bounded skill scales (e.g. $[0, 1]$), the Gaussian model is an approximation: the posterior has unbounded support, so the reported mean is clipped at the boundaries. In practice the bias from clipping is small as long as observations are not consistently near the extremes.

## The diagonal baseline

For comparison, ignoring the off-diagonal covariance gives the diagonal update — only the observed feature is changed:

```python
def diagonal_update(mu, var, j, y_j, obs_noise):
    gain = var[j] / (var[j] + obs_noise ** 2)
    mu[j] += gain * (y_j - mu[j])
    var[j] *= (1 - gain)
    return mu, var
```

The difference between Kalman and diagonal performance measures how much value the covariance structure adds.
