# The Kalman Update

The core of `skillinfer` is a single equation: the multivariate Kalman filter update. When you call `profile.observe(j, y)`, this is what happens.

## The update equations

Given the current posterior belief (mean $\mu$, covariance $\Sigma$), observing feature $j$ with value $y$:

$$
\text{innovation} = y - \mu_j
$$

$$
K = \frac{\Sigma_{:,j}}{\Sigma_{j,j} + \sigma^2_{\text{noise}}}
$$

$$
\delta = K \cdot \text{innovation}
$$

$$
h_i = \begin{cases} 1 & \text{if } i = j \text{ (observed feature)} \\ 1 - \mu_i & \text{if } \delta_i > 0 \\ \mu_i & \text{if } \delta_i \leq 0 \end{cases}
$$

$$
\mu_i \leftarrow \mu_i + \delta_i \cdot h_i
$$

$$
\Sigma \leftarrow \Sigma - K \cdot \Sigma_{j,:}^T
$$

where:

- $K$ is the **Kalman gain** — a K-dimensional vector
- $\sigma^2_{\text{noise}}$ is the observation noise variance (`noise ** 2`)
- $\Sigma_{:,j}$ is the $j$-th column of the covariance matrix
- $h_i$ is the **headroom** — how much room feature $i$ has to move in the direction of the update

## What each part does

### Innovation

The difference between what we observed ($y$) and what we expected ($\mu_j$). If the observation matches our prediction, nothing changes.

### Kalman gain

$K_i = \frac{\Sigma_{i,j}}{\Sigma_{j,j} + \sigma^2_{\text{noise}}}$

The gain for feature $i$ is proportional to $\Sigma_{i,j}$ — how much feature $i$ co-varies with the observed feature $j$:

- If $\Sigma_{i,j} > 0$: feature $i$ moves in the **same direction** as the innovation
- If $\Sigma_{i,j} < 0$: feature $i$ moves in the **opposite direction**
- If $\Sigma_{i,j} \approx 0$: feature $i$ is **unaffected**

The denominator $\Sigma_{j,j} + \sigma^2_{\text{noise}}$ normalizes by the total variance (prior uncertainty + measurement noise). In the implementation, this denominator is clamped to a minimum of $10^{-8}$ for numerical stability when the prior variance is near zero.

### Bounded mean update

The raw shift $\delta_i = K_i \times \text{innovation}$ is scaled by the **headroom** $h_i$ before being applied. Headroom measures how much room feature $i$ has to move in the direction of the update:

- If $\delta_i > 0$ (pushing up), headroom is $1 - \mu_i$ — distance to the upper bound
- If $\delta_i \leq 0$ (pushing down), headroom is $\mu_i$ — distance to the lower bound

This ensures predictions stay within $[0, 1]$ naturally. A feature already at 0.95 being pushed upward has only 0.05 headroom — it barely moves. A feature at 0.5 has 0.5 headroom in either direction and moves freely. Features asymptotically approach 0 and 1 but never cross.

The **observed feature $j$ is exempt** from headroom scaling — its scale factor is always 1.0, so it converges to the observed value regardless of its current position.

### Covariance update

The rank-1 update $\Sigma \leftarrow \Sigma - K \cdot \Sigma_{j,:}^T$ **shrinks** the covariance. After an observation:

- The observed feature's variance drops (we know more about it)
- Correlated features' variances also drop (we learned about them indirectly)
- The covariance between features decreases (less uncertainty = less room for co-variation)

## In code

The implementation in `skillinfer/_kalman.py`:

```python
def kalman_update(mu, Sigma, j, y_j, obs_noise):
    mu = mu.copy()
    Sigma = Sigma.copy()

    S_j = Sigma[:, j]
    denom = max(Sigma[j, j] + obs_noise ** 2, 1e-8)
    K_gain = S_j / denom
    delta = K_gain * (y_j - mu[j])

    # Bounded update: scale each predicted feature's shift by available headroom.
    # The observed feature j is exempt — it should converge to its observed value.
    headroom = np.where(delta > 0, 1.0 - mu, mu)
    scale = np.clip(headroom, 0.0, 1.0)
    scale[j] = 1.0
    mu += delta * scale

    Sigma -= np.outer(K_gain, S_j)

    # Numerical stability: enforce symmetry and positive diagonal
    Sigma = (Sigma + Sigma.T) * 0.5
    np.maximum(np.diag(Sigma), 1e-10, out=Sigma[np.diag_indices_from(Sigma)])

    return mu, Sigma
```

The headroom scaling ensures predictions stay in $[0, 1]$ without clipping. The symmetry enforcement and positive diagonal clamping prevent floating-point drift from breaking the covariance structure over many updates.

## Exactness and the bounded approximation

The standard Kalman update is the **exact Bayesian posterior** when:

1. The prior is Gaussian: $p(\mathbf{x}) = \mathcal{N}(\mu, \Sigma)$
2. The observation is a linear function of the state with Gaussian noise: $y_j = x_j + \epsilon$, where $\epsilon \sim \mathcal{N}(0, \sigma^2_{\text{noise}})$

Under these conditions, the posterior is also Gaussian, and the Kalman update computes its exact mean and covariance. No approximations, no sampling, no variational bounds.

The headroom scaling is a lightweight modification for bounded skill domains. Since a pure Gaussian model has unbounded support, large innovations can push correlated features outside $[0, 1]$. The headroom factor dampens updates as features approach the boundary — a feature at 0.95 barely moves upward, while a feature at 0.5 moves freely. This trades exact Gaussianity for bounded predictions while preserving the covariance transfer that makes the model useful.

## Batch updates

When observing multiple features, `skillinfer` applies the updates **sequentially** — each observation conditions on the updated posterior from the previous one:

```python
def kalman_update_batch(mu, Sigma, obs_indices, obs_values, obs_noise):
    mu = mu.copy()
    Sigma = Sigma.copy()
    for j, y_j in zip(obs_indices, obs_values):
        S_j = Sigma[:, j]
        denom = max(Sigma[j, j] + obs_noise ** 2, 1e-8)
        K_gain = S_j / denom
        delta = K_gain * (y_j - mu[j])
        headroom = np.where(delta > 0, 1.0 - mu, mu)
        scale = np.clip(headroom, 0.0, 1.0)
        scale[j] = 1.0
        mu += delta * scale
        Sigma -= np.outer(K_gain, S_j)
    return mu, Sigma
```

For the standard Kalman filter, sequential application is equivalent to the joint multivariate update (observing all features simultaneously). With headroom scaling, the order of observations can produce slightly different results — the headroom depends on the current mean, which changes between updates. In practice the difference is negligible.

## The diagonal baseline

For comparison, the diagonal update only updates the observed feature:

```python
def diagonal_update(mu, var, j, y_j, obs_noise):
    gain = var[j] / (var[j] + obs_noise ** 2)
    mu[j] += gain * (y_j - mu[j])
    var[j] *= (1 - gain)
    return mu, var
```

This is what you get when you ignore off-diagonal covariance — no information transfers between features. The difference between Kalman and diagonal performance measures how much value the covariance structure adds.
