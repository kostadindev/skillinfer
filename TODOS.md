# TODOS — skillinfer

## v0.2.0 Roadmap

- [ ] **Active Evaluation Scheduler** (`next_eval()`): Multi-entity, budget-constrained evaluation scheduling. Given N agents and a budget of M benchmark runs, decide which (agent, benchmark) cell to fill next, maximizing information gain via the Kalman posterior covariance.
- [ ] **Thompson sampling** in `rank_agents()`: `metric="thompson"` draws from the posterior for exploration. Requires handling near-singular covariance via `scipy.stats.multivariate_normal(allow_singular=True)` or Cholesky with jitter.
- [ ] **Process noise**: Time-varying capabilities for agents that improve/degrade over time. Add Q matrix to the Kalman update for non-stationary features.
- [ ] **GP baseline comparison**: Benchmark against Gaussian Processes for K=6 to honestly evaluate the accuracy/speed tradeoff vs GPs.

## Nice-to-haves

- [ ] Cache PCA in `Taxonomy.__str__()` to avoid recomputing on every print
- [ ] Fuzzy match suggestions in error messages ("did you mean 'MATH Lvl 5'?")
- [ ] `Taxonomy.new_states(names)` helper for batch state creation
- [ ] Probit/ordinal link for binary/ordinal data (O*NET, student assessments)
