# Infinite horizon case: stationary distribution
_Notes based on `docs/QA/infinite_horizon_case.png`_

---

**User**

Explain everything in the picture, especially what the slide is trying to say, go deeper, and give an example.

---

**Cursor**

- The slide is about long-run (infinite-horizon) average reward RL. As the horizon `T → ∞`, the time-average return converges to an expectation over the stationary state-action distribution induced by the policy.
- Stationary distribution `μ` is defined by `μ = T μ`, where `T` is the state-action transition operator (dynamics + policy). This means the distribution is unchanged after one step; equivalently `(T - I) μ = 0`, so `μ` is an eigenvector of `T` with eigenvalue 1. Under standard ergodicity conditions (irreducible, aperiodic Markov chain), such a `μ` exists and is unique.
- Performance objective: maximize `E_{(s,a)~μ}[r(s,a)]`. Optimizing θ adjusts the policy, which in turn changes `μ`.
- Intuition: if you run the policy forever, the empirical frequency of visiting each `(s,a)` converges to `μ`. Averaging rewards over time becomes the same as averaging with respect to `μ`.
- The bottom diagram shows that repeatedly applying the transition operator `T` to `(s_t, a_t)` produces future `(s_{t+k}, a_{t+k})`; at stationarity, applying `T` leaves the distribution unchanged.

Deeper explanation
- Transition operator: for discrete `(s,a)`, `T[(s',a'), (s,a)]` is the probability of moving from `(s,a)` to `(s',a')` in one step under the policy (includes environment dynamics and policy action probabilities).
- Existence/uniqueness: ergodic chains converge to a single `μ` regardless of start state. Without ergodicity, multiple recurrent classes or periodicity can prevent convergence.
- Objective link: `lim_{T→∞} (1/T) Σ_{t=1}^T r(s_t, a_t) = E_{(s,a)~μ}[r(s,a)]` almost surely when the chain mixes; this justifies using `μ` for average-reward optimization.

Concrete example
- MDP with two states `s0`, `s1` and one action each. Transition probabilities (policy is deterministic, so actions are implicit):
  - From `s0`: stays in `s0` with 0.6, moves to `s1` with 0.4.
  - From `s1`: moves to `s0` with 0.2, stays in `s1` with 0.8.
- Stationary distribution solves `μ = μ P`, `μ0 + μ1 = 1`:
  - `μ0 = 0.6 μ0 + 0.2 μ1`
  - `μ1 = 0.4 μ0 + 0.8 μ1`
  - Result: `μ0 = 1/3`, `μ1 = 2/3`.
- If rewards are `r(s0)=1`, `r(s1)=3`, the long-run average reward is `μ0*1 + μ1*3 = (1/3)*1 + (2/3)*3 = 7/3 ≈ 2.33`, matching the expectation under the stationary distribution.

