# Preventing model cheating in world models
_Exported on 11/11/2025 at 20:49:04 GMT+8 from Cursor (2.0.69)_

---

**User**

In the paper World Models(2018), which strategy make sure that the M model not cheat to get a higher score?

---

**Cursor**

They deliberately make the dreamed environment stochastic by sampling the MDN-RNN with a temperature slightly above 1 (e.g., τ ≈ 1.15 in their CarRacing run). The extra randomness keeps the controller from exploiting flaws in the learned world model, so the M model can’t “cheat” its way to an inflated score that wouldn’t transfer back to the real environment.

---

**User**

I donot understand, which strategy make sure that if the C output action: move_left, then the M will really output the move left data? not cheat? explain that in detail and make a example please.

---

**Cursor**

When Ha & Schmidhuber talk about “cheating,” they’re worried that the controller `C` might discover a flaw in the learned world model `M` and exploit it to earn fake reward during dream training. Two things in their pipeline keep that from happening:

- **Supervised training keeps `M` faithful to the real action→next state mapping.**  
  - They first collect real rollouts from the actual environment using random policies.  
  - Each frame is encoded by the VAE into a latent vector `z_t`.  
  - The MDN-RNN (`M`) is trained to model `p(z_{t+1} | z_t, a_t, h_t)` by maximum likelihood.  
  - If `M` ever tried to ignore the action (e.g., predict “go straight” regardless of `move_left`), the log-likelihood loss on all real `move_left` samples would spike and gradients push the model to depend on the action again.  
  - Once `M` is trained, it’s frozen; neither `C` nor the optimizer ever adjusts it, so there’s no way for `M` to drift toward lying about actions.

- **Injected stochasticity stops `C` from gaming small modeling errors.**  
  - During imagination rollouts they sample the MDN-RNN with a temperature slightly above 1 (≈1.15 in CarRacing).  
  - Because next latents are random draws, a fragile open-loop exploit (e.g., “if I always output `move_left` the model glitches into high reward”) falls apart—the randomness pushes trajectories back toward where the real data lived.  
  - Empirically this keeps the policy found in the dream from collapsing when tested in the true environment.

**Concrete example.**  
Suppose the current latent `z_t` encodes a position close to the right edge of the track. The real data says that executing action `a_t = move_left` usually shifts the car left by one lane, giving a latent `z_{t+1}` centered at that lane. During training `M` sees many tuples `(z_t, move_left, z_{t+1})` like this. If its prediction ignores the action and keeps the car in the same lane, the probability assigned to the real `z_{t+1}` is tiny, yielding a large loss. After training, when the frozen world model receives the same `z_t` and action `move_left` from the controller, it produces a Gaussian mixture whose high-probability region is exactly that left-shifted latent. Sampling (with temperature 1.15) might jitter the precise position a bit, but it remains consistent with “car moved left,” mirroring the statistics of the real environment.

So the guarantee comes from (1) supervised modeling of real trajectories conditioned on the action, and (2) stochastic sampling during imagination rollouts that prevents `C` from exploiting rare modeling artifacts.

