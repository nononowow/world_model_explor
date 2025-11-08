# LSTM and GRU

## Overview

Long Short-Term Memory (LSTM) and Gated Recurrent Unit (GRU) are recurrent neural network architectures designed to handle long-term dependencies in sequential data. They are essential components of MDN-RNN in world models.

## The Problem with Standard RNNs

Standard RNNs suffer from:
- **Vanishing gradients**: Gradients become very small over long sequences
- **Exploding gradients**: Gradients become very large (less common)
- **Short-term memory**: Difficulty remembering information from many steps ago

## LSTM (Long Short-Term Memory)

### Architecture

LSTM introduces a **cell state** `C_t` that can maintain information over long periods, controlled by three gates:

1. **Forget Gate**: Decides what to discard from cell state
2. **Input Gate**: Decides what new information to store
3. **Output Gate**: Decides what parts of cell state to output

### Mathematical Formulation

#### Forget Gate

$$f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)$$

Decides what information to throw away from cell state.

#### Input Gate

$$i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i)$$
$$\tilde{C}_t = \tanh(W_C \cdot [h_{t-1}, x_t] + b_C)$$

Decides what new information to store in cell state.

#### Cell State Update

$$C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t$$

Combines old and new information.

#### Output Gate

$$o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o)$$
$$h_t = o_t \odot \tanh(C_t)$$

Decides what to output based on cell state.

### Key Components

- **Cell State `C_t`**: The "memory" that flows through the network
- **Hidden State `h_t`**: The output at each timestep
- **Gates**: Control information flow (all use sigmoid: outputs in [0,1])

## GRU (Gated Recurrent Unit)

### Architecture

GRU is a simpler variant with two gates:
1. **Reset Gate**: Decides how much past information to forget
2. **Update Gate**: Decides how much new information to incorporate

### Mathematical Formulation

#### Reset Gate

$$r_t = \sigma(W_r \cdot [h_{t-1}, x_t] + b_r)$$

#### Update Gate

$$z_t = \sigma(W_z \cdot [h_{t-1}, x_t] + b_z)$$

#### Candidate Activation

$$\tilde{h}_t = \tanh(W \cdot [r_t \odot h_{t-1}, x_t] + b)$$

#### Hidden State Update

$$h_t = (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t$$

### Comparison with LSTM

| Feature | LSTM | GRU |
|---------|------|-----|
| Gates | 3 (forget, input, output) | 2 (reset, update) |
| Cell State | Separate `C_t` | Combined with `h_t` |
| Parameters | More | Fewer |
| Complexity | Higher | Lower |
| Performance | Often better for long sequences | Often faster to train |

## Why They Work

### Gradient Flow

The gates allow gradients to flow through:
- **Additive updates**: `C_t = f_t ⊙ C_{t-1} + i_t ⊙ C̃_t` (addition preserves gradients)
- **Gating mechanism**: Controls what information to keep/forget
- **Tanh activation**: Bounded, helps with stability

### Long-Term Dependencies

- **Cell state**: Can maintain information across many timesteps
- **Forget gate**: Can learn to keep important information
- **Gradient preservation**: Additive structure prevents vanishing gradients

## In World Models

### MDN-RNN Uses LSTM

In world models, LSTM is used to:
1. Process sequences of (latent_state, action) pairs
2. Maintain hidden state across timesteps
3. Predict future latent states with uncertainty

### Architecture

```
Input: (z_t, a_t)  →  LSTM  →  Hidden State  →  MDN Output
                      ↑
                 Previous hidden state
```

### Why LSTM?

- **Temporal dependencies**: Need to remember past states
- **Long sequences**: Planning horizons can be long
- **Uncertainty**: Hidden state captures uncertainty over time

## Implementation Considerations

### Bidirectional LSTM

Processes sequence in both directions:

$$h_t = [\overrightarrow{h}_t; \overleftarrow{h}_t]$$

Useful when future context helps, but not common in world models (causal).

### Stacked LSTM

Multiple LSTM layers:

```
Input → LSTM Layer 1 → LSTM Layer 2 → ... → Output
```

Increases model capacity.

### Dropout

Regularization technique:
- Applied to hidden states
- Prevents overfitting
- Common: 0.2-0.5 dropout rate

## Training Tips

1. **Gradient Clipping**: Prevent exploding gradients
   ```python
   torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
   ```

2. **Sequence Length**: Longer sequences = more memory, better long-term modeling

3. **Initialization**: Proper initialization important (e.g., Xavier, He)

4. **Learning Rate**: Often needs to be lower than feedforward networks

## Common Issues

### 1. Vanishing Gradients (Still Possible)

**Symptom**: Model doesn't learn long-term dependencies
**Solution**: Use gradient clipping, check initialization, use residual connections

### 2. Overfitting

**Symptom**: Good on training, poor on validation
**Solution**: Add dropout, reduce model size, more data

### 3. Slow Training

**Symptom**: Training takes very long
**Solution**: Use GRU instead of LSTM, reduce sequence length, use GPU

## References

- Hochreiter & Schmidhuber (1997). Long Short-Term Memory. Neural Computation.
- Cho et al. (2014). Learning Phrase Representations using RNN Encoder-Decoder. EMNLP.
- Goodfellow et al. (2016). Deep Learning. MIT Press.

