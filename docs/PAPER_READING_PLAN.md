# Paper Reading Plan: World Models to Autonomous Driving

## Overview

This reading plan guides you through the essential papers from foundational world models to advanced simulation and autonomous driving applications. Follow this progression to build a comprehensive understanding.

## Reading Strategy

1. **Read in Order**: Papers build on each other
2. **Take Notes**: Document key ideas and connections
3. **Implement**: Try to implement key concepts
4. **Compare**: Understand how methods evolved
5. **Apply**: Think about applications to your projects

## Timeline

- **Weeks 1-2**: Foundation (Papers 1-3)
- **Weeks 3-4**: Advanced World Models (Papers 4-6)
- **Weeks 5-6**: Simulation & Planning (Papers 7-9)
- **Weeks 7-8**: Autonomous Driving (Papers 10-12)
- **Weeks 9-10**: Recent Advances (Papers 13-15)

---

## Part 1: Foundation - Core World Models

### Paper 1: World Models (2018)
**Title**: World Models  
**Authors**: David Ha, Jürgen Schmidhuber  
**Venue**: arXiv:1803.10122  
**Link**: https://arxiv.org/abs/1803.10122

**Key Concepts**:
- Three-component architecture: VAE, MDN-RNN, Controller
- Latent space planning
- Training in imagination (world model)
- Evolutionary strategies for controller

**Why Read First**:
- Foundational paper for world models
- Introduces core concepts
- Relatively accessible

**Key Takeaways**:
- How to separate perception (VAE), dynamics (RNN), and control
- Benefits of planning in latent space
- Evolutionary strategies for policy learning

**Implementation Focus**:
- Implement the three-stage training
- Understand MDN-RNN for uncertainty

**Reading Time**: 3-4 hours  
**Difficulty**: ⭐⭐ (Medium)

---

### Paper 2: Learning Latent Dynamics for Planning from Pixels (2019)
**Title**: Learning Latent Dynamics for Planning from Pixels (PlaNet)  
**Authors**: Danijar Hafner, Timothy Lillicrap, et al.  
**Venue**: ICML 2019  
**Link**: https://arxiv.org/abs/1811.04551

**Key Concepts**:
- Recurrent state space model (RSSM)
- Latent dynamics model
- Model-based planning with CEM
- Learning from pixels

**Why Read**:
- Improves upon World Models
- Better latent dynamics modeling
- More systematic planning approach

**Key Takeaways**:
- RSSM architecture (deterministic + stochastic components)
- Planning with Cross-Entropy Method
- Learning dynamics from image sequences

**Implementation Focus**:
- Compare RSSM vs MDN-RNN
- Implement CEM planner

**Reading Time**: 4-5 hours  
**Difficulty**: ⭐⭐⭐ (Medium-Hard)

---

### Paper 3: Dreamer: Learning Models by Imagination (2019)
**Title**: Dreamer: Learning Models by Imagination  
**Authors**: Danijar Hafner, Timothy Lillicrap, et al.  
**Venue**: ICLR 2020  
**Link**: https://arxiv.org/abs/1912.01603

**Key Concepts**:
- Actor-critic learning in latent space
- Value function learning
- Policy gradients with world model
- End-to-end differentiable planning

**Why Read**:
- Combines model-based and model-free RL
- More sample-efficient than pure model-free
- Better policy learning than evolutionary strategies

**Key Takeaways**:
- How to learn value functions in latent space
- Actor-critic with world model
- Backpropagation through time for planning

**Implementation Focus**:
- Implement actor-critic in latent space
- Compare with evolutionary strategies

**Reading Time**: 4-5 hours  
**Difficulty**: ⭐⭐⭐ (Medium-Hard)

---

## Part 2: Advanced World Models & Improvements

### Paper 4: DreamerV2: Mastering Atari with Discrete World Models (2021)
**Title**: Mastering Atari with Discrete World Models  
**Authors**: Danijar Hafner, et al.  
**Venue**: ICLR 2021  
**Link**: https://arxiv.org/abs/2010.02193

**Key Concepts**:
- Discrete latent representations
- Vector quantization (VQ-VAE)
- Improved world model architecture
- State-of-the-art on Atari

**Why Read**:
- Significant improvement over Dreamer
- Discrete representations for better abstraction
- Strong empirical results

**Key Takeaways**:
- Benefits of discrete vs continuous latents
- Vector quantization for world models
- Scaling to complex environments

**Reading Time**: 3-4 hours  
**Difficulty**: ⭐⭐⭐ (Medium-Hard)

---

### Paper 5: DreamerV3: Mastering Diverse Domains Through World Models (2023)
**Title**: Mastering Diverse Domains Through World Models  
**Authors**: Danijar Hafner, et al.  
**Venue**: ICLR 2024  
**Link**: https://arxiv.org/abs/2301.04104

**Key Concepts**:
- Unified architecture for diverse domains
- Symlog predictions
- Improved training stability
- Works across continuous/discrete, 2D/3D

**Why Read**:
- Latest advances in world models
- More robust and generalizable
- Better handling of diverse action/reward spaces

**Key Takeaways**:
- Symlog transformation for better learning
- Unified architecture design
- Scaling world models to diverse tasks

**Reading Time**: 4-5 hours  
**Difficulty**: ⭐⭐⭐ (Medium-Hard)

---

### Paper 6: Model-Agnostic Meta-Learning for Fast Adaptation (2017)
**Title**: Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks  
**Authors**: Chelsea Finn, Pieter Abbeel, Sergey Levine  
**Venue**: ICML 2017  
**Link**: https://arxiv.org/abs/1703.03400

**Key Concepts**:
- Meta-learning
- Fast adaptation to new tasks
- Few-shot learning
- MAML algorithm

**Why Read**:
- Relevant for adapting world models to new scenarios
- Important for autonomous driving (different conditions)
- Understanding transfer learning

**Key Takeaways**:
- How to learn to learn quickly
- Meta-learning principles
- Adaptation strategies

**Reading Time**: 3-4 hours  
**Difficulty**: ⭐⭐⭐ (Medium-Hard)

---

## Part 3: Simulation & Planning

### Paper 7: Model-Predictive Policy Learning with Uncertainty Regularization (2019)
**Title**: Model-Predictive Policy Learning with Uncertainty Regularization for Driving in Dense Traffic  
**Authors**: Zhang et al.  
**Venue**: CoRL 2019  
**Link**: https://arxiv.org/abs/1901.05306

**Key Concepts**:
- MPC for autonomous driving
- Uncertainty regularization
- Dense traffic scenarios
- Safety constraints

**Why Read**:
- Direct application to driving
- MPC in real-world scenarios
- Handling uncertainty

**Key Takeaways**:
- How to apply MPC to driving
- Uncertainty-aware planning
- Safety considerations

**Reading Time**: 3-4 hours  
**Difficulty**: ⭐⭐⭐ (Medium-Hard)

---

### Paper 8: End-to-End Driving via Conditional Imitation Learning (2018)
**Title**: End-to-End Driving via Conditional Imitation Learning  
**Authors**: Codevilla et al.  
**Venue**: ICRA 2018  
**Link**: https://arxiv.org/abs/1710.02410

**Key Concepts**:
- Imitation learning for driving
- Conditional policies
- Multi-modal behavior
- CARLA simulator

**Why Read**:
- Important baseline for driving
- Imitation learning approach
- CARLA integration

**Key Takeaways**:
- How to learn from expert demonstrations
- Conditional policies for different behaviors
- Simulator integration

**Reading Time**: 3-4 hours  
**Difficulty**: ⭐⭐ (Medium)

---

### Paper 9: Learning to Drive in a Day (2018)
**Title**: Learning to Drive in a Day  
**Authors**: Kendall et al.  
**Venue**: ICRA 2019  
**Link**: https://arxiv.org/abs/1807.00412

**Key Concepts**:
- Fast learning for driving
- Real-world deployment
- End-to-end learning
- Safety considerations

**Why Read**:
- Real-world application
- Fast learning strategies
- Practical considerations

**Key Takeaways**:
- How to learn quickly
- Real-world deployment challenges
- Safety in learning systems

**Reading Time**: 2-3 hours  
**Difficulty**: ⭐⭐ (Medium)

---

## Part 4: Autonomous Driving - World Models

### Paper 10: World Models for Autonomous Driving (2020)
**Title**: World Models for Autonomous Driving  
**Authors**: Various (Search for recent papers)  
**Venue**: Various  
**Link**: Search arXiv for "world model autonomous driving"

**Key Concepts**:
- World models for driving scenarios
- Multi-modal predictions
- Uncertainty in driving
- Safety-aware planning

**Why Read**:
- Direct application to your interest
- Current state of the art
- Practical considerations

**Key Takeaways**:
- How world models apply to driving
- Multi-modal prediction challenges
- Safety requirements

**Reading Time**: 4-5 hours  
**Difficulty**: ⭐⭐⭐ (Medium-Hard)

---

### Paper 11: Multi-Modal Trajectory Prediction for Autonomous Driving (2019)
**Title**: Multi-Modal Trajectory Prediction for Autonomous Driving  
**Authors**: Various (e.g., Deo & Trivedi)  
**Venue**: Various  
**Link**: Search for recent trajectory prediction papers

**Key Concepts**:
- Predicting other vehicles' trajectories
- Multi-modal predictions
- Interaction modeling
- Uncertainty quantification

**Why Read**:
- Critical for autonomous driving
- Related to world models (predicting future)
- Multi-modality important

**Key Takeaways**:
- How to predict other agents
- Multi-modal distributions
- Interaction modeling

**Reading Time**: 3-4 hours  
**Difficulty**: ⭐⭐⭐ (Medium-Hard)

---

### Paper 12: Learning to Predict Vehicle Trajectories with Model-Based Planning (2021)
**Title**: Learning to Predict Vehicle Trajectories with Model-Based Planning  
**Authors**: Various  
**Venue**: Various  
**Link**: Search for recent papers

**Key Concepts**:
- Model-based planning for vehicles
- Trajectory prediction
- Planning with predictions
- Real-time constraints

**Why Read**:
- Combines prediction and planning
- Real-world constraints
- Model-based approach

**Key Takeaways**:
- Integration of prediction and planning
- Real-time requirements
- Practical implementation

**Reading Time**: 4-5 hours  
**Difficulty**: ⭐⭐⭐ (Medium-Hard)

---

## Part 5: Recent Advances & Specialized Topics

### Paper 13: Waymo Open Dataset: An Autonomous Driving Dataset (2020)
**Title**: Scalability in Perception for Autonomous Driving: Waymo Open Dataset  
**Authors**: Sun et al.  
**Venue**: CVPR 2020  
**Link**: https://arxiv.org/abs/1912.04838

**Key Concepts**:
- Large-scale driving datasets
- Multi-sensor data
- Real-world scenarios
- Evaluation metrics

**Why Read**:
- Understanding real-world data
- Dataset structure
- Evaluation standards

**Key Takeaways**:
- What real driving data looks like
- Multi-modal sensor fusion
- Evaluation metrics

**Reading Time**: 2-3 hours  
**Difficulty**: ⭐⭐ (Medium)

---

### Paper 14: nuScenes: A Multimodal Dataset for Autonomous Driving (2020)
**Title**: nuScenes: A Multimodal Dataset for Autonomous Driving  
**Authors**: Caesar et al.  
**Venue**: CVPR 2020  
**Link**: https://arxiv.org/abs/1903.11027

**Key Concepts**:
- Comprehensive driving dataset
- Multi-modal sensors
- 3D object detection
- Trajectory prediction

**Why Read**:
- Another important dataset
- 3D understanding
- Comprehensive annotations

**Key Takeaways**:
- 3D scene understanding
- Multi-modal data
- Annotation standards

**Reading Time**: 2-3 hours  
**Difficulty**: ⭐⭐ (Medium)

---

### Paper 15: CARLA: An Open Urban Driving Simulator (2017)
**Title**: CARLA: An Open Urban Driving Simulator  
**Authors**: Dosovitskiy et al.  
**Venue**: CoRL 2017  
**Link**: https://arxiv.org/abs/1711.03938

**Key Concepts**:
- Open-source driving simulator
- Realistic urban scenarios
- Sensor simulation
- Benchmarking platform

**Why Read**:
- Important tool for research
- Understanding simulation
- Benchmarking

**Key Takeaways**:
- How simulators work
- Realistic simulation challenges
- Using CARLA for research

**Reading Time**: 2-3 hours  
**Difficulty**: ⭐⭐ (Medium)

---

## Reading Schedule

### Week 1-2: Foundation
- [ ] Paper 1: World Models (2018)
- [ ] Paper 2: PlaNet (2019)
- [ ] Paper 3: Dreamer (2019)

**Focus**: Understand core concepts, implement basic world model

### Week 3-4: Advanced Models
- [ ] Paper 4: DreamerV2 (2021)
- [ ] Paper 5: DreamerV3 (2023)
- [ ] Paper 6: MAML (2017) - Optional but recommended

**Focus**: Advanced architectures, discrete representations

### Week 5-6: Simulation & Planning
- [ ] Paper 7: MPC for Driving (2019)
- [ ] Paper 8: Conditional Imitation Learning (2018)
- [ ] Paper 9: Learning to Drive in a Day (2018)

**Focus**: Planning methods, simulation, practical driving

### Week 7-8: Autonomous Driving
- [ ] Paper 10: World Models for Autonomous Driving
- [ ] Paper 11: Multi-Modal Trajectory Prediction
- [ ] Paper 12: Model-Based Planning for Vehicles

**Focus**: Direct applications to autonomous driving

### Week 9-10: Datasets & Tools
- [ ] Paper 13: Waymo Dataset (2020)
- [ ] Paper 14: nuScenes Dataset (2020)
- [ ] Paper 15: CARLA Simulator (2017)

**Focus**: Understanding data and tools for research

---

## Reading Tips

### Before Reading
1. **Check prerequisites**: Understand basic RL, VAEs, RNNs
2. **Read abstract and intro**: Get overview
3. **Look at figures**: Visual understanding helps

### While Reading
1. **Take notes**: Key concepts, formulas, insights
2. **Draw diagrams**: Visualize architectures
3. **Implement snippets**: Code key algorithms
4. **Ask questions**: What's unclear? What's novel?

### After Reading
1. **Summarize**: Write 1-page summary
2. **Compare**: How does this relate to previous papers?
3. **Implement**: Try to implement key ideas
4. **Discuss**: Talk about with others if possible

## Key Questions to Ask

For each paper, consider:
1. **What problem does it solve?**
2. **What's the key innovation?**
3. **How does it improve upon previous work?**
4. **What are the limitations?**
5. **How can I apply this to my work?**

## Additional Resources

### Review Papers
- "A Survey of Deep Learning for Autonomous Driving" (2021)
- "End-to-End Autonomous Driving: Challenges and Frontiers" (2021)

### Blogs & Tutorials
- Lil'Log (Lilian Weng) - World Models blog post
- Distill.pub - Interactive explanations
- Papers with Code - Implementations

### Communities
- Reddit: r/MachineLearning, r/SelfDrivingCars
- Twitter: Follow authors
- GitHub: Star implementations

## Tracking Your Progress

Create a reading log:

```
Paper: [Title]
Date Read: [Date]
Key Concepts: [List]
Implementation: [Yes/No]
Notes: [Your thoughts]
Rating: [1-5 stars]
```

## Next Steps After Reading

1. **Implement**: Code the key algorithms
2. **Experiment**: Try variations
3. **Apply**: Use in your projects
4. **Extend**: Build upon the ideas
5. **Share**: Write blog posts or tutorials

---

**Start with**: Paper 1 (World Models) - The foundation of everything!

**Remember**: Quality over quantity. It's better to deeply understand a few papers than to skim many.

