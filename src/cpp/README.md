# C++ Optimizations

This directory contains C++ implementations for performance-critical components.

## Building

To build the C++ extensions, you'll need:
- C++ compiler (GCC, Clang, or MSVC)
- pybind11
- CMake (optional)

### Using pybind11

Example for creating Python bindings:

```cpp
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

// Your C++ code here

PYBIND11_MODULE(world_model_cpp, m) {
    m.doc() = "World Model C++ extensions";
    
    // Expose functions/classes
}
```

### Building

```bash
# Install pybind11
pip install pybind11

# Compile (example)
c++ -O3 -Wall -shared -std=c++11 -fPIC \
    `python3 -m pybind11 --includes` \
    world_model_cpp.cpp -o world_model_cpp`python3-config --extension-suffix`
```

## Components

- **Inference acceleration**: Fast forward passes for VAE, RNN, and controller
- **Data loading**: Efficient data preprocessing and batching
- **Planning**: Optimized CEM and MPC implementations

## Note

C++ implementations are optional. The Python implementations are fully functional,
but C++ can provide significant speedups for real-time applications.

