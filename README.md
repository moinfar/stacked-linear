# Parallel Stacked Linear Modules for PyTorch

[![Tests][badge-tests]][tests]
[![Documentation][badge-docs]][documentation]
[![Codecov][badge-codecov]][link-codecov]

Efficient implementation of stacked linear modules in PyTorch, with support for output and stack subsetting.

## Features
- **`LinearLayer`**: A linear layer with support for efficient output subsetting.
- **`StackedLinearLayer`**: A parallelized linear layer that applies multiple independent transformations across different input stacks simultaneously. This is significantly more efficient than for loop over multiple `nn.Linear` layers. This is useful for specialized neural architectures like Additive Decoders.
- **Subsetting Support**: Both layers allow for subsetting output features during the forward pass, and `StackedLinearLayer` additionally supports subsetting stacks.

## Installation

```bash
pip install stacked-linear
```

Or install from source:

```bash
pip install git+https://github.com/moinfar/stacked-linear.git
```

## Quick Start

### Linear Layer with Output Subsetting

```python
import torch
from stacked_linear import LinearLayer

# Initialize a layer (10 inputs, 5 outputs)
layer = LinearLayer(10, 5)
x = torch.randn(2, 10)

# Forward pass on a subset of output features (indices 0, 2, and 4)
subset = torch.tensor([0, 2, 4])
output = layer(x, output_subset=subset)  # Shape: (2, 3)
```

### Stacked Linear Layer

```python
import torch
from stacked_linear import StackedLinearLayer

# 3 parallel stacks, each mapping 10 inputs to 5 outputs
layer = StackedLinearLayer(n_stacks=3, in_features=10, out_features=5)
x = torch.randn(2, 3, 10)  # (batch, stacks, features)

# Efficient parallel forward pass
output = layer(x)  # Shape: (2, 3, 5)

# Forward pass on a subset of output features across all stacks
subset = torch.tensor([1, 3])
output_subset = layer(x, output_subset=subset)  # Shape: (2, 3, 2)

# Forward pass on a subset of stacks
stack_subset = torch.tensor([[0, 2], [1, 2]]) # Indices for each batch item
x_subset = torch.randn(2, 2, 10)
output_stack_subset = layer(x_subset, stack_subset=stack_subset) # Shape: (2, 2, 5)
```


[badge-tests]: https://img.shields.io/github/actions/workflow/status/moinfar/stacked-linear/test.yaml?branch=main
[badge-docs]: https://img.shields.io/readthedocs/stacked-linear
[tests]: https://github.com/moinfar/stacked-linear/actions/workflows/test.yaml
[documentation]: https://stacked-linear.readthedocs.io/en/latest/
[issue tracker]: https://github.com/moinfar/stacked-linear/issues
[badge-codecov]: https://codecov.io/gh/moinfar/stacked-linear/graph/badge.svg?token=THDLKTI14L
[link-codecov]: https://codecov.io/gh/moinfar/stacked-linear
