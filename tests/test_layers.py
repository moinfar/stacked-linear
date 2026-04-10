import torch

from stacked_linear import LinearLayer, StackedLinearLayer


def test_linear_layer_basic():
    in_features = 10
    out_features = 5
    batch_size = 2

    layer = LinearLayer(in_features, out_features)
    x = torch.randn(batch_size, in_features)
    output = layer(x)

    assert output.shape == (batch_size, out_features)
    assert isinstance(output, torch.Tensor)


def test_linear_layer_subset():
    in_features = 10
    out_features = 5
    batch_size = 2

    layer = LinearLayer(in_features, out_features)
    x = torch.randn(batch_size, in_features)

    subset = torch.tensor([0, 2, 4])
    output = layer(x, output_subset=subset)

    assert output.shape == (batch_size, 3)

    # Verify values match full forward pass
    full_output = layer(x)
    assert torch.allclose(output, full_output[:, subset])


def test_stacked_linear_layer_basic():
    n_stacks = 3
    in_features = 10
    out_features = 5
    batch_size = 2

    layer = StackedLinearLayer(n_stacks, in_features, out_features)
    x = torch.randn(batch_size, n_stacks, in_features)
    output = layer(x)

    assert output.shape == (batch_size, n_stacks, out_features)


def test_stacked_linear_layer_no_bias():
    n_stacks = 3
    in_features = 10
    out_features = 5
    batch_size = 2

    layer = StackedLinearLayer(n_stacks, in_features, out_features, bias=False)
    x = torch.randn(batch_size, n_stacks, in_features)
    output = layer(x)

    assert output.shape == (batch_size, n_stacks, out_features)
    assert layer.bias is None


def test_stacked_linear_layer_output_subset():
    n_stacks = 3
    in_features = 10
    out_features = 5
    batch_size = 2

    layer = StackedLinearLayer(n_stacks, in_features, out_features)
    x = torch.randn(batch_size, n_stacks, in_features)

    subset = torch.tensor([1, 3])
    output = layer(x, output_subset=subset)

    assert output.shape == (batch_size, n_stacks, 2)

    # Verify values
    full_output = layer(x)
    assert torch.allclose(output, full_output[:, :, subset])


def test_stacked_linear_layer_stack_subset():
    n_stacks = 5
    in_features = 10
    out_features = 5
    batch_size = 4

    layer = StackedLinearLayer(n_stacks, in_features, out_features)

    # Each item in batch uses a subset of stacks
    s_subset_size = 3
    # stack_subset shape (batch_size, s_subset_size)
    stack_subset = torch.stack([torch.randperm(n_stacks)[:s_subset_size] for _ in range(batch_size)])

    x = torch.randn(batch_size, s_subset_size, in_features)
    output = layer(x, stack_subset=stack_subset)

    assert output.shape == (batch_size, s_subset_size, out_features)

    # Verify values for a specific batch item
    for b in range(batch_size):
        for s_idx in range(s_subset_size):
            actual_stack = stack_subset[b, s_idx]
            expected = torch.matmul(x[b, s_idx], layer.weight[actual_stack]) + layer.bias[actual_stack]
            assert torch.allclose(output[b, s_idx], expected, atol=1e-5)


def test_stacked_linear_layer_combined_subset():
    n_stacks = 5
    in_features = 10
    out_features = 5
    batch_size = 4

    layer = StackedLinearLayer(n_stacks, in_features, out_features)

    s_subset_size = 3
    stack_subset = torch.stack([torch.randperm(n_stacks)[:s_subset_size] for _ in range(batch_size)])
    output_subset = torch.tensor([0, 2, 4])

    x = torch.randn(batch_size, s_subset_size, in_features)
    output = layer(x, output_subset=output_subset, stack_subset=stack_subset)

    assert output.shape == (batch_size, s_subset_size, len(output_subset))

    # Verify values
    for b in range(batch_size):
        for s_idx in range(s_subset_size):
            actual_stack = stack_subset[b, s_idx]
            expected = (
                torch.matmul(x[b, s_idx], layer.weight[actual_stack, :, output_subset])
                + layer.bias[actual_stack, output_subset]
            )
            assert torch.allclose(output[b, s_idx], expected, atol=1e-5)


def test_initialization():
    n_stacks = 1
    in_features = 100
    out_features = 100

    layer = StackedLinearLayer(n_stacks, in_features, out_features)

    # Initialization should be uniform in [-1/sqrt(in_features), 1/sqrt(in_features)]
    bound = 1 / (in_features**0.5)
    assert layer.weight.min() >= -bound
    assert layer.weight.max() <= bound
    assert layer.bias.min() >= -bound
    assert layer.bias.max() <= bound
