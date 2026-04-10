from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class LinearLayer(nn.Linear):
    """Linear layer with support for output weight subsetting.

    This layer behaves like a normal nn.Linear but adds the ability to
    perform the forward pass on a subset of the output features.
    """

    def forward(self, x: torch.Tensor, output_subset: torch.Tensor | None = None) -> torch.Tensor:
        """Forward pass with optional output subsetting.

        Parameters
        ----------
        x
            Input tensor with shape (..., in_features).
        output_subset
            Indices of the output features to compute. If None, all features
            are computed.

        Returns
        -------
        torch.Tensor
            Output tensor with shape (..., out_features) or (..., len(output_subset)).

        Examples
        --------
        >>> import torch
        >>> layer = LinearLayer(10, 5)
        >>> x = torch.randn(2, 10)
        >>> # Standard forward pass
        >>> out = layer(x)
        >>> out.shape
        torch.Size([2, 5])
        >>> # Subset forward pass
        >>> subset = torch.tensor([0, 2])
        >>> out_subset = layer(x, output_subset=subset)
        >>> out_subset.shape
        torch.Size([2, 2])
        """
        if output_subset is None:
            # x: (..., i) -> output: (..., o)
            return super().forward(x)
        elif output_subset.dim() == 1:
            # x: (..., i) -> output_subset: (o_subset)
            bias = self.bias[output_subset] if self.bias is not None else None  # (o_subset)
            weight = self.weight[output_subset]  # (o_subset, i)
            return F.linear(x, weight, bias)  # (..., i) -> (..., o_subset)
        else:
            raise NotImplementedError()
