import torch
import torch.nn as nn

loss_names = ['l1', 'l2']


class MaskedMSELoss(nn.Module):
    def __init__(self):
        super(MaskedMSELoss, self).__init__()

    def forward(self, pred, target, mask = None):
        assert pred.dim() == target.dim(), "inconsistent dimensions"

        if mask is not None:
            # Ensure mask is the same size as pred and target
            assert mask.size() == target.size(), "Mask must be the same size as target"

            # Apply mask
            diff = pred - target
            masked_diff = diff * mask  # Apply mask by element-wise multiplication
            loss = (masked_diff ** 2).mean()
        else:
            # Calculate MSE loss normally if no mask is provided
            loss = ((pred - target) ** 2).mean()

        return loss

class MaskedL1Loss(nn.Module):
    def __init__(self):
        super(MaskedL1Loss, self).__init__()

    def forward(self, pred, target, mask = None):
        assert pred.dim() == target.dim(), "inconsistent dimensions"

        # Calculate L1 loss
        loss = torch.abs(pred - target).mean()
        return loss


class SmoothnessLoss(nn.Module):
    def __init__(self):
        super(SmoothnessLoss, self).__init__()

    def forward(self, depth):
        def second_derivative(x):
            assert x.dim(
            ) == 4, "expected 4-dimensional data, but instead got {}".format(
                x.dim())
            horizontal = 2 * x[:, :, 1:-1, 1:-1] - x[:, :, 1:-1, :
                                                     -2] - x[:, :, 1:-1, 2:]
            vertical = 2 * x[:, :, 1:-1, 1:-1] - x[:, :, :-2, 1:
                                                   -1] - x[:, :, 2:, 1:-1]
            der_2nd = horizontal.abs() + vertical.abs()
            return der_2nd.mean()

        self.loss = second_derivative(depth)
        return self.loss
