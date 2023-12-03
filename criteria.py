import torch
import torch.nn as nn

loss_names = ['l1', 'l2']


class MaskedMSELoss(nn.Module):
    def __init__(self):
        super(MaskedMSELoss, self).__init__()

    def forward(self, pred, target, mask):
        assert pred.dim() == target.dim(), "inconsistent dimensions"

        # Ensure mask is binary
        mask = (mask > 0.5).float()

        # Separate the regions based on mask
        masked_pred = pred * mask
        masked_target = target * mask

        non_masked_pred = pred * (1 - mask)
        non_masked_target = target * (1 - mask)

        # Calculate MSE loss separately
        masked_loss = ((masked_pred - masked_target) ** 2).mean()
        non_masked_loss = ((non_masked_pred - non_masked_target) ** 2).mean()

        return masked_loss, non_masked_loss


class MaskedL1Loss(nn.Module):
    def __init__(self):
        super(MaskedL1Loss, self).__init__()

    def forward(self, pred, target, mask, weight=None):
        assert pred.dim() == target.dim(), "inconsistent dimensions"

        # Ensure mask is binary
        mask = (mask > 0.5).float()

        # Separate the regions based on mask
        masked_pred = pred * mask
        masked_target = target * mask

        non_masked_pred = pred * (1 - mask)
        non_masked_target = target * (1 - mask)

        # Calculate L1 loss separately
        masked_loss = (torch.abs(masked_pred - masked_target)).mean()
        non_masked_loss = (torch.abs(non_masked_pred - non_masked_target)).mean()

        return masked_loss, non_masked_loss


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
