import torch
from torch import nn
from torch.nn import functional as F
    
    
class DiceFocalWithLogitsLoss(nn.Module):
    def __init__(self, gamma=2, alpha=0.25, reduction="mean", eps=1e-6):
        super().__init__()    
        self.critD = FocalWithLogitsLoss(gamma=gamma, alpha=alpha, reduction=reduction)
        self.critF = DiceWithLogitsLoss(reduction=reduction, eps=eps)
    def forward(self, inputs, targets):
        return self.critF(inputs, targets) + self.critD(inputs, targets) / 10    
    
    
class DiceWithLogitsLoss(nn.Module):
    """Computes the Sørensen–Dice loss with logits.

        DC = 2 * intersection(X, Y) / (|X| + |Y|)
    where, X and Y are sets of binary data, in this case, predictions and targets.
    |X| and |Y| are the cardinalities of the corresponding sets. Probabilities are
    computed using softmax.

    The optimizer minimizes the loss function therefore:
        DL = -DC (min(-x) = max(x))
    To make the loss positive (convenience) and because the coefficient is within
    [0, -1], subtract 1.
        DL = 1 - DC

    See: https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient

    Arguments:
        reduction (string, optional): Specifies the reduction to apply to the output:
            'none' | 'mean' | 'sum'. 'none': no reduction will be applied,
            'mean': the sum of the output will be divided by the number of
            elements in the output, 'sum': the output will be summed. Default: 'mean'
        eps (float, optional): small value to avoid division by zero. Default: 1e-6.

    """

    def __init__(self, reduction="mean", eps=1e-6):
        super().__init__()
        self.eps = eps
        if reduction.lower() == "none":
            self.reduction_op = None
        elif reduction.lower() == "mean":
            self.reduction_op = torch.mean
        elif reduction.lower() == "sum":
            self.reduction_op = torch.sum
        else:
            raise ValueError(
                "expected one of ('none', 'mean', 'sum'), got {}".format(reduction)
            )

    def forward(self, inputs, targets):
        if inputs.dim() != 2 and inputs.dim() != 4:
            raise ValueError(
                "expected input of size 4 or 2, got {}".format(inputs.dim())
            )

        if targets.dim() != 1 and targets.dim() != 3:
            raise ValueError(
                "expected target of size 3 or 1, got {}".format(targets.dim())
            )

        if inputs.dim() == 4 and targets.dim() == 3:
            reduce_dims = (0, 3, 2)
        elif inputs.dim() == 2 and targets.dim() == 1:
            reduce_dims = 0
        else:
            raise ValueError(
                "expected target dimension {} for input dimension {}, got {}".format(
                    inputs.dim() - 1, inputs.dim(), targets.dim()
                )
            )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        target_onehot = torch.zeros_like(inputs,dtype=torch.float, device=device).scatter_(1, targets.unsqueeze(1).long(), 1)
        probabilities = nn.functional.softmax(inputs, 1)

        # Dice = 2 * intersection(X, Y) / (|X| + |Y|)
        # X and Y are sets of binary data, in this case, probabilities and targets
        # |X| and |Y| are the cardinalities of the corresponding sets
        num = torch.sum(target_onehot * probabilities, dim=reduce_dims)
        den_t = torch.sum(target_onehot, dim=reduce_dims)
        den_p = torch.sum(probabilities, dim=reduce_dims)
        loss = 1 - (2 * (num / (den_t + den_p + self.eps)))

        if self.reduction_op is not None:
            return self.reduction_op(loss)
        else:
            return loss     


class FocalWithLogitsLoss(nn.Module):
    """Computes the focal loss with logits.

    The Focal Loss is designed to address the one-stage object detection scenario in
    which there is an extreme imbalance between foreground and background classes during
    training (e.g., 1:1000). Focal loss is defined as:

        FL = alpha(1 - p)^gamma * CE(p, y)
    where p are the probabilities, after applying the softmax layer to the logits,
    alpha is a balancing parameter, gamma is the focusing parameter, and CE(p, y) is the
    cross entropy loss. When gamma=0 and alpha=1 the focal loss equals cross entropy.

    See: https://arxiv.org/abs/1708.02002

    Arguments:
        gamma (float, optional): focusing parameter. Default: 2.
        alpha (float, optional): balancing parameter. Default: 0.25.
        reduction (string, optional): Specifies the reduction to apply to the output:
            'none' | 'mean' | 'sum'. 'none': no reduction will be applied,
            'mean': the sum of the output will be divided by the number of
            elements in the output, 'sum': the output will be summed. Default: 'mean'
        eps (float, optional): small value to avoid division by zero. Default: 1e-6.

    """

    def __init__(self, gamma=2, alpha=0.25, reduction="mean"):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        if reduction.lower() == "none":
            self.reduction_op = None
        elif reduction.lower() == "mean":
            self.reduction_op = torch.mean
        elif reduction.lower() == "sum":
            self.reduction_op = torch.sum
        else:
            raise ValueError(
                "expected one of ('none', 'mean', 'sum'), got {}".format(reduction)
            )

    def forward(self, inputs, target):
        if inputs.dim() == 4:
            inputs = inputs.permute(0, 2, 3, 1)
            inputs = inputs.contiguous().view(-1, inputs.size(-1))
        elif inputs.dim() != 2:
            raise ValueError(
                "expected input of size 4 or 2, got {}".format(inputs.dim())
            )

        if target.dim() == 3:
            target = target.contiguous().view(-1)
        elif target.dim() != 1:
            raise ValueError(
                "expected target of size 3 or 1, got {}".format(target.dim())
            )

        if target.dim() != inputs.dim() - 1:
            raise ValueError(
                "expected target dimension {} for input dimension {}, got {}".format(
                    inputs.dim() - 1, inputs.dim(), target.dim()
                )
            )
            
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ttarget = torch.zeros_like(inputs,dtype=torch.float, device=device).scatter_(1, target.unsqueeze(1).long(), 1)
        probabilities = F.softmax(torch.sum(inputs * ttarget, dim=(1)), dim=0)
        focal = self.alpha * (1 - probabilities).pow(self.gamma)
        ce = nn.functional.cross_entropy(inputs, target, reduction="none")
        loss = focal * ce

        if self.reduction_op is not None:
            return self.reduction_op(loss)
        else:
            return loss       