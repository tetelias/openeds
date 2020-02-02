import torch


def dice_granular(inputs, targets, eps=1e-8):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inputs = torch.zeros_like(inputs, dtype=torch.long, device=device).scatter_(1, torch.argmax(inputs, dim=1, keepdim=True), 1)
    targets = torch.zeros_like(inputs,dtype=torch.long, device=device).scatter_(1, targets.long(), 1)
    xmod = inputs.sum([0,2,3]).float()
    ymod = targets.sum([0,2,3]).float()
    return torch.div(2 * torch.sum(targets*inputs, dim=(0,2,3)).float()+eps, (xmod+ymod+eps))[1:].mean().item()