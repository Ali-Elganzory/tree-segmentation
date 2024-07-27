import torch


class FocalTverskyLoss(torch.nn.Module):
    """
    Focal Tversky Loss.

    Args:
        gamma (float): Focal parameter.
    """

    def __init__(self, alpha: float = 0.7, beta: float = 0.3, gamma: float = 0.75):
        super(FocalTverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor):
        """
        Forward pass.

        Args:
            y_pred (torch.Tensor): Predictions. Shape (B, C, H, W). B: Batch size, C: Number of classes, H: Height, W: Width.
            y_true (torch.Tensor): Ground truth. Shape (B, H, W). B: Batch size, H: Height, W: Width.

        Returns:
            torch.Tensor: Loss.
        """
        num_classes = y_pred.shape[1]

        # Get predictions
        y_pred = y_pred.argmax(dim=1)

        # One-hot encode
        y_true = torch.nn.functional.one_hot(y_true, num_classes).permute(0, 3, 1, 2)
        y_pred = torch.nn.functional.one_hot(y_pred, num_classes).permute(0, 3, 1, 2)

        # Calculate True Positives, False Positives, False Negatives
        tp = torch.sum(y_pred * y_true, dim=(1, 2, 3))
        fp = torch.sum(y_pred * (1 - y_true), dim=(1, 2, 3))
        fn = torch.sum((1 - y_pred) * y_true, dim=(1, 2, 3))

        tversky = (tp + 1e-6) / (tp + self.alpha * fp + self.beta * fn + 1e-6)

        # Calculate Focal Tversky
        focal_tversky = (1 - tversky) ** (1 / self.gamma)

        # Calculate loss
        loss = torch.sum(focal_tversky, dim=1, requires_grad=True)

        return loss.mean()
