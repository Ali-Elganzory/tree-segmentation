import torch

from torch.functional import F


class FocalLoss(torch.nn.Module):
    """
    Focal Loss.

    Args:
        gamma (float): Focal parameter.
    """

    def __init__(self, gamma: float = 2.0, alpha: float = 0.25):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        weight: torch.Tensor = None,
    ):
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
        logits = y_pred
        y_pred = y_pred.sigmoid()

        # One-hot encode ground truth
        y_true = (
            torch.nn.functional.one_hot(y_true, num_classes)
            .permute(0, 3, 1, 2)
            .to(y_pred.dtype)
        )

        # Calculate loss
        ce_loss = F.binary_cross_entropy_with_logits(logits, y_true, reduction="none")
        p_t = y_pred * y_true + (1 - y_pred) * (1 - y_true)
        loss = ce_loss * ((1 - p_t) ** self.gamma)

        if self.alpha >= 0:
            alpha_t = self.alpha * y_true + (1 - self.alpha) * (1 - y_true)
            loss = alpha_t * loss

        return loss.mean()


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
        y_pred = y_pred.softmax(dim=1)

        # One-hot encode ground truth
        y_true = (
            torch.nn.functional.one_hot(y_true, num_classes)
            .permute(0, 3, 1, 2)
            .to(y_pred.dtype)
        )

        # Calculate True Positives, False Positives, False Negatives
        tp = torch.sum(y_pred * y_true, dim=[2, 3])
        fp = torch.sum(y_pred * (1 - y_true), dim=[2, 3])
        fn = torch.sum((1 - y_pred) * y_true, dim=[2, 3])

        tversky = (tp + 1e-6) / (tp + self.alpha * fp + self.beta * fn + 1e-6)

        # Calculate Focal Tversky
        focal_tversky = (1 - tversky) ** (1 / self.gamma)

        # Calculate loss
        loss = torch.sum(focal_tversky, dim=1)

        return loss.mean()
