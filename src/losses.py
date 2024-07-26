import torch


class FocalTverskyLoss(torch.nn.Module):
    """
    Focal Tversky Loss.

    Args:
        gamma (float): Focal parameter.
    """

    def __init__(self, gamma=0.75):
        super(FocalTverskyLoss, self).__init__()
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
        y_pred = y_pred.argmax(dim=1).squeeze(1)

        # One-hot encode
        y_true = (
            torch.nn.functional.one_hot(y_true, num_classes=y_pred.shape[1])
            .permute(0, 3, 1, 2)
            .float()
        )
        y_pred = (
            torch.nn.functional.one_hot(y_pred, num_classes=y_pred.shape[1])
            .permute(0, 3, 1, 2)
            .float()
        )

        print(y_pred.shape, y_true.shape)
        exit()

        return focal_tversky.mean()
