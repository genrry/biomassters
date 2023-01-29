from torch.nn import MSELoss
from torch import sqrt

class RMSELoss(MSELoss):
    
    def forward(self, input, target):
        mse_loss = super().forward(input, target)
        return sqrt(mse_loss)