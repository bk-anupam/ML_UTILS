import torch
import torch.nn as nn

class LossType:
    MSE = "mse"
    BCE_WITH_LOGITS = "bcewithlogits"
    PEARSON = "pearson"

# https://www.kaggle.com/competitions/ubiquant-market-prediction/discussion/302977
class CorrLoss(nn.Module):
    """
    Use 1 - correlational coefficience between the output of the network and the target as the loss
    input (o, t):
        o: Variable of size (batch_size, 1) output of the network
        t: Variable of size (batch_size, 1) target value
    output (corr):
        corr: Variable of size (1)
    """

    def __init__(self):
        super(CorrLoss, self).__init__()

    def forward(self, o: torch.Tensor, t: torch.Tensor):
        assert o.size() == t.size()
        # calculate z-score for o and t
        o_m = o.mean(dim=0)
        o_s = o.std(dim=0)
        o_z = (o - o_m) / (o_s + 1e-7)

        t_m = t.mean(dim=0)
        t_s = t.std(dim=0)
        t_z = (t - t_m) / (t_s + 1e-7)

        # calculate corr between o and t
        tmp = o_z * t_z
        corr = tmp.mean(dim=0)
        return 1 - corr