"""

"""
from .bsm_loss import BlackScholesLoss
from .spread_loss import SpreadLoss

LOSS_FUNCS = {
    "bsm_call": BlackScholesLoss,
    "spread_call": SpreadLoss,
}


def get_loss(func):
    """

    """

    loss_func = LOSS_FUNCS.get(func)
    if loss_func:
        return loss_func()
    else:
        raise NotImplementedError(f"{func} not implemented")
