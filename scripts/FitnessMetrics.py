import numpy as np

def monotonicity(x, rounding_precision = 3):
    """Calculates monotonicity metric of a value of[0-1] for a given array.\nFor an array of length n, monotonicity is calculated as follows:\nmonotonicity=abs[(num. positive gradients)/(n-1)-(num. negative gradients)/(n-1)]."""

    n = x.shape[0]
    grad = np.gradient(x)
    pos_grad = np.sum(grad>0)
    neg_grad = np.sum(grad<0)
    monotonicity = np.abs( pos_grad/(n-1) - neg_grad/(n-1) )
    return np.round(monotonicity, rounding_precision)

def trendability(x):
    """Calculate trendability metric of a value of [0-1] for a given array of features.""" 

    t_array = np.array([])

    for feature in x:
        n = feature.shape[0]
        grad_first = np.gradient(feature)
        grad_second = np.gradient(grad_first)
        pos_grad_first = np.sum(grad_first>0)
        pos_grad_second = np.sum(grad_second>0)
        t = (pos_grad_first)/(n-1) + (pos_grad_second)/(n-2)
        t_array = np.append(t_array, t)

    trendability = 1-np.std(t_array)
    return trendability
