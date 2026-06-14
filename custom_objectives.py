"""
Custom XGBoost objective functions for cost-sensitive medical diagnosis.

In diabetes screening a *false negative* (telling a diabetic they are healthy) is far
more dangerous than a *false positive* (sending a healthy person for a confirmatory blood
test). Standard XGBoost minimizes binary log-loss, whose gradient/hessian penalize both
error types symmetrically. The objectives below reshape that gradient so the model pays a
much larger price for missing a positive (diabetic) case.

XGBoost is gradient-agnostic: it never needs the loss value itself, only the per-sample
first derivative (gradient) and second derivative (hessian) of the loss with respect to
the raw margin score `z`. Hand it those two arrays and it builds trees against *our* loss.

All factories return a callable matching the XGBoost scikit-learn custom-objective
signature:

    obj(y_true, y_pred) -> (grad, hess)

where `y_pred` is the raw margin (NOT a probability — XGBoost has not applied the sigmoid
yet). Throughout: p = sigmoid(z), y in {0, 1}. Every hessian is floored at HESS_FLOOR so a
tree split never sees a zero/negative second derivative (which would destabilize training).
"""

import numpy as np

# Floor for the hessian. XGBoost divides by the summed hessian when computing leaf weights;
# a value of exactly 0 (or negative) produces NaN/unstable splits, so we clamp.
HESS_FLOOR = 1e-6

# Clip probabilities away from {0, 1} so log() and 1/p never overflow.
EPS = 1e-6


def sigmoid(z):
    """Numerically stable logistic sigmoid. Works on scalars or numpy arrays."""
    z = np.asarray(z, dtype=np.float64)
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


# The objectives are implemented as small *callable classes* rather than closures so that a
# fitted XGBClassifier carrying one of them can be pickled (joblib) and reloaded for serving
# — closures are not picklable. The `*_obj()` factory functions preserve a clean call style.


# ---------------------------------------------------------------------------
# 1. Weighted cross-entropy  (constant false-negative weight w > 1)
# ---------------------------------------------------------------------------
class WeightedLogLoss:
    """
    Asymmetric log-loss: positive (diabetic) samples carry weight `w`, negatives weight 1.

    Derivation (per sample): standard log-loss has grad = p - y, hess = p(1-p). Scaling a
    sample's loss by a constant `weight` scales both derivatives by that same constant:

        weight = w if y == 1 else 1
        grad   = weight * (p - y)
        hess   = weight * p * (1 - p)

    A larger `w` tilts every split toward correctly classifying positives, directly trading
    precision for recall. `w` is typically seeded at N_neg / N_pos (the class-imbalance
    ratio) and increased to be more aggressive.
    """

    def __init__(self, w):
        self.w = w

    def __call__(self, y_true, y_pred):
        y = np.asarray(y_true, dtype=np.float64)
        p = sigmoid(y_pred)
        weight = np.where(y == 1.0, self.w, 1.0)
        grad = weight * (p - y)
        hess = np.maximum(weight * p * (1.0 - p), HESS_FLOOR)
        return grad, hess


def weighted_logloss_obj(w):
    return WeightedLogLoss(w)


# ---------------------------------------------------------------------------
# 2. Focal loss  (alpha class-balance + gamma hard-example focusing)
# ---------------------------------------------------------------------------
def _robust_pow(base, power):
    """Sign-preserving power, so a (possibly negative) base raised to a fractional power
    stays real-valued and keeps its sign."""
    return np.sign(base) * np.abs(base) ** power


class FocalLoss:
    """
    Binary focal loss (Lin et al., 2017, "Focal Loss for Dense Object Detection").

        FL = -alpha_t * (1 - p_t)^gamma * log(p_t)
        p_t     = p     if y == 1 else 1 - p
        alpha_t = alpha if y == 1 else 1 - alpha

    The modulating factor (1 - p_t)^gamma shrinks the loss of easy, already-correct examples
    and lets the *hard* ones dominate the gradient. For a confidently-missed positive (a
    false negative: y=1 but p -> 0) the factor approaches 1, so that sample keeps a large
    gradient and the trees keep working on it — exactly the medical behavior we want.
    `alpha` > 0.5 additionally up-weights the positive class.

    The closed-form margin-space gradient/hessian below follow the standard XGBoost focal
    derivation; `_robust_pow` preserves the sign when raising a possibly-negative base to a
    fractional power. Hessian floored positive.
    """

    def __init__(self, gamma=2.0, alpha=0.75):
        self.gamma = gamma
        self.alpha = alpha

    def __call__(self, y_true, y_pred):
        gamma = self.gamma
        y = np.asarray(y_true, dtype=np.float64)
        p = np.clip(sigmoid(y_pred), EPS, 1.0 - EPS)

        # alpha_t weighting per sample.
        at = np.where(y == 1.0, self.alpha, 1.0 - self.alpha)

        g1 = p * (1.0 - p)                       # dp/dz
        g2 = y + ((-1.0) ** y) * p               # = p_t complement helper
        g3 = p + y - 1.0
        g4 = 1.0 - y - ((-1.0) ** y) * p         # = p_t
        g5 = y + ((-1.0) ** y) * p

        grad = gamma * g3 * _robust_pow(g2, gamma) * np.log(g4 + 1e-9) \
            + ((-1.0) ** y) * _robust_pow(g5, gamma + 1.0)

        hess_1 = _robust_pow(g2, gamma) \
            + gamma * ((-1.0) ** y) * g3 * _robust_pow(g2, gamma - 1.0)
        hess_2 = ((-1.0) ** y) * g3 * _robust_pow(g2, gamma) / g4
        hess = ((hess_1 * np.log(g4 + 1e-9) - hess_2) * gamma
                + (gamma + 1.0) * _robust_pow(g5, gamma)) * g1

        grad = at * grad
        hess = np.maximum(at * hess, HESS_FLOOR)
        return grad, hess


def focal_loss_obj(gamma=2.0, alpha=0.75):
    return FocalLoss(gamma, alpha)


# ---------------------------------------------------------------------------
# 3. Custom exponential false-negative penalty
# ---------------------------------------------------------------------------
class ExponentialFN:
    """
    A bespoke loss that *exponentially* penalizes false negatives, the most literal reading
    of the design brief.

        positive (y=1):  L = exp(gamma * (1 - p)) * (-log p)
        negative (y=0):  L = -log(1 - p)            (ordinary log-loss)

    For a positive whose predicted probability collapses toward 0 (a confident false
    negative), the multiplier exp(gamma*(1-p)) -> exp(gamma): the loss — and therefore the
    corrective gradient — is scaled up exponentially in `gamma`. Negatives are left on the
    standard log-loss so we do not gratuitously inflate false positives.

    Gradient (positive class), in margin space (p = sigmoid(z), dp/dz = p(1-p)):

        Let A = exp(gamma*(1-p)).
        dL/dz = -A * (1 - p) * (1 - gamma * p * log p)

    As p -> 0 this tends to -exp(gamma): a strong but *bounded* upward push on the margin
    (p*log p -> 0), so training does not diverge. The hessian uses a Gauss-Newton positive
    approximation hess ~= A * p(1-p): it drops the unstable higher-order curvature terms but
    is guaranteed >= 0, which is what keeps the boosting iterations stable. Floored positive.
    """

    def __init__(self, gamma=2.0):
        self.gamma = gamma

    def __call__(self, y_true, y_pred):
        gamma = self.gamma
        y = np.asarray(y_true, dtype=np.float64)
        p = np.clip(sigmoid(y_pred), EPS, 1.0 - EPS)

        # exp(gamma*(1-p)); exponent is in [0, gamma] since (1-p) in (0,1), so it cannot blow up.
        A = np.exp(gamma * (1.0 - p))

        # Positive-class gradient/hessian (exponentially weighted log-loss).
        grad_pos = -A * (1.0 - p) * (1.0 - gamma * p * np.log(p))
        hess_pos = A * p * (1.0 - p)

        # Negative-class gradient/hessian (standard log-loss): grad = p, hess = p(1-p).
        grad_neg = p
        hess_neg = p * (1.0 - p)

        is_pos = y == 1.0
        grad = np.where(is_pos, grad_pos, grad_neg)
        hess = np.maximum(np.where(is_pos, hess_pos, hess_neg), HESS_FLOOR)
        return grad, hess


def exponential_fn_obj(gamma=2.0):
    return ExponentialFN(gamma)
