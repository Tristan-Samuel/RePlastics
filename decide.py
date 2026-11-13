"""Hold rule for the two-class belt classifier.

``uncertain`` is not a trained class. It means the winning score, after
the frozen temperature, is below the frozen threshold. The stage 1
``OTHERS`` resin tag is a different thing: those photos were trained
as metal.
"""

from pathlib import Path

import torch


UNCERTAIN = "uncertain"

# Temperature was fit on data_belt_stage1 validation by
# confidence_report.py. It minimizes unweighted NLL (reported as
# 1.011839). The same run's measured hold was 0.97: the lowest cutoff
# from 0.50 to 0.99 that catches the most validation errors while
# holding at most 5% of that split. The locked test split was scored
# once after that measurement.
#
# The live cutoff is 0.85. This checkpoint is a public showcase, so a
# few confident mistakes are sorted instead of parking the frame.
# On validation, 0.85 holds 14 of 1,251 images (1.12%) and leaves
# three errors sorted: 90.88% PP, 96.11% HDPE, and 99.06% HDPE.
# The softer misses, 76.21% and 84.61%, stay held. 0.97 held 39.
#
# Those two numbers were fit on the 5 Oct belt checkpoint. The same
# cutoff is used for every checkpoint: it is applied after the softmax,
# so it does not depend on which file produced the logits.
TEMPERATURE = 1.0118387937545776
THRESHOLD = 0.85


def frozen_rule_applies(model_path):
    """True for every checkpoint. The hold is not tied to one filename."""
    return Path(model_path).suffix == ".pt"


def scoring_temperature():
    """Positive temperature. Unset uses the raw softmax."""
    if TEMPERATURE is None:
        return 1.0
    temperature = float(TEMPERATURE)
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    return temperature


def scoring_threshold(override):
    """Threshold for this run. An explicit override wins."""
    if override is not None:
        return float(override)
    if THRESHOLD is None:
        return None
    return float(THRESHOLD)


def scaled_probabilities(logits, temperature=None):
    """Softmax of logits / temperature. None uses the frozen value."""
    if temperature is None:
        temperature = scoring_temperature()
    else:
        temperature = float(temperature)
        if temperature <= 0:
            raise ValueError("temperature must be positive")
    return torch.softmax(logits / temperature, dim=-1)


def decide(probabilities, threshold, idx_to_class):
    """Return ``(label, confidence)`` for one image.

    ``label`` is ``uncertain`` when the winning score is below
    ``threshold``. A missing threshold keeps the winning class.
    """
    if not torch.is_tensor(probabilities):
        probabilities = torch.tensor(probabilities, dtype=torch.float32)
    if probabilities.ndim != 1:
        raise ValueError(
            "decide() expects one image, a 1-D probability vector"
        )
    confidence_tensor, index_tensor = torch.max(probabilities, dim=-1)
    index = int(index_tensor)
    confidence = float(confidence_tensor)
    label = idx_to_class[index]
    if threshold is not None and confidence < float(threshold):
        return UNCERTAIN, confidence
    return label, confidence
