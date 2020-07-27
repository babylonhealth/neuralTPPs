import torch as th

from typing import Dict, Optional, Tuple

from tpp.models.encoders.base.encoder import Encoder
from tpp.utils.events import Events


class StubEncoder(Encoder):
    """An encoder that does nothing. Used for e.g. the Hawkes decoder that
        needs no encoding.

    Args:
        marks: The distinct number of marks (classes) for the process. Defaults
            to 1.
    """
    def __init__(self, marks: Optional[int] = 1, **kwargs):
        super(StubEncoder, self).__init__(
            name="stub", output_size=0, marks=marks, **kwargs)

    def forward(self, events: Events) -> Tuple[th.Tensor, th.Tensor, Dict]:
        return th.Tensor(), th.Tensor(), dict()
