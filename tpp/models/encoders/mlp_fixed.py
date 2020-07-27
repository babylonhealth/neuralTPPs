from typing import List, Optional

from tpp.models.encoders.base.fixed_history import FixedHistoryEncoder
from tpp.pytorch.models import MLP


class MLPFixedEncoder(FixedHistoryEncoder):
    """MLP network using a fixed history encoder.

    Args
        units_mlp: List of hidden layers sizes.
        activation_mlp: Activation functions. Either a list or a string.
        constraint_mlp: Constraint of the network. Either none, nonneg or
            softplus.
        dropout_mlp: Dropout rates, either a list or a float.
        activation_final_mlp: Last activation of the MLP.
        history_size: The size of each history.
        marks: The distinct number of marks (classes) for the process. Defaults
            to 1.
    """
    def __init__(
            self,
            units_mlp: List[int],
            activation_mlp: Optional[str] = "relu",
            dropout_mlp: Optional[float] = 0.,
            constraint_mlp: Optional[str] = None,
            activation_final_mlp: Optional[str] = None,
            history_size: Optional[int] = 2,
            marks: Optional[int] = 1,
            **kwargs):
        mlp = MLP(
            units=units_mlp,
            activations=activation_mlp,
            constraint=constraint_mlp,
            dropout_rates=dropout_mlp,
            input_shape=history_size,
            activation_final=activation_final_mlp)
        super(MLPFixedEncoder, self).__init__(
            name="mlp-fixed",
            net=mlp,
            output_size=units_mlp[-1],
            history_size=history_size,
            marks=marks,
            **kwargs)
