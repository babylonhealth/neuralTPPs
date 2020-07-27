from tpp.utils.stability import epsilon


def non_neg_param(inputs):
    mask = inputs > 0
    outputs = inputs * mask.float()
    return outputs + epsilon(dtype=outputs.dtype, device=outputs.device)
