import torch as th
import matplotlib.pyplot as plt

from torch import nn
from pprint import pprint
from tqdm import tqdm

from tpp.models.base.enc_dec import EncDecProcess
from tpp.models.encoders.mlp_variable import MLPVariableEncoder
from tpp.models.decoders.self_attention_cm import SelfAttentionCmDecoder
from tpp.models.decoders.mlp_cm import MLPCmDecoder

from tpp.utils.events import get_events, get_window

th.manual_seed(0)

times = th.Tensor([1, 2, 6]).float().reshape(1, -1)
query = th.linspace(start=0.0, end=10.1, steps=10).float().reshape(1, -1)
window_start, window_end = get_window(times=times, window=10.)
events = get_events(
    times=times,
    mask=th.ones_like(times),
    window_start=window_start,
    window_end=window_end)

# dec = SelfAttentionCmDecoder(
#     encoding="temporal",
#     units_mlp=[32, 1],
#     constraint_mlp="nonneg",
#     activation_final_mlp="softplus",
#     attn_activation="sigmoid")
dec = MLPCmDecoder(
    encoding="times_only",
    units_mlp=[32, 1],
    constraint_mlp="nonneg",
    activation_final_mlp="softplus",
    attn_activation="sigmoid")
enc = MLPVariableEncoder(units_mlp=[dec.input_size], encoding="marks_only")
process = EncDecProcess(encoder=enc, decoder=dec)

(_, _, _, artifacts) = process.artifacts(query=query, events=events)
intensity_integrals = artifacts["decoder"]["intensity_integrals"]
#
# optimiser = th.optim.Adam(params=process.parameters())
# for _ in tqdm(range(1000)):
#     optimiser.zero_grad()
#     nll, nll_mask, _ = process.neg_log_likelihood(events=events)
#     nll = nll * nll_mask
#     nll = sum(nll) / sum(nll_mask)
#     nll.backward()
#     optimiser.step()
#     print(float(nll.detach().cpu().numpy()))
# dict(process.named_parameters())

intensity_integrals = artifacts["decoder"]["intensity_integrals"]
intensity_mask = artifacts["decoder"]["intensity_mask"]

x = query.detach().cpu().numpy()
x = x[intensity_mask != 0]
y = intensity_integrals.detach().cpu().numpy()
y = y[intensity_mask != 0]
plt.figure()
plt.plot(x.reshape(-1), y.reshape(-1))
for t in times.reshape(-1):
    plt.axvline(x=t, color="red")
plt.title("cumulative intensity")
plt.show()

intensity, intensity_mask = process.intensity(query=query, events=events)
x = query.detach().cpu().numpy()
x = x[intensity_mask != 0]
y = intensity.detach().cpu().numpy()
y = y[intensity_mask != 0]
plt.plot(x.reshape(-1), y.reshape(-1))
for t in times.reshape(-1):
    plt.axvline(x=t, color="red")
plt.title("intensity")
plt.show()
