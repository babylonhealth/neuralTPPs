import numpy as np
import torch as th

import torch.nn
import matplotlib.pyplot as plt

from tpp.pytorch.models import MLP


def detach(x: th.Tensor) -> np.ndarray:
    return x.detach().cpu().numpy()


th.manual_seed(0)

x_min, x_max, steps = 0., 100., 3000
alpha = 1.
beta = 1.
n_events = 20
epochs = 1000
cumulative = True
units = [32, 32, 32, 1]

x_train = th.linspace(x_min, x_max, steps=steps)

events = th.randperm(steps)[:n_events]
events = x_train[events]
events = th.sort(events).values

x_train = x_train[x_train > th.min(events)]

y_train = th.zeros_like(x_train)
for e in events:
    delta_t = x_train - e
    intensity_delta = alpha * th.exp(-beta * delta_t)
    intensity_delta[delta_t <= 0] = th.zeros_like(intensity_delta[
                                                      delta_t <= 0])
    y_train = y_train + intensity_delta

prev_times = th.zeros_like(x_train)
for i, t in enumerate(x_train):
    prev_time = t - events
    prev_time[prev_time < 0] = x_max + 1.
    prev_time = th.argmin(prev_time)
    prev_time = events[prev_time]
    prev_times[i] = prev_time

tau = x_train - prev_times

activations, constraint, activation_final = "relu", None, None
if cumulative:
    (activations, constraint,
     activation_final) = "gumbel", "nonneg", "parametric_softplus"

mlp = MLP(
    units=units,
    input_shape=1,
    activations=activations,
    constraint=constraint,
    activation_final=activation_final)

optimiser = th.optim.Adam(params=mlp.parameters(), lr=1.e-3)

mse = torch.nn.MSELoss()
tau_r = tau.reshape(-1, 1)
y_train_r = y_train.reshape(-1, 1)

if cumulative:
    tau_r.requires_grad = True

for i in range(epochs):
    optimiser.zero_grad()
    y_pred = mlp(tau_r)
    if cumulative:
        y_pred = th.autograd.grad(
            y_pred, tau_r,
            grad_outputs=th.ones_like(y_pred),
            retain_graph=True,
            create_graph=True)[0]
    loss = mse(y_train_r, y_pred)
    loss = th.sum(loss)
    loss.backward()
    optimiser.step()
    print("epoch: {} loss: {}".format(i, float(loss)))

plt.figure(figsize=(18, 4))
plt.plot(x_train, y_train, label="true")
plt.plot(x_train, detach(y_pred), label="pred")
plt.xlabel("time")
plt.ylabel("intensity")
plt.legend()
plt.show()
