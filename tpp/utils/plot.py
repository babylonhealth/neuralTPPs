import json
import mlflow
import os

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch as th

from argparse import Namespace
from matplotlib.figure import Figure
from pathlib import Path
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Tuple

from tpp.models.base.process import Process
from tpp.models.hawkes import HawkesProcess
from tpp.utils.data import Dataset, get_loader
from tpp.utils.events import get_events, get_window
from tpp.utils.mlflow import get_epoch_str

sns.set(style="white")
plt.rc('text', usetex=True)
plt.rc('font', family='serif')


def get_test_loader(args: Namespace, seed: int) -> DataLoader:
    test_seed = seed
    dataset = Dataset(args=args, size=1, seed=test_seed, name="test")
    loader = get_loader(dataset, args=args, shuffle=False)
    return loader


def fig_hawkes(
        intensities: Dict[str, Tuple[np.ndarray, np.ndarray]],
        cumulative_intensities: Dict[str, Tuple[np.ndarray, np.ndarray]],
        event_times: np.ndarray,
        event_labels: np.ndarray,
        class_names: Dict,
        legend: Optional[bool] = True,
        epoch: Optional[int] = None) -> Tuple[Figure, np.ndarray]:
    """Plot a 1D Hawkes process.py for multiple models.
    Args:
        intensities: A dictionary of time, intensity pairs, each of which is a
            2D numpy array [T,M]. The key will be used as the model label in the
            legend.
        cumulative_intensities: A dictionary of time, cumulative_intensities
            pairs, each of which is a
            2D numpy array [T,M]. The key will be used as the model label in
            the legend.
        event_times: A 1D numpy array of the times of true events (of which the
            above intensities should be related to).
        event_labels: A 1D numpy array of the labels of true events (of which
            the above intensities should be related to).
        class_names: The names of each class as an order list.
        legend: If `True` adds a legend to the intensity plot. Defaults to
            `True`.
        epoch: Epoch on which the model is evaluated
    Returns:
        The figure and axes.
    """
    marks = next(iter(intensities.items()))[-1][-1].shape[-1]
    class_keys = list(class_names.keys())
    class_names = list(class_names.values())

    if class_names is None:
        class_names = ["$y_{}$".format(m) for m in range(marks)]

    fig, axs = plt.subplots(
        nrows=3 * marks, ncols=1, figsize=(10, 10), sharex=True,
        gridspec_kw={'height_ratios': [4, 4, 1] * marks})

    intensity_axs = [x for i, x in enumerate(axs) if i % 3 == 0]
    cumulative_intensity_axs = [x for i, x in enumerate(axs) if i % 3 == 1]
    event_axs = [x for i, x in enumerate(axs) if i % 3 == 2]

    for model_name, (query_times, model_intensities) in intensities.items():
        for i, class_key in enumerate(class_keys):
            intensity_ax = intensity_axs[i]
            intensity_ax.plot(
                query_times, model_intensities[:, i], label=model_name)
            intensity_ax.set_ylabel(r"$\lambda(t)_{}$".format(i))

    for model_name, (query_times, cum_model_intensities) in cumulative_intensities.items():
        for i, class_key in enumerate(class_keys):
            cum_intensity_ax = cumulative_intensity_axs[i]
            cum_intensity_ax.plot(
                query_times, cum_model_intensities[:, i], label=model_name)
            cum_intensity_ax.set_ylabel(r"$\Lambda(t)_{}$".format(i))

    for (class_key, class_name, event_ax) in zip(
            class_keys, class_names, event_axs):
        is_event_type_key = event_labels == int(class_key)
        times_key = event_times[is_event_type_key]
        event_ax.scatter(times_key, 0.5 * np.ones_like(times_key))
        event_ax.set_yticks([0.5])
        event_ax.set_yticklabels([class_name])

    event_axs[-1].set_xlabel(r"$t$", fontsize=24)
    if legend:
        intensity_axs[0].legend(loc='upper left')
    if epoch is not None:
        fig.suptitle("Epoch: " + str(epoch), fontsize=16)
    return fig, axs


def filter_by_mask(x, mask, mask_pos_value=1.):
    return x[mask == mask_pos_value]


def fig_src_attn_weights(
        attn_weights: th.Tensor,
        event_times: np.ndarray,
        event_labels: np.ndarray,
        idx_times: np.ndarray,
        class_names: List[str],
        epoch: Optional[int] = None):
    """
    Plot inter-events attention coefficients
    Args:
        attn_weights: Attention weights.
        event_times: A 1D numpy array of the times of true events.
        event_labels: A 1D numpy array of the labels of true events.
        idx_times: a 1D numpy array to give the index of each encounter
        class_names: The names of each class as an order list.
        epoch: Epoch on which the model is evaluated.
    Returns:
        fig: The inter-events attention figure.
    """
    n = event_labels.shape[0]
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(n/3, n/3))
    m = attn_weights.cpu().detach().numpy()[:n+1, :n+1]
    idx_times = np.insert(idx_times+1, 0, 0)
    m = m[idx_times][:, idx_times]
    ax.matshow(m, cmap=plt.get_cmap("Blues"))

    event_tick_labels = [str(np.around(i, decimals=1)) for i in event_times]

    x_event_tick_labels = [
        str(i) + "\n" + x + "\n{}".format(class_names[l])
        for i, (x, l) in enumerate(zip(event_tick_labels, event_labels))]
    x_event_tick_labels = ["idx\ntime\nclass"] + x_event_tick_labels

    y_event_tick_labels = [
        str(i) + "," + x + ",{}".format(class_names[l])
        for i, (x, l) in enumerate(zip(event_tick_labels, event_labels))]
    y_event_tick_labels = ["w-"] + y_event_tick_labels

    ax.set_xticks(ticks=np.arange(n + 1))
    ax.set_yticks(ticks=np.arange(n + 1))

    ax.set_xticklabels(x_event_tick_labels, fontsize=7)
    ax.set_yticklabels(y_event_tick_labels, fontsize=7)

    ax.set_ylabel(
        "Attention coefficient for event/query (i)\n(each row rums to one)")
    ax.set_xlabel("Attention coefficient for key (j)")

    for i, row in enumerate(m):
        for j, value in enumerate(row):
            txt = "{:.1f}".format(value)
            color = "white" if value > 0.6 else "black"
            ax.text(j, i, txt, va='center', ha='center',
                    color=color, fontsize=5)
    if epoch is not None:
        fig.suptitle("Epoch: " + str(epoch), fontsize=16)
    return fig


def fig_tgt_attn_weights(
        attn_weights: th.Tensor,
        event_times: np.ndarray,
        event_labels: np.ndarray,
        idx_times: np.ndarray,
        query: np.ndarray,
        class_names: List[str],
        epoch: Optional[int] = None):
    """
    Plot attention coefficients between events and queries
    Args:
        attn_weights: Attention weights.
        event_times: A 1D numpy array of the times of true events.
        event_labels: A 1D numpy array of the labels of true events.
        idx_times: a 1D numpy array to give the index of each encounter
        query: Queries on which attention coefficients are evaluated
        mask: A 1D numpy array of the mask of true events.
        class_names: The names of each class as an order list.
        epoch: Epoch on which the model is evaluated.
    Returns:
        fig: Attention coefficients between events and queries
    """
    n = event_labels.shape[0]
    m = attn_weights.cpu().detach().numpy()
    idx_times = np.insert(idx_times+1, 0, 0)
    m = m[:, idx_times]
    fig, axs = plt.subplots(
        nrows=n + 2, ncols=1, figsize=(10, 10), sharex='all')
    axs[0].plot(query, m[:, 0])
    axs[0].set_ylim(-0.1, 1.1)
    axs[0].set_yticks(ticks=[0.5])
    axs[0].set_yticklabels([-1])
    for i, label in enumerate(event_labels):
        axs[i+1].plot(query, m[:, i+1])
        axs[i+1].set_ylim(-0.1, 1.1)
        axs[i+1].set_yticks(ticks=[0.5])
        axs[i+1].set_yticklabels([class_names[label]])
    axs[-1].scatter(event_times, 0.5 * np.ones_like(event_times))
    axs[-1].set_ylim(-0.1, 1.1)
    axs[-1].set_yticks(list())
    axs[-1].set_xlabel(r"$t$", fontsize=24)
    if epoch is not None:
        fig.suptitle("Epoch: " + str(epoch), fontsize=16)
    return fig


def save_fig(
        fig: Figure,
        name: str,
        args: Namespace,
        dpi: Optional[int] = 300,
        save_on_mlflow: Optional[bool] = True):
    """
    Separate figure to save figures
    Args:
        fig: The figure to be saved
        name: The path and name of the figure
        args: Namespace
        dpi: DPI for fig saving.
        save_on_mlflow: Whether to save the figure on mlflow
    Returns:
        name: The path and name of the saved figure
    """
    mlflow_path = name[:-4]
    name = os.path.join(args.plots_dir, name)
    fig.savefig(fname=name, dpi=dpi, bbox_inches='tight')
    if save_on_mlflow:
        mlflow.log_artifact(name, mlflow_path)
    return name


def plot_attn_weights(
        model_artifacts: Dict,
        event_times: np.ndarray,
        event_labels: np.ndarray,
        idx_times: np.ndarray,
        query: np.ndarray,
        args: Namespace,
        class_names: List[str],
        epoch: Optional[int] = None,
        images_urls: Optional[Dict] = None,
        save_on_mlflow: Optional[bool] = True):
    """

    Args:
        model_artifacts: Dict
        event_times:
        event_labels:
        idx_times: a 1D numpy array to give the index of each encounter
        query: Queries
        args: Namespace
        class_names: The names of each class as an order list. If `None`, then
                     classes y_0, y_1, ... are used. Default to `None`.
        epoch: Epoch
        images_urls: urls of figures
        save_on_mlflow: Whether to save the figure on mlflow
    """
    if epoch is not None:
        epoch_str = get_epoch_str(epoch=epoch, max_epochs=args.train_epochs)
        src_name = "epoch_" + epoch_str + ".jpg"
        tgt_name = "epoch_" + epoch_str + ".jpg"
    else:
        src_name = "fig.jpg"
        tgt_name = "fig.jpg"
    for k, v in model_artifacts.items():
        for m_k, m_v in v.items():
            if "encoder" in v.keys() \
                    and "attention_weights" in v["encoder"].keys():
                m_v = v
            if "encoder" in m_v.keys() \
                    and "attention_weights" in m_v["encoder"].keys():
                if len(m_v["encoder"]["attention_weights"]) > 1:
                    for i in range(len(m_v["encoder"]["attention_weights"])):
                        fig = fig_src_attn_weights(
                            attn_weights=
                            m_v["encoder"]["attention_weights"][i][0],
                            event_times=event_times,
                            event_labels=event_labels,
                            idx_times=idx_times,
                            class_names=class_names,
                            epoch=epoch)
                        name = os.path.join(
                            "src_attn", k, "head_" + str(i), src_name)
                        Path(os.path.join(
                            args.plots_dir,
                            "src_attn", k, "head_" + str(i))).mkdir(
                            parents=True, exist_ok=True)
                        url = save_fig(fig=fig, name=name, args=args,
                                       save_on_mlflow=save_on_mlflow)
                else:
                    fig = fig_src_attn_weights(
                        attn_weights=
                        m_v["encoder"]["attention_weights"][0][0],
                        event_times=event_times,
                        event_labels=event_labels,
                        idx_times=idx_times,
                        class_names=class_names,
                        epoch=epoch)
                    name = os.path.join("src_attn", k, src_name)
                    Path(os.path.join(args.plots_dir, "src_attn", k)).mkdir(
                        parents=True, exist_ok=True)
                    url = save_fig(fig=fig, name=name, args=args,
                                   save_on_mlflow=save_on_mlflow)
                if images_urls is not None:
                    images_urls['src_attn'].append(url)
            if "decoder" in m_v.keys() \
                    and "attention_weights" in m_v["decoder"].keys():
                if len(m_v["decoder"]["attention_weights"]) > 1:
                    for i in range(len(m_v["decoder"]["attention_weights"])):
                        fig = fig_tgt_attn_weights(
                            attn_weights=
                            m_v["decoder"]["attention_weights"][i][0],
                            event_times=event_times,
                            event_labels=event_labels,
                            idx_times=idx_times,
                            query=query,
                            class_names=class_names,
                            epoch=epoch)
                        name = os.path.join(
                            "tgt_attn", k, "head_" + str(i), tgt_name)
                        Path(os.path.join(
                            args.plots_dir,
                            "tgt_attn", k, "head_" + str(i))).mkdir(
                            parents=True, exist_ok=True)
                        url = save_fig(fig=fig, name=name, args=args,
                                       save_on_mlflow=save_on_mlflow)
                else:
                    fig = fig_tgt_attn_weights(
                        attn_weights=
                        m_v["decoder"]["attention_weights"][0][0],
                        event_times=event_times,
                        event_labels=event_labels,
                        idx_times=idx_times,
                        query=query,
                        class_names=class_names,
                        epoch=epoch)
                    name = os.path.join("tgt_attn", k, tgt_name)
                    Path(os.path.join(args.plots_dir, "tgt_attn", k)).mkdir(
                        parents=True, exist_ok=True)
                    url = save_fig(fig=fig, name=name, args=args,
                                   save_on_mlflow=save_on_mlflow)
                if images_urls is not None:
                    images_urls['tgt_attn'].append(url)

        return images_urls


def log_figures(
        model: Process,
        test_loader: DataLoader,
        args: Namespace,
        epoch: Optional[int] = None,
        images_urls: Optional[dict] = None,
        save_on_mlflow: Optional[bool] = True):
    models = dict()

    models[model.name.replace("_", "-")] = model
    if args.load_from_dir in [None, "hawkes"]:
        true_model = HawkesProcess(marks=args.marks)

        true_model.alpha.data = th.tensor(args.alpha)
        true_model.beta.data = th.tensor(args.beta)
        true_model.mu.data = th.tensor(args.mu)
        true_model.to(args.device)

        models["ground truth"] = true_model

    batch = next(iter(test_loader))
    times, labels = batch["times"], batch["labels"]
    times, labels = times.to(args.device), labels.to(args.device)
    length = (times != args.padding_id).sum(-1)
    i = th.argmax(length)
    times = times[i][:20].reshape(1, -1)
    labels = labels[i][:20].reshape(1, -1, args.marks)
    mask = (times != args.padding_id).type(times.dtype)
    times = times * args.time_scale

    window_start, window_end = get_window(times=times, window=args.window)
    events = get_events(
        times=times, mask=mask, labels=labels,
        window_start=window_start, window_end=window_end)

    if args.window is not None:
        query = th.linspace(
            start=0.001, end=args.window, steps=500)
    else:
        query = th.linspace(
            start=0.001, end=float(events.window_end[0]), steps=500)
    query = query.reshape(1, -1)
    query = query.to(device=args.device)

    event_times = events.times.cpu().detach().numpy().reshape(-1)
    event_labels = events.labels.cpu().detach().numpy().reshape(
        event_times.shape[0], -1)
    idx_times = np.where(event_labels == 1.)[0]
    event_times = event_times[idx_times]
    event_labels = np.where((event_labels == 1.))[1]

    unpadded = event_times != args.padding_id
    event_times, event_labels = event_times[unpadded], event_labels[unpadded]

    model_intensities = {
        k: m.intensity(query=query, events=events) for k, m in models.items()}
    model_intensities = {
        k: (ints.cpu().detach().numpy()[0], mask.cpu().detach().numpy()[0])
        for k, (ints, mask) in model_intensities.items()}

    model_artifacts = {k:
        m.artifacts(query=query, events=events) for k, m in models.items()}

    model_cumulative_intensities = {
        k: (v[1], v[2]) for k, v in model_artifacts.items()}
    model_cumulative_intensities = {
        k: (ints.cpu().detach().numpy()[0], mask.cpu().detach().numpy()[0])
        for k, (ints, mask) in model_cumulative_intensities.items()}
    model_artifacts = {k: v[3] for k, v in model_artifacts.items()}

    with open(os.path.join(
            args.save_dir, 'int_to_codes_to_plot.json'), 'r') as h:
        int_to_codes_to_plot = json.load(h)
    with open(os.path.join(
            args.save_dir, 'int_to_codes.json'), 'r') as h:
        int_to_codes = json.load(h)
    with open(os.path.join(args.save_dir, 'codes_to_names.json'), 'r') as h:
        codes_to_names = json.load(h)
    int_to_names_to_plot = {
        k: codes_to_names[v] for k, v in int_to_codes_to_plot.items()}
    int_to_names = {k: codes_to_names[v] for k, v in int_to_codes.items()}

    query = query.cpu().detach().numpy()[0]
    model_intensities = {
        k: (filter_by_mask(query, mask=mask),
            filter_by_mask(ints, mask=mask))
        for k, (ints, mask) in model_intensities.items()}
    model_intensities = {
        k: (q, ints[:, [int(i) for i in int_to_names_to_plot.keys()]])
        for k, (q, ints) in model_intensities.items()}

    model_cumulative_intensities = {
        k: (filter_by_mask(query, mask=mask),
            filter_by_mask(ints, mask=mask))
        for k, (ints, mask) in model_cumulative_intensities.items()}
    model_cumulative_intensities = {
        k: (q, ints[:, [int(i) for i in int_to_names_to_plot.keys()]])
        for k, (q, ints) in model_cumulative_intensities.items()}

    images_urls = plot_attn_weights(
        model_artifacts=model_artifacts,
        event_times=event_times,
        event_labels=event_labels,
        idx_times=idx_times,
        query=query,
        args=args,
        class_names=list(int_to_names.values()),
        epoch=epoch,
        images_urls=images_urls)

    f, a = fig_hawkes(
        intensities=model_intensities,
        cumulative_intensities=model_cumulative_intensities,
        event_times=event_times,
        event_labels=event_labels,
        class_names=int_to_names_to_plot,
        epoch=epoch)

    if epoch is not None:
        epoch_str = get_epoch_str(epoch=epoch, max_epochs=args.train_epochs)

    intensity_dir = os.path.join(args.plots_dir, "intensity")
    if epoch is not None:
        plot_path = os.path.join(intensity_dir, "epoch_" + epoch_str + ".jpg")
    else:
        plot_path = intensity_dir + ".jpg"

    f.savefig(plot_path, dpi=300, bbox_inches='tight')
    if save_on_mlflow:
        assert epoch is not None, "Epoch must not be None with mlflow active"
        mlflow.log_artifact(plot_path, "intensity/epoch_" + epoch_str)
    if images_urls is not None:
        images_urls['intensity'].append(plot_path)

    return images_urls
