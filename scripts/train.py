import mlflow
import mlflow.pytorch
import imageio
import json
import numpy as np
import os
import stat
import time
import torchvision

import torch as th
from torch.optim import Adam
from torch.utils.data import DataLoader

from argparse import Namespace
from copy import deepcopy
from typing import Dict, Tuple, Optional
from pathlib import Path
from tqdm import tqdm

from tpp.utils.events import get_events, get_window

from tpp.utils.mlflow import params_log_dict, get_epoch_str, log_metrics

from tpp.models import get_model
from tpp.models.base.process import Process
from tpp.utils.cli import parse_args
from tpp.utils.metrics import eval_metrics
from tpp.utils.plot import log_figures
from tpp.utils.data import get_loader, load_data
from tpp.utils.logging import get_status
from tpp.utils.lr_scheduler import create_lr_scheduler
from tpp.utils.run import make_deterministic
from tpp.utils.stability import check_tensor

torchvision.__version__ = '0.4.0'


def get_loss(
        model: Process,
        batch: Dict[str, th.Tensor],
        args: Namespace,
        eval_metrics: Optional[bool] = False,
        dynamic_batch_length: Optional[bool] = True,
) -> Tuple[th.Tensor, th.Tensor, Dict]:
    times, labels = batch["times"], batch["labels"]
    labels = (labels != 0).type(labels.dtype)

    if dynamic_batch_length:
        seq_lens = batch["seq_lens"]
        max_seq_len = seq_lens.max()
        times, labels = times[:, :max_seq_len], labels[:, :max_seq_len]

    mask = (times != args.padding_id).type(times.dtype)
    times = times * args.time_scale

    window_start, window_end = get_window(times=times, window=args.window)
    events = get_events(
        times=times, mask=mask, labels=labels,
        window_start=window_start, window_end=window_end)

    loss, loss_mask, artifacts = model.neg_log_likelihood(events=events)  # [B]

    if eval_metrics:
        events_times = events.get_times()
        log_p, y_pred_mask = model.log_density(
            query=events_times, events=events)  # [B,L,M], [B,L]
        if args.multi_labels:
            y_pred = log_p  # [B,L,M]
            labels = events.labels
        else:
            y_pred = log_p.argmax(-1).type(log_p.dtype)  # [B,L]
            labels = events.labels.argmax(-1).type(events.labels.dtype)
        artifacts['y_pred'] = y_pred
        artifacts['y_true'] = labels
        artifacts['y_pred_mask'] = y_pred_mask

    return loss, loss_mask, artifacts


def detach(x: th.Tensor):
    return x.cpu().detach().numpy()


def evaluate(model: Process, args: Namespace, loader: DataLoader
             ) -> Dict[str, float]:
    """Evaluate a model on a specific dataset.

    Args:
        model: The model to evaluate.
        args: Arguments for evaluation
        loader: The loader corresponding to the dataset to evaluate on.

    Returns:
        Dictionary containing all metrics evaluated and averaged over total
            sequences.

    """
    model.eval()

    t0, epoch_loss, epoch_loss_per_time, n_seqs = time.time(), 0., 0., 0.
    pred_labels, gold_labels, mask_labels = [], [], []
    results, count = {}, 0

    for batch in tqdm(loader) if args.verbose else loader:
        loss, loss_mask, artifacts = get_loss(  # [B]
            model, batch=batch, eval_metrics=args.eval_metrics, args=args,
            # For eval, use padded data for metrics evaluation
            dynamic_batch_length=False)
        if count == int(args.val_size / args.batch_size):
            break
        count += 1

        loss = loss * loss_mask  # [B]
        epoch_loss += detach(th.sum(loss))

        loss_per_time = loss / artifacts["interval"]
        epoch_loss_per_time += detach(th.sum(loss_per_time))

        n_seqs_batch = detach(th.sum(loss_mask))
        n_seqs += n_seqs_batch

        if args.eval_metrics:
            pred_labels.append(detach(artifacts['y_pred']))
            gold_labels.append(detach(artifacts['y_true']))
            mask_labels.append(detach(artifacts['y_pred_mask']))

    if args.eval_metrics:
        results = eval_metrics(
            pred=pred_labels,
            gold=gold_labels,
            mask=mask_labels,
            results=results,
            n_class=args.marks,
            multi_labels=args.multi_labels)

    dur = time.time() - t0
    results["dur"] = dur

    results["loss"] = float(epoch_loss / n_seqs)
    results["loss_per_time"] = float(epoch_loss_per_time / n_seqs)

    return results


def train(
        model: Process,
        args: Namespace,
        loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader) -> Tuple[Process, dict]:
    """Train a model.

    Args:
        model: Model to be trained.
        args: Arguments for training.
        loader: The dataset for training.
        val_loader: The dataset for evaluation.
        test_loader: The dataset for testing

    Returns:
        Best trained model from early stopping.

    """
    if args.include_poisson:
        processes = model.processes.keys()
        modules = []
        for p in processes:
            if p != 'poisson':
                modules.append(getattr(model, p))
        optimizer = Adam(
            [{'params': m.parameters()} for m in modules] + [
                {'params': model.alpha}] + [
                {'params': model.poisson.parameters(),
                 'lr': args.lr_poisson_rate_init}
            ], lr=args.lr_rate_init)
    else:
        optimizer = Adam(model.parameters(), lr=args.lr_rate_init)
    lr_scheduler = create_lr_scheduler(optimizer=optimizer, args=args)

    parameters = dict(model.named_parameters())

    lr_wait, cnt_wait, best_loss, best_epoch = 0, 0, 1e9, 0
    best_state = deepcopy(model.state_dict())
    train_dur, val_dur, images_urls = list(), list(), dict()
    images_urls['intensity'] = list()
    images_urls['src_attn'] = list()
    images_urls['tgt_attn'] = list()

    epochs = range(args.train_epochs)
    if args.verbose:
        epochs = tqdm(epochs)

    for epoch in epochs:
        t0, _ = time.time(), model.train()

        if args.lr_scheduler != 'plateau':
            lr_scheduler.step()

        for i, batch in enumerate((tqdm(loader)) if args.verbose else loader):
            optimizer.zero_grad()
            loss, loss_mask, _ = get_loss(model, batch=batch, args=args)  # [B]
            loss = loss * loss_mask
            loss = th.sum(loss)
            check_tensor(loss)
            loss.backward()
            optimizer.step()
        train_dur.append(time.time() - t0)

        train_metrics = evaluate(model, args=args, loader=loader)
        val_metrics = evaluate(model, args=args, loader=val_loader)

        val_dur.append(val_metrics["dur"])

        if args.lr_scheduler == 'plateau':
            lr_scheduler.step(metrics=val_metrics["loss"])

        new_best = val_metrics["loss"] < best_loss
        if args.loss_relative_tolerance is not None:
            abs_rel_loss_diff = (val_metrics["loss"] - best_loss) / best_loss
            abs_rel_loss_diff = abs(abs_rel_loss_diff)
            above_numerical_tolerance = (abs_rel_loss_diff >
                                         args.loss_relative_tolerance)
            new_best = new_best and above_numerical_tolerance

        if new_best:
            best_loss, best_t = val_metrics["loss"], epoch
            cnt_wait, lr_wait = 0, 0
            best_state = deepcopy(model.state_dict())
        else:
            cnt_wait, lr_wait = cnt_wait + 1, lr_wait + 1

        if cnt_wait == args.patience:
            print("Early stopping!")
            break

        if epoch % args.save_model_freq == 0 and parsed_args.use_mlflow:
            current_state = deepcopy(model.state_dict())
            model.load_state_dict(best_state)
            epoch_str = get_epoch_str(epoch=epoch,
                                      max_epochs=args.train_epochs)

            mlflow.pytorch.log_model(model, "models/epoch_" + epoch_str)
            images_urls = log_figures(
                model=model,
                test_loader=test_loader,
                epoch=epoch,
                args=args,
                images_urls=images_urls)
            model.load_state_dict(current_state)

        lr = optimizer.param_groups[0]['lr']
        train_metrics["lr"] = lr
        if args.include_poisson:
            lr_poisson = optimizer.param_groups[-1]['lr']
        else:
            lr_poisson = lr

        status = get_status(
            args=args, epoch=epoch, lr=lr, lr_poisson=lr_poisson,
            parameters=parameters, train_loss=train_metrics["loss"],
            val_metrics=val_metrics, cnt_wait=cnt_wait)
        print(status)

        if args.use_mlflow and epoch % args.logging_frequency == 0:
            loss_metrics = {
                "lr": train_metrics["lr"],
                "train_loss": train_metrics["loss"],
                "train_loss_per_time": train_metrics["loss_per_time"],
                "valid_loss": val_metrics["loss"],
                "valid_loss_per_time": val_metrics["loss_per_time"]}
            log_metrics(
                model=model,
                metrics=loss_metrics,
                val_metrics=val_metrics,
                args=args,
                epoch=epoch)

    model.load_state_dict(best_state)
    return model, images_urls


def main(args: Namespace):
    if args.verbose:
        print(args)

    datasets = load_data(args=args)

    loaders = dict()
    loaders["train"] = get_loader(datasets["train"], args=args, shuffle=True)
    loaders["val"] = get_loader(datasets["val"], args=args, shuffle=False)
    loaders["test"] = get_loader(datasets["test"], args=args, shuffle=False)

    model = get_model(args)

    if args.mu_cheat and "poisson" in model.processes:
        poisson = model.processes["poisson"].decoder
        mu = th.from_numpy(args.mu).type(
            poisson.mu.dtype).to(poisson.mu.device)
        poisson.mu.data = mu

    model, images_urls = train(
        model, args=args, loader=loaders["train"],
        val_loader=loaders["val"], test_loader=loaders["test"])

    metrics = {
        k: evaluate(model=model, args=args, loader=l)
        for k, l in loaders.items()}
    if args.verbose:
        print(metrics)

    if args.use_mlflow:
        loss_metrics = {
            "train_loss": metrics["train"]["loss"],
            "train_loss_per_time": metrics["train"]["loss_per_time"],
            "valid_loss": metrics["test"]["loss"],
            "valid_loss_per_time": metrics["test"]["loss_per_time"]}
        log_metrics(
            model=model,
            metrics=loss_metrics,
            val_metrics=metrics["test"],
            args=args,
            epoch=args.train_epochs)
        mlflow.pytorch.log_model(model, "models")


if __name__ == "__main__":
    parsed_args = parse_args()

    if parsed_args.load_from_dir is not None:
        parsed_args.data_dir = os.path.expanduser(parsed_args.data_dir)
        parsed_args.save_dir = os.path.join(parsed_args.data_dir,
                                            parsed_args.load_from_dir)
        with open(os.path.join(parsed_args.save_dir, 'args.json'), 'r') as fp:
            args_dict_json = json.load(fp)
        args_dict = vars(parsed_args)
        print("Warning: overriding some args from json:")
        shared_keys = set(args_dict_json).intersection(set(args_dict))
        for k in shared_keys:
            v1, v2 = args_dict[k], args_dict_json[k]
            is_equal = np.allclose(v1, v2) if isinstance(
                v1, np.ndarray) else v1 == v2
            if not is_equal:
                print(f"    {k}: {v1} -> {v2}")
        args_dict.update(args_dict_json)
        parsed_args = Namespace(**args_dict)
        parsed_args.mu = np.array(parsed_args.mu, dtype=np.float32)
        parsed_args.alpha = np.array(
            parsed_args.alpha, dtype=np.float32).reshape(
            parsed_args.mu.shape * 2)
        parsed_args.beta = np.array(
            parsed_args.beta, dtype=np.float32).reshape(
            parsed_args.mu.shape * 2)

    else:
        parsed_args.data_dir = os.path.expanduser(parsed_args.data_dir)
        parsed_args.save_dir = os.path.join(parsed_args.data_dir, "None")
        Path(parsed_args.save_dir).mkdir(parents=True, exist_ok=True)

    cuda = th.cuda.is_available() and not parsed_args.disable_cuda
    if cuda:
        parsed_args.device = th.device('cuda')
    else:
        parsed_args.device = th.device('cpu')

    # check_repo(allow_uncommitted=not parsed_args.use_mlflow)
    make_deterministic(seed=parsed_args.seed)

    # Create paths for plots
    parsed_args.plots_dir = os.path.expanduser(parsed_args.plots_dir)
    Path(parsed_args.plots_dir).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(parsed_args.plots_dir, "src_attn")).mkdir(
        parents=True, exist_ok=True)
    Path(os.path.join(parsed_args.plots_dir, "tgt_attn")).mkdir(
        parents=True, exist_ok=True)
    Path(os.path.join(parsed_args.plots_dir, "intensity")).mkdir(
        parents=True, exist_ok=True)

    if parsed_args.use_mlflow:
        mlflow.set_tracking_uri(parsed_args.remote_server_uri)
        mlflow.set_experiment(parsed_args.experiment_name)
        mlflow.start_run(run_name=parsed_args.run_name)
        params_to_log = params_log_dict(parsed_args)
        mlflow.log_params(params_to_log)

    main(args=parsed_args)
