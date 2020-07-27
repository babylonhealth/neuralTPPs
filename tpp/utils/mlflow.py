import json
import mlflow
import numpy as np
import os

from argparse import Namespace
from typing import Dict
from itertools import islice

from tpp.models.base.process import Process
from tpp.models.base.enc_dec import EncDecProcess


def log_metrics(
        model: Process,
        metrics: dict,
        val_metrics: dict,
        args: Namespace,
        epoch: int):
    if args.eval_metrics:
        with open(
                os.path.join(args.save_dir,
                             'int_to_codes.json'), 'r') as h:
            int_to_codes = json.load(h)
        with open(
                os.path.join(args.save_dir,
                             'codes_to_names.json'), 'r') as h:
            codes_to_names = json.load(h)
        int_to_names = {
            k: codes_to_names[v] for k, v in int_to_codes.items()}
        metrics_log = metrics_log_dict(
            args=args,
            int_to_names=int_to_names,
            val_metrics=val_metrics)
        metrics.update(metrics_log)

    if (model.name == "stub_hawkes"
            and args.load_from_dir in [None, "hawkes"]):
        metrics_hawkes = hawkes_log_dict(args=args, model=model)
        metrics.update(metrics_hawkes)

    mlflow.log_metrics(metrics, step=epoch)


def vector_log_dict(x: np.ndarray, prefix: str):
    return {
        "{}__{}".format(prefix, i): xi
        for i, xi in enumerate(x)}


def vector_from_log_dict(x: Dict, prefix: str, dtype=np.float32):
    x = {k: v for k, v in x.items() if k.startswith(prefix)}
    x = {k[len(prefix + "__"):]: v for k, v in x.items()}
    x = {int(k): v for k, v in x.items()}
    x = [x[k] for k in sorted(x.keys())]
    result = np.stack(x).astype(dtype)
    return result


def matrix_log_dict(x: np.ndarray, prefix: str):
    return {
        "{}__{}_{}".format(prefix, i, j): xij
        for i, xi in enumerate(x)
        for j, xij in enumerate(xi)}


def matrix_from_log_dict(x: Dict, prefix: str, dtype=np.float32):
    x = {k: v for k, v in x.items() if k.startswith(prefix)}
    x = {k[len(prefix + "__"):]: v for k, v in x.items()}
    x = {tuple(k.split("_")): v for k, v in x.items()}
    x = {tuple(int(k1) for k1 in k): v for k, v in x.items()}
    row_idxs, col_idxs = zip(*x.keys())
    rows, cols = max(row_idxs) + 1, max(col_idxs) + 1
    result = np.zeros(shape=[rows, cols], dtype=dtype)
    for (i, j), v in x.items():
        result[i, j] = v
    return result


def max_abs_rel_diff(x, y):
    return np.max(np.abs((x - y) / y))


def hawkes_log_dict(args: Namespace, model: Process):
    if isinstance(model, EncDecProcess):
        model = model.decoder

    alpha_model = np.exp(model.log_alpha.detach().cpu().numpy())
    beta_model = model.beta.detach().cpu().numpy()
    mu_model = np.exp(model.log_mu.detach().cpu().numpy())

    alpha_model_log = matrix_log_dict(alpha_model, prefix="alpha_model")
    beta_model_log = matrix_log_dict(beta_model, prefix="beta_model")
    mu_model_log = vector_log_dict(mu_model, prefix="mu_model")
    model_log = {**alpha_model_log, **beta_model_log, **mu_model_log}

    alpha_true_log = matrix_log_dict(args.alpha, prefix="alpha_true")
    beta_true_log = matrix_log_dict(args.beta, prefix="beta_true")
    mu_true_log = vector_log_dict(args.mu, prefix="mu_true")
    true_log = {**alpha_true_log, **beta_true_log, **mu_true_log}

    alpha_max_abs_rel_diff = max_abs_rel_diff(alpha_model, args.alpha)
    beta_max_abs_rel_diff = max_abs_rel_diff(beta_model, args.beta)
    mu_max_abs_rel_diff = max_abs_rel_diff(mu_model, args.mu)
    diff_log = {
        "alpha": alpha_max_abs_rel_diff,
        "beta": beta_max_abs_rel_diff,
        "mu": mu_max_abs_rel_diff}
    diff_log = {k + "_max_abs_rel_diff": v for k, v in diff_log.items()}

    return {**model_log, **true_log, **diff_log}


def metrics_log_dict(args: Namespace, int_to_names: dict, val_metrics: dict):
    log_dict = {}

    if args.multi_labels:
        if args.eval_metrics_per_class and args.marks < 20:
            for i in range(args.marks):
                log_dict["roc_auc_" + ''.join(
                    x for x in int_to_names[
                        str(i)] if x.isalnum())] = val_metrics["roc_auc"][i]

        log_dict["roc_auc_macro"] = val_metrics["roc_auc_macro"]
        log_dict["roc_auc_micro"] = val_metrics["roc_auc_micro"]
        log_dict["roc_auc_weighted"] = val_metrics["roc_auc_weighted"]

    else:
        if args.eval_metrics_per_class:
            for i in range(args.marks):
                log_dict["precision_" + ''.join(
                    x for x in int_to_names[
                        str(i)] if x.isalnum())] = val_metrics["precision"][i]
                log_dict["recall_" + ''.join(
                    x for x in int_to_names[
                        str(i)] if x.isalnum())] = val_metrics["recall"][i]
                log_dict["f1_" + ''.join(
                    x for x in int_to_names[
                        str(i)] if x.isalnum())] = val_metrics["f1"][i]
                log_dict["acc_" + ''.join(
                    x for x in int_to_names[
                        str(i)] if x.isalnum())] = val_metrics["acc"][i]

        log_dict["precision_macro"] = val_metrics["pre_macro"]
        log_dict["precision_micro"] = val_metrics["pre_micro"]
        log_dict["precision_weighted"] = val_metrics["pre_weighted"]
        log_dict["recall_macro"] = val_metrics["rec_macro"]
        log_dict["recall_micro"] = val_metrics["rec_micro"]
        log_dict["recall_weighted"] = val_metrics["rec_weighted"]
        log_dict["f1_macro"] = val_metrics["f1_macro"]
        log_dict["f1_micro"] =  val_metrics["f1_micro"]
        log_dict["f1_weighted"] = val_metrics["f1_weighted"]
        log_dict["acc_macro"] = val_metrics["acc_macro"]
        log_dict["acc_micro"] = val_metrics["acc_micro"]
        log_dict["acc_weighted"] = val_metrics["acc_weighted"]

    return log_dict


def params_log_dict(args: Namespace):
    params_to_log = dict(vars(args))
    alpha_params = matrix_log_dict(params_to_log.pop("alpha"), prefix="alpha")
    beta_params = matrix_log_dict(params_to_log.pop("beta"), prefix="beta")
    mu_params = vector_log_dict(params_to_log.pop("mu"), prefix="mu")
    return {**alpha_params, **beta_params, **mu_params, **params_to_log}


def chunks(x, chunksize=10):
    it = iter(x)
    for i in range(0, len(x), chunksize):
        yield {k: x[k] for k in islice(it, chunksize)}


def batch_log_params(params, chunksize=10):
    param_chunks = chunks(params, chunksize=chunksize)
    for chunk in param_chunks:
        return mlflow.log_params(chunk)


def get_epoch_str(epoch: int, max_epochs: int):
    epoch_str = str(epoch)
    epoch_str_pad_len = len(str(max_epochs)) - len(epoch_str)
    if epoch_str_pad_len > 0:
        epoch_str = ("0" * epoch_str_pad_len) + epoch_str
    return epoch_str
