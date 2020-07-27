import numpy as np
import pandas as pd
import tabulate
import torch as th

from typing import Optional

from argparse import Namespace

tabulate.MIN_PADDING = 0


def _format_key(x, split=".", key_length=5):
    x = x.split(split)
    x = [y[:key_length] for y in x]
    x = split.join(x)
    return x


def _format_dict(x):
    keys = sorted(list(x.keys()))
    results = ["{} {:03f}".format(_format_key(k), x[k]) for k in keys]
    return " | ".join(results)


def get_summary(parameters, formatted=True):
    summary = {
        k: np.mean(v.cpu().detach().numpy()) for k, v in parameters.items()}
    if formatted:
        summary = _format_dict(summary)
    return summary


def get_log_dicts(
        args: Namespace,
        epoch: int,
        lr: float,
        lr_poisson: float,
        parameters: dict,
        train_loss: th.float,
        val_metrics: dict,
        cnt_wait: int,
        key_length=5):
    train = {
        "epoch": f"{epoch}/{args.train_epochs}",
        "lr": lr,
        "lr_poisson": lr_poisson,
        "patience": f"{cnt_wait}/{args.patience}"}
    params = get_summary(parameters, formatted=False)

    encoder_params = {k: v for k, v in params.items() if ".encoder." in k}
    decoder_params = {k: v for k, v in params.items() if ".decoder." in k}
    other_params = {k: v for k, v in params.items() if k not in
                    set(encoder_params.keys()).union(
                        set(decoder_params.keys()))}

    encoder_params = {
        k.split(".encoder.")[-1]: v for k, v in encoder_params.items()}
    encoder_params = {
        _format_key(k, key_length=key_length): v
        for k, v in encoder_params.items()}

    decoder_params = {
        k.split(".decoder.")[-1]: v for k, v in decoder_params.items()}
    decoder_params = {
        _format_key(k, key_length=key_length): v
        for k, v in decoder_params.items()}

    other_params = {
        _format_key(k, key_length=key_length): v for
        k, v in other_params.items()}

    params = {
        "encoder": encoder_params,
        "decoder": decoder_params,
        "other": other_params}

    metrics = {"train_loss": train_loss,
               "val_loss": val_metrics["loss"]}
    if args.eval_metrics:
        if args.multi_labels:
            metrics.update({
                "roc_auc_macro": val_metrics["roc_auc_macro"],
                "roc_auc_micro": val_metrics["roc_auc_micro"],
                "roc_auc_weighted": val_metrics["roc_auc_weighted"]})
        else:
            metrics.update({
                "pre_weighted": val_metrics["pre_weighted"],
                "rec_weighted": val_metrics["rec_weighted"],
                "f1_weighted": val_metrics["f1_weighted"],
                "acc_weighted": val_metrics["acc_weighted"]})

    return {"train": train, "params": params, "metrics": metrics}


def get_status(
        args: Namespace,
        epoch: int,
        lr: float,
        lr_poisson: float,
        parameters: dict,
        train_loss: th.float,
        val_metrics: dict,
        cnt_wait: int,
        print_header_freq: Optional[int] = 10,
        key_length: Optional[int] = 5):
    log_dicts = get_log_dicts(
        args=args, epoch=epoch, lr=lr, lr_poisson=lr_poisson,
        parameters=parameters, train_loss=train_loss, val_metrics=val_metrics,
        cnt_wait=cnt_wait)

    params_dict = log_dicts.pop("params")
    for k, v in params_dict.items():
        params_dict[k] = {k1.replace(".", "\n"): v1 for k1, v1 in v.items()}
    params_dict = {
        _format_key(k, key_length=key_length): v
        for k, v in params_dict.items()}

    log_dicts = {
        _format_key(k, key_length=key_length):
            {_format_key(k1, key_length=key_length): v1
             for k1, v1 in v.items()}
        for k, v in log_dicts.items()}

    log_dicts.update(params_dict)

    log_dicts = {
        k: {k1: [v1] for k1, v1 in v.items()} for k, v in log_dicts.items()}
    log_dicts = {k: pd.DataFrame(v) for k, v in log_dicts.items()}
    log_df = pd.concat(log_dicts.values(), axis=1, keys=log_dicts.keys())
    h = list(map('\n'.join, log_df.columns.tolist()))

    msg_str = tabulate.tabulate(
        log_df, headers=h, tablefmt='grid', showindex=False, floatfmt=".4f",
        numalign="center", stralign="center")
    msg_split = msg_str.split("\n")
    header = msg_split[:-2]
    record = msg_split[-2]

    msg = record
    if epoch % print_header_freq == 0:
        msg = "\n".join(header + [msg])

    return msg


def get_status_old(
        args: Namespace,
        epoch: int,
        lr: float,
        parameters: dict,
        train_loss: th.float,
        val_metrics: dict,
        cnt_wait: int) -> str:
    status = f"epoch {epoch}/{args.train_epochs} |"
    status += f" lr: {lr:.4f} | "
    status += get_summary(parameters)
    status += f" | train_loss: {train_loss:.4f}"
    val_loss = val_metrics["loss"]
    status += f" | val_loss: {val_loss:.4f}"
    if args.eval_metrics:
        acc = val_metrics["pre_macro"]
        recall = val_metrics["rec_macro"]
        f1 = val_metrics['f1_macro']
        status += f" | precision: {acc:.4f}"
        status += f" | recall: {recall:.4f}"
        status += f" | F1: {f1:.4f}"
    status += f" | patience: {cnt_wait}/{args.patience}"
    return status
