"""Top-level CLI dispatcher.

Subcommands:
    preprocess  --dataset {replay,3dmad,csmad,all} [--limit N]
    train       [--config PATH] [--protocol {combined,replay,3dmad,csmad}] [overrides...]
    train-cv    [--config PATH] [--n-folds 5] [--protocol ...] [overrides...]
    eval        --checkpoint PATH [--split {train,devel,test}] [--datasets ...]
    tune        [--n-trials 30] [--epochs 15] [--protocol ...]
    eval-cross  --checkpoint PATH [--datasets ...]   (devel-calibrated HTER on test)
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.config import PROJECT_ROOT


# ---------------------------------------------------------------------------
# preprocess
# ---------------------------------------------------------------------------

def cmd_preprocess(args: argparse.Namespace) -> None:
    if args.dataset in ("replay", "all"):
        from src.preprocessing import extract_replay
        extract_replay.run(limit=args.limit)
    if args.dataset in ("3dmad", "all"):
        from src.preprocessing import extract_3dmad
        extract_3dmad.run(limit=args.limit)
    if args.dataset in ("csmad", "all"):
        from src.preprocessing import extract_csmad
        extract_csmad.run(limit=args.limit)


# ---------------------------------------------------------------------------
# train
# ---------------------------------------------------------------------------

def cmd_train(args: argparse.Namespace) -> None:
    from src.training.train import TrainConfig, parse_overrides, run

    cfg_path = Path(args.config) if args.config else PROJECT_ROOT / "configs" / "default.yaml"
    cfg = TrainConfig.from_yaml(cfg_path)
    cfg = parse_overrides(args, cfg)
    run(cfg, max_steps=args.max_steps)


def cmd_train_cv(args: argparse.Namespace) -> None:
    from src.training.cross_val import run_cv
    from src.training.train import TrainConfig, parse_overrides

    cfg_path = Path(args.config) if args.config else PROJECT_ROOT / "configs" / "default.yaml"
    cfg = TrainConfig.from_yaml(cfg_path)
    cfg = parse_overrides(args, cfg)
    run_cv(cfg, n_folds=args.n_folds)


# ---------------------------------------------------------------------------
# tune
# ---------------------------------------------------------------------------

def cmd_tune(args: argparse.Namespace) -> None:
    from src.training.train import TrainConfig, parse_overrides
    from src.tuning.optuna_search import run_search

    cfg_path = Path(args.config) if args.config else PROJECT_ROOT / "configs" / "default.yaml"
    cfg = TrainConfig.from_yaml(cfg_path)
    cfg = parse_overrides(args, cfg)
    run_search(
        base_cfg=cfg,
        n_trials=args.n_trials,
        epochs_per_trial=args.epochs_per_trial,
        study_name=args.study_name,
    )


# ---------------------------------------------------------------------------
# eval
# ---------------------------------------------------------------------------

def cmd_eval(args: argparse.Namespace) -> None:
    from src.training.eval_runner import evaluate_split

    evaluate_split(
        ckpt_path=Path(args.checkpoint),
        split=args.split,
        eval_datasets=args.datasets,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        use_amp=not args.no_amp,
    )


def cmd_eval_cross(args: argparse.Namespace) -> None:
    from src.training.eval_runner import evaluate_cross

    evaluate_cross(
        ckpt_path=Path(args.checkpoint),
        eval_datasets=args.datasets,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        use_amp=not args.no_amp,
    )


# ---------------------------------------------------------------------------
# arg parsing
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(prog="liveness")
    sub = parser.add_subparsers(dest="cmd", required=True)

    # preprocess --------------------------------------------------------------
    pre = sub.add_parser("preprocess", help="extract face crops from raw datasets")
    pre.add_argument("--dataset", choices=["replay", "3dmad", "csmad", "all"], required=True)
    pre.add_argument("--limit", type=int, default=None, help="cap number of videos (for smoke tests)")
    pre.set_defaults(func=cmd_preprocess)

    # train -------------------------------------------------------------------
    tr = sub.add_parser("train", help="train AttackNet v2.2")
    tr.add_argument("--config", type=str, default=None,
                    help="path to YAML config (default: configs/default.yaml)")
    tr.add_argument("--protocol", choices=["combined", "replay", "3dmad", "csmad"], default=None,
                    help="shorthand for setting train+val datasets")
    tr.add_argument("--epochs", type=int, default=None)
    tr.add_argument("--batch-size", type=int, default=None)
    tr.add_argument("--lr", type=float, default=None)
    tr.add_argument("--dropout", type=float, default=None)
    tr.add_argument("--run-name", type=str, default=None)
    tr.add_argument("--no-amp", action="store_true", help="disable mixed precision (debug)")
    tr.add_argument("--max-steps", type=int, default=None,
                    help="cap training steps per epoch (for smoke tests)")
    tr.set_defaults(func=cmd_train)

    # train-cv ----------------------------------------------------------------
    cv = sub.add_parser("train-cv", help="subject-disjoint K-fold cross-validation")
    cv.add_argument("--config", type=str, default=None,
                    help="path to YAML config (default: configs/default.yaml)")
    cv.add_argument("--n-folds", type=int, default=5)
    cv.add_argument("--protocol", choices=["combined", "replay", "3dmad", "csmad"], default=None,
                    help="shorthand for setting train+val datasets")
    cv.add_argument("--epochs", type=int, default=None)
    cv.add_argument("--batch-size", type=int, default=None)
    cv.add_argument("--lr", type=float, default=None)
    cv.add_argument("--dropout", type=float, default=None)
    cv.add_argument("--run-name", type=str, default=None)
    cv.add_argument("--no-amp", action="store_true", help="disable mixed precision")
    cv.set_defaults(func=cmd_train_cv)

    # tune --------------------------------------------------------------------
    tu = sub.add_parser("tune", help="Optuna hyperparameter search")
    tu.add_argument("--config", type=str, default=None,
                    help="path to YAML config (default: configs/default.yaml)")
    tu.add_argument("--n-trials", type=int, default=30,
                    help="number of Optuna trials to run")
    tu.add_argument("--epochs-per-trial", type=int, default=15,
                    help="max epochs per trial (pruner may cut short)")
    tu.add_argument("--study-name", type=str, default="attacknet_v22",
                    help="Optuna study name (for DB grouping)")
    tu.add_argument("--protocol", choices=["combined", "replay", "3dmad", "csmad"], default=None,
                    help="shorthand for setting train+val datasets")
    tu.add_argument("--run-name", type=str, default=None)
    tu.add_argument("--no-amp", action="store_true", help="disable mixed precision")
    tu.set_defaults(func=cmd_tune)

    # eval --------------------------------------------------------------------
    ev = sub.add_parser("eval", help="evaluate a checkpoint on a split")
    ev.add_argument("--checkpoint", type=str, required=True)
    ev.add_argument("--split", choices=["train", "devel", "test"], default="test")
    ev.add_argument("--datasets", nargs="*", default=None,
                    help="filter to these dataset names; default = all")
    ev.add_argument("--batch-size", type=int, default=64)
    ev.add_argument("--num-workers", type=int, default=4)
    ev.add_argument("--no-amp", action="store_true")
    ev.set_defaults(func=cmd_eval)

    # eval-cross --------------------------------------------------------------
    evc = sub.add_parser(
        "eval-cross",
        help="calibrate threshold on devel, report HTER on test",
    )
    evc.add_argument("--checkpoint", type=str, required=True)
    evc.add_argument("--datasets", nargs="*", default=None,
                     help="filter eval splits to these dataset names")
    evc.add_argument("--batch-size", type=int, default=64)
    evc.add_argument("--num-workers", type=int, default=4)
    evc.add_argument("--no-amp", action="store_true")
    evc.set_defaults(func=cmd_eval_cross)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
