"""Train and select the neural ageing surrogates.

    python -m surrogate.train_neural cv                      # grouped cross-validation, resumable
    python -m surrogate.train_neural cv --arms gru ams-cenn  # a subset, e.g. to run in parallel
    python -m surrogate.train_neural summarize               # table and selection.json
    python -m surrogate.train_neural tune --family cenn-certified --trials 0 30   # random search
    python -m surrogate.train_neural tune-summarize --family cenn-certified       # register the winner
    python -m surrogate.train_neural final --arms gru        # train and save a model for shipping
    python -m surrogate.train_neural final --selected        # the selected arms from selection.json

Settings are in the [neural] section of inputs/surrogate.toml.
"""
import argparse
import json
import logging

from inputs import ProjectConfig
from surrogate.features import TrajectoryFeaturizer
from surrogate.neural.training import NeuralTrainer

logging.basicConfig(
    filename="formation_system.log",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    filemode="a",
)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["cv", "summarize", "final", "tune", "tune-summarize"])
    parser.add_argument("--family", choices=sorted(NeuralTrainer.FAMILIES), default="cenn-certified",
                        help="tune: which model family to search")
    parser.add_argument("--trials", nargs=2, type=int, default=[0, 30], metavar=("FIRST", "LAST"),
                        help="tune: trial indices [FIRST, LAST); disjoint ranges can run in parallel")
    parser.add_argument("--arms", nargs="+", default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--selected", action="store_true",
                        help="final: train the arms named in selection.json")
    args = parser.parse_args()

    config = ProjectConfig()
    trainer = NeuralTrainer(config.surrogate, config, TrajectoryFeaturizer(config.surrogate))
    if args.command == "cv":
        trainer.cross_validate(args.arms, args.seeds)
    elif args.command == "summarize":
        trainer.summarize()
    elif args.command == "tune":
        trainer.tune(args.family, *args.trials)
    elif args.command == "tune-summarize":
        trainer.tune_summarize(args.family)
    else:
        arms = list(args.arms or [])
        if args.selected:
            selection = json.loads((trainer.cv_dir / "selection.json").read_text(encoding="utf-8"))
            for key in ("best_overall", "best_cellular", "best_certified", "best_reference"):
                if selection.get(key) and selection[key] not in arms:
                    arms.append(selection[key])
        if not arms:
            parser.error("final needs --arms or --selected")
        for arm in arms:
            trainer.train_final(arm, args.seeds)


if __name__ == "__main__":
    main()
