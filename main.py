import argparse
import os
import sys
import warnings
from typing import Union

import sparsing

from data_loading import DATASET_NAMES
from perform_experiment import perform_experiment

# the only warning raised is ConvergenceWarning for linear SVM, which is
# acceptable (max_iter is already higher than default); unfortunately, we
# have to do this globally for all warnings to affect child processes in
# cross-validation
if not sys.warnoptions:
    warnings.simplefilter("ignore")
    os.environ["PYTHONWARNINGS"] = "ignore"  # also affect subprocesses


def ensure_bool(data: Union[bool, str]) -> bool:
    if isinstance(data, bool):
        return data
    elif data.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif data.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Local Topological Profile")
    parser.add_argument(
        "--dataset_name",
        choices=[
            "all",
            "DD",
            "NCI1",
            "PROTEINS_full",
            "ENZYMES",
            "IMDB-BINARY",
            "IMDB-MULTI",
            "REDDIT-BINARY",
            "REDDIT-MULTI-5K",
            "COLLAB",
        ],
        default="all",
        help="Dataset name, use 'all' to run the entire benchmark.",
    )
    parser.add_argument(
        "--degree_sum",
        type=ensure_bool,
        default=False,
        help="Add degree sum feature from LDP?",
    )
    parser.add_argument(
        "--shortest_paths",
        type=ensure_bool,
        default=False,
        help="Add shortest paths feature from LDP?",
    )
    parser.add_argument(
        "--edge_betweenness",
        type=ensure_bool,
        default=True,
        help="Add edge betweenness centrality proposed in LTP?",
    )
    parser.add_argument(
        "--jaccard_index",
        type=ensure_bool,
        default=True,
        help="Add Jaccard Index proposed in LTP?",
    )
    parser.add_argument(
        "--local_degree_score",
        type=ensure_bool,
        default=True,
        help="Add Local Degree Score proposed in LTP?",
    )
    parser.add_argument(
        "--n_bins",
        type=int,
        default=50,
        help="Number of bins for aggregation.",
    )
    parser.add_argument(
        "--normalization",
        choices=[
            "none",
            "graph",
            "dataset",
        ],
        default="none",
        help="Normalization scheme.",
    )
    parser.add_argument(
        "--aggregation",
        choices=[
            "histogram",
            "EDF",
        ],
        default="histogram",
        help="Aggregation scheme.",
    )
    parser.add_argument(
        "--log_degree",
        type=bool,
        default=False,
        help="Use log scale for degree features from LDP?",
    )
    parser.add_argument(
        "--model_type",
        choices=[
            "LinearSVM",
            "KernelSVM",
            "RandomForest",
        ],
        default="RandomForest",
        help="Classification algorithm to use.",
    )
    parser.add_argument(
        "--tune_feature_extraction_hyperparams",
        type=bool,
        default=False,
        help="Perform hyperparameter tuning for feature extraction?",
    )
    parser.add_argument(
        "--tune_model_hyperparams",
        type=bool,
        default=False,
        help="Perform hyperparameter tuning for classification model?",
    )
    parser.add_argument(
        "--use_features_cache",
        type=bool,
        default=True,
        help="Cache calculated features to speed up subsequent experiments?",
    )
    parser.add_argument(
        "--verbose",
        type=bool,
        default=False,
        help="Should print out verbose output?",
    )

    return parser.parse_args()

NORM_TAB = [-3, -2.75, -2.5, -2.25, -2]

if __name__ == "__main__":
    args = parse_args()

    if args.dataset_name == "all":
        datasets = DATASET_NAMES
    else:
        datasets = [args.dataset_name]

    attemtps = 1
    for algorithm_type, algorithm_name, powers in [
        (None, "NoSparsification", [None]),
        (sparsing.Random, "Rng", [(i + 1) / 100 for i in range(10)]),
        (sparsing.ArithmeticNorm, "ArithmeticNorm", [0.003]),
        (sparsing.GeometricNorm, "GeometricNorm", [(i + 1) / 1000 for i in range(20)]),
        (sparsing.HarmonicNorm, "HarmonicNorm", [(i + 1) / 1000 for i in range(20)]),
        (sparsing.Jaccard, "JaccardIndex", [0.01, 0.02, 0.04, 0.06, 0.08, 0.1]),
        (sparsing.CommonNeighbor, "CommonNeighborIndex", [0.01, 0.02, 0.04, 0.06, 0.08, 0.1]),
        (sparsing.PreferentialAttachment, "PreferentialAttachment", [0.01, 0.02, 0.04, 0.06, 0.08, 0.1]),
        (sparsing.AdamicAdar, "AdamicAdar", [0.01, 0.02, 0.04, 0.06, 0.08, 0.1]),
        (sparsing.AdjustedRand, "AdjustedRand", [0.01, 0.02, 0.04, 0.06, 0.08, 0.1]),
        (sparsing.AlgebraicDistance, "AlgebraicDistance", [0.01, 0.02, 0.04, 0.06, 0.08, 0.1]),
        (sparsing.Katz, "Katz", [-3, -2.5, -2, -1.5, -1]),
        (sparsing.Jaccard, "JaccardIndex", NORM_TAB),
        (sparsing.CommonNeighbor, "CommonNeighborIndex", NORM_TAB),
        (sparsing.PreferentialAttachment, "PreferentialAttachment", NORM_TAB),
        (sparsing.AdamicAdar, "AdamicAdar", NORM_TAB),
        (sparsing.AdjustedRand, "AdjustedRand", NORM_TAB),
        (sparsing.AlgebraicDistance, "AlgebraicDistance", NORM_TAB),
        (sparsing.Katz, "Katz", NORM_TAB),
        ]:
        for power in powers:
            algorithm = None
            if algorithm_type is not None:
                algorithm = algorithm_type(power=power)
            acc_mean, acc_stddev = None, None
            for attempt in range(attemtps):
                for dataset_name in datasets:
                    try:
                        os.system("rm -rf features_cache")
                    except:
                        pass
                    if attemtps > 1:
                        print(f'Attempt {attempt + 1} of {attemtps}')
                    print(dataset_name)
                    with open(f'exp-tune.csv', 'a') as f:
                        f.write(f'{dataset_name},{algorithm_name},{power},')
                    if attempt < 2:
                        acc_mean, acc_stddev = perform_experiment(
                            dataset_name=dataset_name,
                            degree_sum=args.degree_sum,
                            shortest_paths=args.shortest_paths,
                            edge_betweenness=args.edge_betweenness,
                            jaccard_index=args.jaccard_index,
                            local_degree_score=args.local_degree_score,
                            n_bins=args.n_bins,
                            normalization=args.normalization,
                            aggregation=args.aggregation,
                            log_degree=args.log_degree,
                            model_type=args.model_type,
                            tune_feature_extraction_hyperparams=args.tune_feature_extraction_hyperparams,
                            tune_model_hyperparams=args.tune_model_hyperparams,
                            use_features_cache=args.use_features_cache,
                            verbose=args.verbose,
                            sparsing_algorithm=algorithm,
                        )
                    else:
                        perform_experiment(
                            dataset_name=dataset_name,
                            degree_sum=args.degree_sum,
                            shortest_paths=args.shortest_paths,
                            edge_betweenness=args.edge_betweenness,
                            jaccard_index=args.jaccard_index,
                            local_degree_score=args.local_degree_score,
                            n_bins=args.n_bins,
                            normalization=args.normalization,
                            aggregation=args.aggregation,
                            log_degree=args.log_degree,
                            model_type=args.model_type,
                            tune_feature_extraction_hyperparams=args.tune_feature_extraction_hyperparams,
                            tune_model_hyperparams=args.tune_model_hyperparams,
                            use_features_cache=args.use_features_cache,
                            verbose=args.verbose,
                            sparsing_algorithm=algorithm,
                            break_after_feature_extraction=True,
                        )
                    print(f"Accuracy: {100 * acc_mean:.2f} +- {100 * acc_stddev:.2f}")
                    with open(f'exp-tune.csv', 'a') as f:
                        f.write(f'{round(100 * acc_mean, 2)},')
                        f.write(f'{round(100 * acc_stddev, 2)}\n')
