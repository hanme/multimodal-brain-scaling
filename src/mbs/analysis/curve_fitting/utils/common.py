import argparse

from mbs.core import deep_update, get_md5_hash, load_yaml

__all__ = ["deep_update", "get_args", "get_md5_hash", "load_yaml"]


def get_args():
    argparser = argparse.ArgumentParser(description="Bootstrapping for scaling law curve fittings")

    argparser.add_argument(
        "--experiment-config",
        dest="experiment_config",
        type=str,
        default=None,
        help="Path to the yaml file containing the experiment configuration."
    )

    argparser.add_argument(
        "--results-csv",
        dest="results_csv",
        type=str,
        default="../../../visualize/scaling_laws.csv",
        help="Path to the csv file containing the experimental results."
    )

    argparser.add_argument(
        "--output-dir",
        dest="output_dir",
        type=str,
        default="./fitting_results",
        help="Path to the output directory."
    )
    argparser.add_argument(
        "--artifact-dir",
        dest="artifact_dir",
        type=str,
        default="./fitting_results",
        help="Path to bootstrapped results directory."
    )
    argparser.add_argument(
        "--experiment-name",
        dest="experiment_name",
        type=str,
        default=None,
        help="Name of the experiment."
    )
    argparser.add_argument(
        "--target-metric",
        dest="target_metric",
        type=str,
        default=None,
        help="Name of the target metric. If not provided, it will be taken from the configuration file."
    )
    argparser.add_argument(
        "--num-workers",
        dest="num_workers",
        type=int,
        default=8,
        help="Number of workers to use for multiprocessing (default is 8)."
    )
    argparser.add_argument(
        "--overwrite",
        dest="overwrite",
        action='store_true',
        help="Whether to overwrite existing files."
    )
    argparser.add_argument(
        "--verbose",
        dest="verbose",
        action='store_true',
        help="Whether to print verbose output."
    )
    argparser.add_argument(
        "--num-bootstraps",
        dest="num_bootstraps",
        type=int,
        default=None,
        help="Number of bootstraps to perform. The default is None, which is the intended use for using the value from the configuration file." \
                "If a value is provided, it will override the value from the configuration file." \
                "Useful for testing purposes."
    )

    return argparser
