"""Plots results (losses and accuracies) using results yaml files."""
from typing import Any, List, Dict
import pathlib

import pandas as pd
import plotly.express as px
import yaml


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(prog="Classifier Results Plotter")
    parser.add_argument(
        "--results_dir", help="path to image directory", default="./results/"
    )

    args = parser.parse_args()

    # get all results files
    results_files = pathlib.Path(args.results_dir).glob("*.yaml")

    # load each result into dataframe
    results_records: List[Dict[str, Any]] = []
    for result_fp in results_files:
        with open(result_fp, "r") as result_file:
            result_record = yaml.load(result_file, Loader=yaml.SafeLoader)

        model_name, epoch = result_fp.name.split("_")
        epoch = int(epoch.split(".")[0])

        results_records.append({"Model": model_name, "Epoch": epoch, **result_record})

    results_df = pd.DataFrame(results_records)

    # create grouped bar charts
    accuracy_fig = px.bar(
        results_df,
        x="Model",
        facet_row="Epoch",
        y="accuracy",
        category_orders={"Epoch": [15, 30, 50]},
        title="Accuracy of Image Classifiers by Model, Epoch",
        template="presentation",
        color="Model",
    )
    accuracy_fig.show()

    loss_fig = px.bar(
        results_df,
        x="Model",
        facet_row="Epoch",
        y="loss",
        category_orders={"Epoch": [15, 30, 50]},
        title="Loss of Image Classifiers by Model, Epoch",
        template="presentation",
        color="Model",
    )
    loss_fig.show()
