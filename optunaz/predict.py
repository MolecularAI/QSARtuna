import logging
import argparse
import pathlib
import pickle
import sys

import pandas as pd


def main():
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(
        description="Predict responses for a given OptunaAZ model"
    )

    # fmt: off
    requiredNamed = parser.add_argument_group('required named arguments')
    requiredNamed.add_argument("--model-file", type=pathlib.Path, help="Model file name", required=True)

    parser.add_argument("--input-smiles-csv-file", type=pathlib.Path, help="Name of input CSV file with Input SMILES")
    parser.add_argument("--input-smiles-csv-column", type=str, help="Column name of SMILES column in input CSV file", default="SMILES")
    parser.add_argument("--input-aux-column", type=str, help="Column name of auxiliary descriptors column in input CSV file", default=None)
    parser.add_argument("--output-prediction-csv-column", type=str, help="Column name of prediction column in output CSV file", default="Prediction")
    parser.add_argument("--output-prediction-csv-file", type=str, help="Name of output CSV file")
    parser.add_argument("--predict-uncertainty", action="store_true", help="Predict with uncertainties (model must provide this functionality)")
    parser.add_argument("--predict-explain", action="store_true", help="Predict with SHAP or ChemProp explainability")
    parser.add_argument("--uncertainty_quantile", type=float, help="Apply uncertainty threshold to predictions", default=None)
    # fmt: on

    args, leftovers = parser.parse_known_args()
    assert not (
        args.predict_uncertainty and args.predict_explain
    ), "Cannot provide both uncertainty and explainability at the same time"
    if args.uncertainty_quantile is not None:
        assert (
            args.predict_uncertainty
        ), "Must predict with uncertainty to perform uncertainty_quantile"
        assert (
            0.0 <= args.uncertainty_quantile <= 1.0
        ), "uncertainty_quantile must range 0.0-1.0"

    with open(args.model_file, "rb") as f:
        model = pickle.load(f)

    if model.metadata["buildconfig"].get("data").get("aux_column"):
        assert (
            args.input_aux_column
        ), "Model was trained with auxiliary data, please provide an input auxiliary column"

    if model.aux_transform:
        assert (
            args.input_aux_column
        ), "Input auxiliary column required since it appears the model was trained with an auxiliary transform"

    incolumn = args.input_smiles_csv_column
    outcolumn = args.output_prediction_csv_column

    if args.input_smiles_csv_file is not None:
        df = pd.read_csv(args.input_smiles_csv_file, skipinitialspace=True)
    elif len(leftovers) > 0:
        df = pd.DataFrame({incolumn: leftovers})
    else:
        print("No SMILES specified, exiting.", file=sys.stderr)
        exit(1)
    if args.input_aux_column is not None:
        aux = df[args.input_aux_column]
    else:
        aux = None

    if args.predict_explain:
        df = model.predict_from_smiles(
            df[incolumn],
            explain=args.predict_explain,
            aux=aux,
            aux_transform=model.aux_transform,
        )
    else:
        df[outcolumn] = model.predict_from_smiles(
            df[incolumn], aux=aux, aux_transform=model.aux_transform
        )
        if args.predict_uncertainty:
            df[f"{outcolumn}_uncert"] = model.predict_from_smiles(
                df[incolumn],
                uncert=args.predict_uncertainty,
                aux=aux,
                aux_transform=model.aux_transform,
            )
            if args.uncertainty_quantile is not None:
                uncert_thr = df[f"{outcolumn}_uncert"].quantile(
                    args.uncertainty_quantile
                )
                df = df[df[f"{outcolumn}_uncert"] > uncert_thr].sort_values(
                    f"{outcolumn}_uncert", ascending=False
                )
    if args.output_prediction_csv_file is None:
        args.output_prediction_csv_file = sys.stdout
    df.to_csv(args.output_prediction_csv_file, index=False, float_format="%g")
