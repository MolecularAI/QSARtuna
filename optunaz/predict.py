import argparse
import logging
import pathlib
import pickle
import sys

import pandas as pd


def main():
    parser = argparse.ArgumentParser(
        description="Predict responses for a given OptunaAZ model"
    )

    # fmt: off
    requiredNamed = parser.add_argument_group('required named arguments')
    requiredNamed.add_argument("--model-file", type=pathlib.Path, help="Model file name", required=True)

    parser.add_argument("--input-smiles-csv-file", type=pathlib.Path, help="Name of input CSV file with Input SMILES")
    parser.add_argument("--input-smiles-csv-column", type=str, help="Column name of SMILES column in input CSV file", default="SMILES")
    parser.add_argument("--output-prediction-csv-column", type=str, help="Name of output CSV file", default="Prediction")
    parser.add_argument("--output-prediction-csv-file", type=str, help="Name of output CSV file")
    # fmt: on

    args, leftovers = parser.parse_known_args()

    with open(args.model_file, "rb") as f:
        model = pickle.load(f)

    incolumn = args.input_smiles_csv_column
    outcolumn = args.output_prediction_csv_column

    if args.input_smiles_csv_file is not None:
        df = pd.read_csv(args.input_smiles_csv_file, skipinitialspace=True)
    elif len(leftovers) > 0:
        df = pd.DataFrame({incolumn: leftovers})
    else:
        print("No SMILES specified, exiting.", file=sys.stderr)
        exit(1)

    df[outcolumn] = model.predict_from_smiles(df[incolumn])
    if args.output_prediction_csv_file is None:
        args.output_prediction_csv_file = sys.stdout
    df.to_csv(args.output_prediction_csv_file, index=False, float_format="%g")
