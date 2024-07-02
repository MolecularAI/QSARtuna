import logging
import argparse
import pathlib
import pickle
import sys

import pandas as pd


class ArgsError(Exception):
    "Thrown when there is an issue with basic args at inference time"


class UncertaintyError(Exception):
    "Thrown when uncertainty parameters are not set correctly at inference"


class AuxCovariateMissing(Exception):
    "Thrown when a model is trained using Auxiliary (covariate) data which is not supplied at inference"


class PrecomputedError(Exception):
    "Raised when a model is trained with precomputed descriptor not supplied at runtime or due to a missing argument"


def validate_args(args):
    try:
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
    except AssertionError as e:
        raise ArgsError(e)


def validate_uncertainty(args, model):
    if args.predict_uncertainty:
        if not hasattr(model.predictor, "predict_uncert"):
            raise UncertaintyError("Uncertainty not availble for this model")


def check_precomp_args(args):
    try:
        assert (
            args.input_precomputed_file is not None
        ), "Must supply precomputed descriptor parameters"
        assert (
            args.input_precomputed_input_column
        ), "Must supply input column for precomputed descriptor"
        assert (
            args.input_precomputed_response_column
        ), "Must supply response column for precomputed descriptor"
    except AssertionError as e:
        raise PrecomputedError(e)


def validate_set_precomputed(args, model):
    descriptor_str = model.descriptor.name
    if descriptor_str == "CompositeDescriptor":
        precomp_idx = [
            idx
            for idx, d in enumerate(model.descriptor.parameters.descriptors)
            if d.name == "PrecomputedDescriptorFromFile"
        ]
        if len(precomp_idx) == 0:
            logging.warning(
                f"{descriptor_str} has no Precomputed descriptors... ignoring precomputed descriptor parameters"
            )
            return model
        else:
            if len(precomp_idx) > 1:
                raise PrecomputedError(
                    "Inference for > precomputed descriptor not currently available"
                )
            check_precomp_args(args)
            params = model.descriptor.parameters.descriptors[precomp_idx[0]].parameters
            params.file = args.input_precomputed_file
            params.input_column = args.input_precomputed_input_column
            params.response_column = args.input_precomputed_response_column
    elif descriptor_str != "PrecomputedDescriptorFromFile":
        logging.warning(
            f"Model was trained using {descriptor_str}... ignoring precomputed descriptor parameters"
        )
        return model
    else:  # must be precomputed
        check_precomp_args(args)
        params = model.descriptor.parameters
        params.file = args.input_precomputed_file
        params.input_column = args.input_precomputed_input_column
        params.response_column = args.input_precomputed_response_column
    return model


def validate_aux(args, model):
    try:
        if model.metadata["buildconfig"].get("data").get("aux_column"):
            assert (
                args.input_aux_column
            ), "Model was trained with auxiliary data, please provide an input auxiliary column"
        if model.aux_transform:
            assert (
                args.input_aux_column
            ), "Input auxiliary column required since it appears the model was trained with an auxiliary transform"
    except AssertionError as e:
        raise AuxCovariateMissing(e)


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
    parser.add_argument("--input-aux-column", type=str, help="Column name of auxiliary descriptors in input CSV file", default=None)
    parser.add_argument("--input-precomputed-file", type=str, help="Filename of precomputed descriptors input CSV file", default=None)
    parser.add_argument("--input-precomputed-input-column", type=str, help="Column name of precomputed descriptors identifier", default=None)
    parser.add_argument("--input-precomputed-response-column", type=str, help="Column name of precomputed descriptors response column", default=None)
    parser.add_argument("--output-prediction-csv-column", type=str, help="Column name of prediction column in output CSV file", default="Prediction")
    parser.add_argument("--output-prediction-csv-file", type=str, help="Name of output CSV file")
    parser.add_argument("--predict-uncertainty", action="store_true", help="Predict with uncertainties (model must provide this functionality)")
    parser.add_argument("--predict-explain", action="store_true", help="Predict with SHAP or ChemProp explainability")
    parser.add_argument("--uncertainty_quantile", type=float, help="Apply uncertainty threshold to predictions", default=None)
    # fmt: on

    args, leftovers = parser.parse_known_args()
    validate_args(args)

    with open(args.model_file, "rb") as f:
        model = pickle.load(f)

    validate_uncertainty(args, model)
    model = validate_set_precomputed(args, model)
    validate_aux(args, model)

    incolumn = args.input_smiles_csv_column
    outcolumn = args.output_prediction_csv_column

    if args.input_smiles_csv_file is not None:
        df = pd.read_csv(args.input_smiles_csv_file, skipinitialspace=True)
    elif len(leftovers) > 0:
        df = pd.DataFrame({incolumn: leftovers})
    else:
        logging.info("No SMILES specified, exiting.")
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
        if args.predict_uncertainty:
            pred, unc_pred = model.predict_from_smiles(
                df[incolumn],
                uncert=args.predict_uncertainty,
                aux=aux,
                aux_transform=model.aux_transform,
            )
            df[f"{outcolumn}"] = pred
            df[f"{outcolumn}_uncert"] = unc_pred
            if args.uncertainty_quantile is not None:
                uncert_thr = df[f"{outcolumn}_uncert"].quantile(
                    args.uncertainty_quantile
                )
                df = df[df[f"{outcolumn}_uncert"] > uncert_thr].sort_values(
                    f"{outcolumn}_uncert", ascending=False
                )
        else:
            df[outcolumn] = model.predict_from_smiles(
                df[incolumn], aux=aux, aux_transform=model.aux_transform
            )
    if args.output_prediction_csv_file is None:
        args.output_prediction_csv_file = sys.stdout
    df.to_csv(args.output_prediction_csv_file, index=False, float_format="%g")
