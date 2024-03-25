import inspect
import json
from textwrap import dedent
from typing import Any, Tuple, Optional

import apischema
from apischema import schema
from apischema.json_schema import deserialization_schema, JsonSchemaVersion
from apischema.schemas import Schema

from optunaz.config.optconfig import Algorithm
from optunaz.config.optconfig import OptimizationConfig
from optunaz.descriptors import MolDescriptor, RdkitDescriptor
from optunaz.utils.schema import (
    replacekey,
    addsibling,
    delsibling,
    copytitle,
    replaceenum,
    addtitles,
)

from optunaz.utils.preprocessing.splitter import Splitter
from optunaz.utils.preprocessing.deduplicator import Deduplicator


def doctitle(doc: str) -> tuple[None, None] | tuple[str, str]:
    """Returns the first line of the docstring as the title."""

    if not doc:
        return None, None
    fulltitle, sep, rest = doc.partition("\n")  # https://stackoverflow.com/a/11833277
    title = fulltitle.rstrip(".")
    description = dedent(rest).strip()
    return title, description


def type_base_schema(tp: Any) -> Optional[Schema]:
    """Adds title and description from docstrings.

    See https://wyfo.github.io/apischema/0.16/json_schema/#base-schema
    """

    # Add only to Algorithms and Descriptors, ignore all the rest.
    # Otherwise, it starts adding description to classes like 'int' and 'list'.
    # Could be enough to only ignore built-in classes,
    # and add descriptions to all classes under optunaz/qputna package.
    if not inspect.isclass(tp):
        return None
    if not issubclass(tp, (Algorithm, MolDescriptor, Splitter, Deduplicator)):
        return None

    if not hasattr(tp, "__doc__"):
        return None

    title, description = doctitle(tp.__doc__)

    return schema(
        title=title,
        description=description,
    )


apischema.settings.base_schema.type = type_base_schema


def patch_schema_generic(schema):
    # Adds titles to all fields, even without schema(title=...).
    # TODO(alex): instead of calling addtitles, add proper titles to all fields.
    addtitles(schema)

    # Replace singleton enums with const.
    # For some reason, this was not needed in AZDock. A mystery.
    schema = replaceenum(schema)

    # Replace "anyOf" with "oneOf".
    schema = replacekey(schema)

    # Add "type": "object" to any elements that contain "oneOf": [...].
    schema = addsibling(schema)

    # Delete "type": "string" for "enum".
    schema = delsibling(schema, {"enum": "type"})

    # Delete most of the stuff for "const".
    schema = delsibling(schema, {"const": "type"})
    schema = delsibling(schema, {"const": "default"})
    schema = delsibling(schema, {"const": "title"})

    # Copy title from $refs into oneOf.
    schema = copytitle(schema, schema)

    return schema


def patch_schema_optunaz(schema):
    (
        schema.get("$defs", {})
        .get("MolData", {})
        .get("properties", {})
        .get("file_path", {})
    )["format"] = "uri"

    # Dataset
    (
        schema.get("$defs", {})
        .get("Dataset", {})
        .get("properties", {})
        .get("save_intermediate_files", {})
    )["const"] = True
    (
        schema.get("$defs", {})
        .get("Dataset", {})
        .get("properties", {})
        .get("intermediate_training_dataset_file", {})
    )["const"] = "{{run.path}}/intermediate_training_dataset_file.csv"
    (
        schema.get("$defs", {})
        .get("Dataset", {})
        .get("properties", {})
        .get("intermediate_test_dataset_file", {})
    )["const"] = "{{run.path}}/intermediate_test_dataset_file.csv"
    (
        schema.get("$defs", {})
        .get("Dataset", {})
        .get("properties", {})
        .pop("test_dataset_file", None)
    )
    # (
    #     schema.get("$defs", {})
    #     .get("Dataset", {})
    #     .get("properties", {})
    #     .get("training_dataset_file", {})
    # )["format"] = "file"
    (
        schema.get("$defs", {})
        .get("MolData", {})
        .get("properties", {})
        .get("file_path", {})
    )["format"] = "uri"

    # Root OptimizationConfig
    (
        schema.get("$defs", {})
        .get("OptimizationConfig", {})
        .get("properties", {})
        .pop("mode", None)
    )
    (
        schema.get("$defs", {})
        .get("OptimizationConfig", {})
        .get("properties", {})
        .pop("visualization", None)
    )
    # (
    #     schema.get("$defs", {})
    #     .get("OptimizationConfig", {})
    #     .get("properties", {})
    # )["mode"] = {
    #     "$ref": "#/$defs/ModelMode",
    #     "title": "Mode mode: regression or classification",
    #     "default": "regression"
    # }

    (schema.get("$defs", {}).get("OptimizationConfig", {}).get("properties", {}))[
        "slurmOptions"
    ] = {
        "title": "Slurm Options",
        "type": "object",
        "format": "slurm-summary",
        "description": "Resources and limits options - like CPU, Memory, time limits and reservations.",
        "properties": {
            "reservation": {
                "title": "Reinvent reservation.",
                "description": "Run the job on the nodes reserved for Reinvent.",
                "type": "boolean",
                "default": False,
            },
            "cpusPerTask": {
                "title": "Number of CPU cores",
                "description": "How many cores to request for the job."
                " Rule of thumb: use as many as the number of folds in cross-validation.",
                "type": "number",
                "default": 5,
            },
            "timeout": {
                "title": "Time limit for the job",
                "description": (
                    "How much time the job will have to finish."
                    " Acceptable time formats include"
                    " 'minutes', 'minutes:seconds', 'hours:minutes:seconds',"
                    " 'days-hours', 'days-hours:minutes'"
                    " and 'days-hours:minutes:seconds'."
                ),
                "type": "string",
                "pattern": (
                    "^("
                    "([0-9]+)"  # minutes
                    "|([0-9]+:[0-9]+)"  # minutes:seconds
                    "|([0-9]+:[0-9]+:[0-9]+)"  # hours:minutes:seconds
                    "|([0-9]+[-][0-9]+)"  # days-hours
                    "|([0-9]+[-][0-9]+:[0-9]+)"  # days-hours:minutes
                    "|([0-9]+[-][0-9]+:[0-9]+:[0-9]+)"  # days-hours:minutes:seconds
                    ")$"
                ),
                "default": "100:0:0",
            },
            "memPerCpu": {
                "title": "Memory allocation per CPU",
                "description": "How much memory should be allocated per core used. Default units are megabytes. Different units can be specified using the suffix [K|M|G|T]: e.g. 28G = 28 Gigabytes.",
                "type": "string",
                "pattern": "[0-9]+[K|M|G|T]?",
                "default": "4G",
            },
            "nodes": {"const": 1},
            "ntasks": {"const": 1},
        },
    }

    (
        schema.get("$defs", {})
        .get("Settings", {})
        .get("properties", {})
        .pop("mode", None)
    )

    (schema.get("$defs", {}).get("Settings", {}).get("properties", {}))["n_jobs"] = {
        "const": -1
    }

    (schema.get("$defs", {}).get("Settings", {}).get("properties", {}))[
        "track_to_mlflow"
    ] = {"const": False}

    (schema.get("$defs", {}).get("Settings", {}).get("properties", {}))[
        "optuna_storage"
    ] = {"const": "sqlite:///{{run.path}}/optuna_storage.sqlite"}

    (
        schema.get("$defs", {})
        .get("Settings", {})
        .get("properties", {})
        .pop("shuffle", None)
    )

    (
        schema.get("$defs", {})
        .get("Settings", {})
        .get("properties", {})
        .pop("direction", None)
    )

    (
        schema.get("$defs", {})
        .get("Settings", {})
        .get("properties", {})
        .pop("scoring", None)
    )

    (
        schema.get("$defs", {})
        .get("Settings", {})
        .get("properties", {})
        .pop("tracking_rest_endpoint", None)
    )

    (schema.get("$defs", {}).get("Settings", {}))["format"] = "collapsed"

    (
        schema.get("$defs", {})
        .get("ScaledDescriptorParameters", {})
        .get("properties", {})
        .pop("scaler", None)
    )

    (
        schema.get("$defs", {})
        .get("PhyschemDescriptors", {})
        .get("properties", {})
        .pop("parameters", {})
    )

    (
        schema.get("$defs", {})
        .get("JazzyDescriptors", {})
        .get("properties", {})
        .pop("parameters", {})
    )

    (
        schema.get("$defs", {})
        .get("SmilesAndSideInfoFromFileParams", {})
        .get("properties", {})
        .get("file", {})
    )["format"] = "uri"

    (
        schema.get("$defs", {})
        .get("Stratified", {})
        .get("properties", {})
        .pop("bins", {})
    )

    (
        schema.get("$defs", {})
        .get("Dataset", {})
        .get("properties", {})
        .get("training_dataset_file", {})
    )[
        "format"
    ] = "qsartuna-file"  # this format emphasized csv and sdf support with basic validation and automated column handling

    (
        schema.get("$defs", {})
        .get("Dataset", {})
        .get("properties", {})
        .get("response_type", {})
    )[
        "const"
    ] = None  # hide response_type on UI, not implemented yet

    (
        schema.get("$defs", {})
        .get("Dataset", {})
        .get("properties", {})
        .get("input_column", {})
    )[
        "format"
    ] = "single-string-select"  # for QSARtuna model this field is string, but in GUI we generate values for it based on qsartuna-file

    (
        schema.get("$defs", {})
        .get("Dataset", {})
        .get("properties", {})
        .get("response_column", {})
    )[
        "format"
    ] = "single-string-select"  # for QSARtuna model this field is string, but in GUI we generate values for it based on qsartuna-file

    (
        schema.get("$defs", {})
        .get("MolDescriptor", {})
        .get("anyOf", [])
        .remove({"$ref": "#/$defs/PrecomputedDescriptorFromFile"})
    )

    (
        schema.get("$defs", {})
        .get("RandomForestRegressorParams", {})
        .get("properties", {})
        .get("max_features", {})
        .get("items", {})
        .update(default="auto")
    )

    # Make GUI elements dependency for PTR.
    # First, copy "threshold" and "std" into "dependencies" object,
    # then delete them from the original place.
    (
        schema.get("$defs", {})
        .get("Dataset", {})
        .update(
            dependencies={
                "probabilistic_threshold_representation": {
                    "oneOf": [
                        {
                            "properties": {
                                "probabilistic_threshold_representation": {
                                    "enum": [False]
                                }
                            }
                        },
                        {
                            "properties": {
                                "probabilistic_threshold_representation": {
                                    "enum": [True]
                                },
                                "probabilistic_threshold_representation_threshold": (
                                    schema.get("$defs", {})
                                    .get("Dataset", {})
                                    .get("properties", {})
                                    .get(
                                        "probabilistic_threshold_representation_threshold",
                                        {},
                                    )
                                ),
                                "probabilistic_threshold_representation_std": (
                                    schema.get("$defs", {})
                                    .get("Dataset", {})
                                    .get("properties", {})
                                    .get(
                                        "probabilistic_threshold_representation_std",
                                        {},
                                    )
                                ),
                            },
                            "required": [
                                "probabilistic_threshold_representation_threshold",
                                "probabilistic_threshold_representation_std",
                            ],
                        },
                    ]
                },
            }
        )
    )
    (
        schema.get("$defs", {})
        .get("Dataset", {})
        .get("properties", {})
        .pop("probabilistic_threshold_representation_std", None)
    )
    (
        schema.get("$defs", {})
        .get("Dataset", {})
        .get("properties", {})
        .pop("probabilistic_threshold_representation_threshold", None)
    )

    # Make GUI elements dependency for log transform.
    # First, copy "log" and "negate" into "dependencies" object,
    # then delete them from the original place.
    (
        schema.get("$defs", {})
        .get("Dataset", {})
        .update(
            dependencies={
                "log_transform": {
                    "oneOf": [
                        {"properties": {"log_transform": {"enum": [False]}}},
                        {
                            "properties": {
                                "log_transform": {"enum": [True]},
                                "log_transform_base": (
                                    schema.get("$defs", {})
                                    .get("Dataset", {})
                                    .get("properties", {})
                                    .get(
                                        "log_transform_base",
                                        {},
                                    )
                                ),
                                "log_transform_negative": (
                                    schema.get("$defs", {})
                                    .get("Dataset", {})
                                    .get("properties", {})
                                    .get(
                                        "log_transform_negative",
                                        {},
                                    )
                                ),
                                "log_transform_unit_conversion": (
                                    schema.get("$defs", {})
                                    .get("Dataset", {})
                                    .get("properties", {})
                                    .get(
                                        "log_transform_unit_conversion",
                                        {},
                                    )
                                ),
                            },
                            "required": [
                                "log_transform_base",
                                "log_transform_negative",
                                "log_transform_unit_conversion",
                            ],
                        },
                    ]
                },
            }
        )
    )
    (
        schema.get("$defs", {})
        .get("Dataset", {})
        .get("properties", {})
        .pop("log_transform_base", None)
    )
    (
        schema.get("$defs", {})
        .get("Dataset", {})
        .get("properties", {})
        .pop("log_transform_negative", None)
    )
    (
        schema.get("$defs", {})
        .get("Dataset", {})
        .get("properties", {})
        .pop("log_transform_unit_conversion", None)
    )

    for alg in Algorithm.__subclasses__():
        (
            schema.get("$defs", {})
            .get(alg.__name__, {})
            .update(format="select-with-description")
        )

    for descriptor in RdkitDescriptor.__subclasses__():
        (
            schema.get("$defs", {})
            .get(descriptor.__name__, {})
            .update(format="select-with-description")
        )

    for descriptor in MolDescriptor.__subclasses__():
        (
            schema.get("$defs", {})
            .get(descriptor.__name__, {})
            .update(format="select-with-description")
        )
    return schema


def main():
    schema = deserialization_schema(
        OptimizationConfig, all_refs=True, version=JsonSchemaVersion.DRAFT_2019_09
    )
    schema = patch_schema_optunaz(schema)
    schema = patch_schema_generic(schema)
    print(json.dumps(schema, indent=2))


if __name__ == "__main__":
    main()
