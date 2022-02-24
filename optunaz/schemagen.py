import json

from apischema.json_schema import deserialization_schema, JsonSchemaVersion

from optunaz.config.optconfig import OptimizationConfig
from optunaz.utils.schema import (
    replacekey,
    addsibling,
    delsibling,
    copytitle,
    replaceenum,
    addtitles,
)


def patch_schema_generic(schema):

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
    # (
    #     schema.get("$defs", {})
    #     .get("Dataset", {})
    #     .get("properties", {})
    #     .pop("test_dataset_file", None)
    # )
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

    drop_algs = {"PLS", "RandomForest", "XGBregressor"}
    drop_refs = {f"#/$defs/{alg}" for alg in drop_algs}
    alg_items = (
        schema.get("$defs", {})
        .get("OptimizationConfig", {})
        .get("properties", {})
        .get("algorithms", {})
        .get("items", {})
    )
    algs = alg_items.get("anyOf", {})
    alg_items["anyOf"] = [alg for alg in algs if alg["$ref"] not in drop_refs]

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
        .get("Stratified", {})
        .get("properties", {})
        .pop("bins", {})
    )

    (
        schema.get("$defs", {})
        .get("Dataset", {})
        .get("properties", {})
        .get("training_dataset_file", {})
    )["format"] = "uri"

    (
        schema.get("$defs", {})
        .get("MolDescriptor", {})
        .get("anyOf", [])
        .remove({"$ref": "#/$defs/PrecomputedDescriptorFromFile"})
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
