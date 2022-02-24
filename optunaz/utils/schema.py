import collections
import copy
from typing import Union, Dict, List, Any


def create_dependency(key, value):
    newkey = f"is_enabled_{key}"
    title = value.get("title", key)
    newvalue = {
        "title": f"Enable {title}",
        "default": "No",
        "enum": ["No", "Yes"],
    }
    dep = {
        "oneOf": [
            {"properties": {newkey: {"enum": ["No"]}}},
            {"properties": {newkey: {"enum": ["Yes"]}, key: value}, "required": [key]},
        ]
    }

    return newkey, newvalue, dep


def remove_schema_properties(schema: Dict[str, Any], properties: List[str]) -> None:
    schemaprops = schema.get("properties", {})
    for prop in properties:
        schemaprops.pop(prop, None)


def add_boolean_guards_for_schema_properties(
    schema: Dict[str, Any], properties: List[str]
) -> None:
    if "dependencies" not in schema:
        schema["dependencies"] = {}
    props = schema.get("properties", {})
    new_props = {}
    for key, value in props.items():
        if key in properties:
            newkey, newvalue, dep = create_dependency(key, value)
            new_props[newkey] = newvalue
            schema["dependencies"][newkey] = dep
        else:
            new_props[key] = value

    schema["properties"] = new_props


def replacekey(input: Union[Dict, List, object]) -> Any:
    replacements = {"anyOf": "oneOf"}
    if isinstance(input, dict):
        # For every key in the dict, get either a replacement, or the key itself.
        # Call this function recursively for values.
        return {replacements.get(k, k): replacekey(v) for k, v in input.items()}
    elif isinstance(input, list):
        return [replacekey(item) for item in input]
    else:
        return input


def replacevalue(input: Union[Dict, List, object]) -> Any:
    if isinstance(input, dict):
        return {k: replacevalue(input[k]) for k in input}
    elif isinstance(input, list):
        return [replacevalue(item) for item in input]
    else:
        replacements = {"integer": "number"}
        return replacements.get(input, input)


def addsibling(input: Union[Dict, List, object]) -> Any:
    if isinstance(input, dict):
        d = {k: addsibling(input[k]) for k in input}
        if "oneOf" in input:
            d["type"] = "object"
        return d
    elif isinstance(input, list):
        return [addsibling(item) for item in input]
    else:
        return input


def delsibling(input: Union[Dict, List, object], siblings: Dict[str, str]) -> Any:
    if isinstance(input, dict):
        d = {k: delsibling(input[k], siblings) for k in input}
        for key, value in siblings.items():
            if key in input:
                d.pop(value, None)
        return d
    elif isinstance(input, list):
        return [delsibling(item, siblings) for item in input]
    else:
        return input


def getref(path: str, context: Dict):
    """Recursively returns nested items from a dict."""
    if not path.startswith("#/"):
        raise ValueError("ref path does not start with #/")

    items = path[2:].split("/")

    def recursive_get(keys, structure):
        if len(keys) == 0:
            return structure
        else:
            return recursive_get(keys[1:], structure[keys[0]])

    return recursive_get(items, context)


def copytitle(input, context):
    """Copies "title" from "$ref" into oneOf."""
    if isinstance(input, dict):
        output = {}
        for key in input:
            if key == "oneOf":
                initems = input[key]
                outitems = []
                for initem in initems:
                    outitem = copy.deepcopy(initem)
                    if "$ref" in initem and not "title" in initem:
                        ref = getref(initem["$ref"], context)
                        default_title = initem["$ref"].split("/")[-1]
                        outitem["title"] = ref.get("title", default_title)
                    outitems.append(outitem)
                output[key] = outitems
            else:
                output[key] = copytitle(input[key], context)

        return output
    elif isinstance(input, list):
        return [copytitle(item, context) for item in input]
    else:
        return input


def replaceenum(input: Union[Dict, List, object]) -> Any:
    """Replace singleton enums with const."""
    if isinstance(input, collections.Mapping):
        d1 = {
            k: replaceenum(v)
            for k, v in input.items()
            if not (k == "enum" and isinstance(v, list) and len(v) == 1)
        }
        d2 = {
            "const": v[0]
            for k, v in input.items()
            if (k == "enum" and isinstance(v, list) and len(v) == 1)
        }
        return {**d1, **d2}  # Merge two dicts: https://stackoverflow.com/a/26853961
    elif isinstance(input, list):
        return [replaceenum(item) for item in input]
    else:
        return input


def addtitles(schema):
    if isinstance(schema, dict):
        for name, prop in schema.get("properties", {}).items():
            prop["title"] = prop.get("title", name.capitalize())  # or whatever
        for value in schema.values():
            addtitles(value)
    elif isinstance(schema, list):
        for elt in schema:
            addtitles(elt)