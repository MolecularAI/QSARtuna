import json


def loadJSON(path):
    with open(path) as file:
        json_string = file.read().replace("\r", "").replace("\n", "")
    return json.loads(json_string)
