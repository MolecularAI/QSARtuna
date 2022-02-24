import dataclasses
import json
from typing import Any, Dict


def mkdict(obj: Any) -> Dict:

    # To recursively convert nested dataclasses to dict, use json machinery.

    # https://stackoverflow.com/a/51286749
    class EnhancedJSONEncoder(json.JSONEncoder):
        def default(self, o):
            if dataclasses.is_dataclass(o):
                return dataclasses.asdict(o)
            return super().default(o)

    objstr = json.dumps(obj, cls=EnhancedJSONEncoder)
    objdict = json.loads(objstr)
    return objdict