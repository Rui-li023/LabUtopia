"""Extract a logged initial scene without executing Python from the log.

The fixture restores scene state, not an unrecorded RNG continuation. A fixed
fixture is a development case, never an additional independent sample.
"""

import argparse
import ast
import hashlib
import io
import json
import re
import tokenize
from pathlib import Path

ANSI = re.compile(r"\x1b\[[0-9;]*m")
MARKER = "Set init state for episode "


class ArrayLiterals(ast.NodeTransformer):
    def visit_Call(self, node):
        if (
            not isinstance(node.func, ast.Name)
            or node.func.id != "array"
            or len(node.args) != 1
            or len(node.keywords) > 1
            or any(
                key.arg != "dtype" or not isinstance(key.value, ast.Name) or key.value.id not in {"float32", "float64"}
                for key in node.keywords
            )
        ):
            raise ValueError("Only literal NumPy array representations are allowed")
        return self.visit(node.args[0])


def literal_dict_prefix(text: str) -> str:
    lines = text.splitlines(keepends=True)
    depth = 0
    for token in tokenize.generate_tokens(io.StringIO(text).readline):
        if token.type != tokenize.OP:
            continue
        if token.string == "{":
            depth += 1
        elif token.string == "}":
            depth -= 1
            if depth == 0:
                line, column = token.end
                return "".join(lines[: line - 1]) + lines[line - 1][:column]
    raise ValueError("Incomplete initial-state dictionary")


def extract_states(log: str) -> list[dict]:
    cleaned = ANSI.sub("", log)
    states = []
    for match in re.finditer(re.escape(MARKER), cleaned):
        literal = literal_dict_prefix(cleaned[match.end() :])
        tree = ArrayLiterals().visit(ast.parse(literal, mode="eval"))
        state = ast.literal_eval(tree)
        if not isinstance(state, dict) or not isinstance(state.get("object_poses"), dict):
            raise ValueError("Missing object poses in recorded initial state")
        # Reject non-finite or non-JSON values; no constructors are executed.
        json.dumps(state, allow_nan=False)
        states.append(state)
    return states


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path)
    parser.add_argument("--attempt", type=int, required=True, help="Zero-based attempted episode, including failures")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    content = args.log.read_bytes()
    states = extract_states(content.decode(errors="replace"))
    if not 0 <= args.attempt < len(states):
        parser.error(f"Attempt must be within the {len(states)} recorded initial states")
    fixture = {
        "source_log": str(args.log.resolve()),
        "source_log_sha256": hashlib.sha256(content).hexdigest(),
        "attempt_index": args.attempt,
        "init_state": states[args.attempt],
        "limitation": "Scene state only; original RNG continuation was not logged. Development fixture, not acceptance.",
    }
    with args.out.open("x") as output:
        json.dump(fixture, output, indent=2, allow_nan=False)
    print(json.dumps({"fixture": str(args.out), "attempt": args.attempt, "recorded_states": len(states)}))


if __name__ == "__main__":
    main()
