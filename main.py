"""Official CLI: python main.py --input input.json --output output.json"""
import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

from agent import ReasoningAgent


def load_input(path):
    with Path(path).open("r", encoding="utf-8-sig") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError("Input must be one JSON object")
    problem_id, problem = data.get("id"), data.get("problem")
    if not isinstance(problem_id, str) or not problem_id.strip():
        raise ValueError("id must be a non-empty string")
    if not isinstance(problem, str) or not problem.strip():
        raise ValueError("problem must be a non-empty string")
    return problem_id.strip(), problem.strip()


def write_output(path, problem_id, answer):
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {"id": problem_id, "answer": answer if isinstance(answer, str) else ""}
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=str(destination.parent),
            prefix=destination.name + ".", suffix=".tmp", delete=False
        ) as handle:
            temporary = Path(handle.name)
            json.dump(payload, handle, ensure_ascii=False, separators=(",", ":"))
            handle.write("\n")
        os.replace(temporary, destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Offline local-Ollama reasoning agent")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    input_path, output_path = Path(args.input).resolve(), Path(args.output).resolve()
    if input_path == output_path:
        print("INPUT ERROR: input and output must differ", file=sys.stderr)
        return 2
    try:
        problem_id, problem = load_input(input_path)
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        print(f"INPUT ERROR: {exc}", file=sys.stderr)
        # Even a malformed input should never leave a stale/non-JSON output.
        try:
            write_output(output_path, "", "")
        except OSError as write_exc:
            print(f"OUTPUT ERROR: {write_exc}", file=sys.stderr)
        return 2

    answer = ""
    agent = None
    try:
        agent = ReasoningAgent()
        answer = agent.run(problem)
        if not isinstance(answer, str):
            answer = ""
    except Exception as exc:
        print(f"AGENT ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
    finally:
        if agent is not None:
            try:
                agent.close()
            except Exception:
                pass
    try:
        write_output(output_path, problem_id, answer)
    except OSError as exc:
        print(f"OUTPUT ERROR: {exc}", file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
