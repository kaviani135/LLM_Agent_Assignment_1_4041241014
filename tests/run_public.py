"""Optional development runner for the ten public questions; not used at grading time."""

import argparse
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
QUESTIONS = Path(__file__).with_name("public_questions.json")


def save(path, records):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", type=int, nargs="*", default=None)
    parser.add_argument("--timeout", type=float, default=420.0,
                        help="Per-question CLI timeout in seconds")
    parser.add_argument("--output", default="public_results.json")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    cases = json.loads(QUESTIONS.read_text(encoding="utf-8-sig"))
    if not isinstance(cases, list) or len(cases) != 10:
        raise SystemExit("Public question file must contain exactly 10 cases")

    selected = set(args.only) if args.only is not None else set(range(1, 11))
    if not selected or any(n < 1 or n > 10 for n in selected):
        parser.error("--only accepts question numbers 1..10")

    chosen = [(n, case) for n, case in enumerate(cases, 1) if n in selected]
    if args.dry_run:
        for n, case in chosen:
            print(f"{n:02d} {case['id']}: {case.get('topic','')}")
        print(f"Validated {len(chosen)} selected questions. No model calls.")
        return 0

    output = Path(args.output)
    records = []
    print(f"Running {len(chosen)} selected question(s). Results are saved after each case.",
          flush=True)

    for n, case in chosen:
        print(f"[{n:02d}/10] {case['id']}: running...", flush=True)
        with tempfile.TemporaryDirectory(prefix="agent_public_") as temp:
            inp, out = Path(temp)/"input.json", Path(temp)/"output.json"
            inp.write_text(json.dumps(
                {"id":case["id"],"problem":case["problem"]},
                ensure_ascii=False), encoding="utf-8")
            started = time.monotonic()
            try:
                proc = subprocess.run(
                    [sys.executable, str(ROOT/"main.py"), "--input", str(inp),
                     "--output", str(out)],
                    cwd=str(ROOT), capture_output=True, text=True,
                    encoding="utf-8", errors="replace", timeout=args.timeout,
                )
                elapsed = round(time.monotonic()-started, 1)
                if proc.returncode != 0:
                    record = {"id":case["id"],"answer":"","status":"ERROR",
                              "seconds":elapsed,"stderr":proc.stderr.strip()[-1000:]}
                else:
                    try:
                        result = json.loads(out.read_text(encoding="utf-8-sig"))
                        answer = result.get("answer") if isinstance(result,dict) else None
                        if result.get("id") != case["id"] or not isinstance(answer,str):
                            raise ValueError("Output does not match official JSON contract")
                        record = {"id":case["id"],"answer":answer,
                                  "status":"OK" if answer.strip() else "EMPTY",
                                  "seconds":elapsed}
                    except Exception as exc:
                        record = {"id":case["id"],"answer":"","status":"BAD_OUTPUT",
                                  "seconds":elapsed,"error":str(exc)}
                    if proc.stderr.strip():
                        record["stderr"] = proc.stderr.strip()[-1000:]
            except subprocess.TimeoutExpired:
                record = {"id":case["id"],"answer":"","status":"TIMEOUT",
                          "seconds":round(time.monotonic()-started,1)}
            except KeyboardInterrupt:
                save(output, records)
                print(f"\nStopped. Completed results saved to {output.resolve()}")
                return 130

        records.append(record)
        save(output, records)  # persist progress immediately
        print(f"  {record['status']} | answer={record['answer']!r} | "
              f"{record['seconds']:.1f}s", flush=True)

    ok = sum(r["status"] == "OK" for r in records)
    print(f"\nSaved: {output.resolve()}")
    print(f"Non-empty valid-format responses: {ok}/{len(records)} (not an accuracy score).")
    return 0 if ok == len(records) else 1


if __name__ == "__main__":
    raise SystemExit(main())
