# Test Report

This file records the checks run on the current project version.

## 1. Unit tests

Command:

```powershell
python -m unittest discover -s tests -q
```

Result: **95 tests passed**.

The suite covers answer formatting, input/output handling, the calculation tools, supported problem families, invalid or incomplete plans, timeouts, and fallback behavior. A few tests intentionally print simulated error messages; the final `OK` result means those test cases passed.

## 2. Command-line interface

Command:

```powershell
python main.py --input input.json --output output.json
```

Result: the program created a valid JSON output. For the included example input, the output was:

```json
{"id":"example_001","answer":"4"}
```

## 3. Public development questions

Command:

```powershell
python tests/run_public.py
```

The local `auto` run returned non-empty, correctly formatted output for all ten public questions. This is a format check only; it does not confirm that every answer is correct.

## 4. Model and hidden-set limitations

The full evaluation with the instructor's 27B model has not been completed. On the development computer, attempts to use smaller local models in `model-first` mode were affected by memory limits, timeouts, or an empty model candidate. In those cases, the agent could still return an answer through its local solving tools, so those results should not be counted as evidence of model reasoning accuracy.

The unit tests that use a fake model check the agent's request flow and error handling. They do not measure the reasoning quality of the real model. Accuracy on the hidden evaluation set is therefore **not known**.

## 5. Re-running the checks

From the project folder, run:

```powershell
python -m unittest discover -s tests -q
python main.py --input input.json --output output.json
python tests/run_public.py
```

For model testing, set `OLLAMA_MODEL` to the exact tag supplied for the evaluation and make sure that model is installed and available in the local Ollama service.
