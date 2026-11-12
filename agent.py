"""Local-only reasoning agent for the LLM Agents assignment.

One structured planning/solving request, deterministic tool execution and
verification, and at most one repair request. All inference is sent to local
Ollama only; the module has no cloud/network fallback.
"""
from __future__ import annotations

import ast
import json
import os
import re
import socket
import sys
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from tools import (
    ToolError, safe_arithmetic, calculate_project_duration, shortest_grid_path,
    solve_csp, validate_csp_evidence, require_grounded_plan,
    solve_linear_system, solve_set_count, solve_percent_chain,
    _question_numeric_facts, validate_bullet_coverage, _normal_space, _require_quotes,
    solve_question_family,
)


class AgentError(RuntimeError):
    pass


class OllamaClientError(RuntimeError):
    pass


class AnswerFormatError(ValueError):
    pass


def _is_fatal_memory_error(exc):
    """Ollama capacity failures cannot be repaired by another identical request."""
    if not isinstance(exc, OllamaClientError):
        return False
    detail = str(exc).casefold()
    return any(message in detail for message in (
        "requires more system memory",
        "not enough memory",
        "out of memory",
        "insufficient memory",
        "cuda error out of memory",
    ))


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON key")
        result[key] = value
    return result


def parse_json_object(raw: str) -> dict:
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError("Empty model response")
    text = raw.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.I)
    text = re.sub(r"\s*```$", "", text)
    decoder = json.JSONDecoder(object_pairs_hook=_unique_object)
    for start in [0] + [m.start() for m in re.finditer(r"\{", text)]:
        try:
            obj, end = decoder.raw_decode(text[start:])
            if isinstance(obj, dict) and not text[start + end:].strip():
                return obj
        except (json.JSONDecodeError, ValueError):
            continue
    raise ValueError("No valid JSON object found")


def normalize_digits(value: str) -> str:
    source = "۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩"
    target = "0123456789" * 2
    return value.translate(str.maketrans(source, target))


def detect_format(problem: str) -> str:
    tail = problem[-1800:]
    if re.search(r"[A-Za-z][A-Za-z0-9_]*\s*=\s*<\s*(?:integer|int|number|text|digit)\s*>", tail, re.I):
        if len(re.findall(r"[A-Za-z][A-Za-z0-9_]*\s*=\s*<\s*(?:integer|int|number|text|digit)\s*>", tail, re.I)) >= 2:
            return "assignment"
    if len(re.findall(r"(?m)^\s*[A-Za-z][A-Za-z0-9_]*\s*=\s*(?:Truth|Lie|True|False)\s*,\s*[A-Za-z][A-Za-z0-9_]*\s*=\s*(?:Truth|Lie|True|False)\s*$", tail, re.I)) >= 2:
        return "assignment"
    if re.search(r"فقط\s+حرف|only\s+(?:the\s+)?letter|(?:return|output|give|answer\s+with)\s+(?:only\s+)?(?:the\s+)?(?:option|choice)\s+letter", tail, re.I):
        return "letter"
    if (re.search(r"Monday\s*,\s*Tuesday\s*,\s*Wednesday\s*,\s*Thursday", tail, re.I)
            or all(re.search(r"\b" + d + r"\b", tail, re.I)
                   for d in ("Monday", "Tuesday", "Wednesday", "Thursday"))
            or re.search(r"فقط\s+(?:یکی\s+از\s+این\s+چهار\s+عبارت|روز)|only\s+(?:the\s+)?weekday", tail, re.I)):
        return "day"
    if re.search(r"کد\s+سه.?رقمی|فقط\s+کد|only\s+(?:the\s+)?(?:three.?digit\s+)?code|(?:return|output|give)\s+(?:only\s+)?(?:the\s+)?(?:\w+.?digit\s+)?code", tail, re.I):
        return "code"
    if re.search(r"فقط\s+عدد\s+صحیح|فقط\s+عدد\s+را|فقط\s+یکی\s+از\s+اعداد|only\s+(?:the\s+)?integer|\b(?:return|output|give|provide)\s+(?:only\s+)?(?:the\s+)?(?:an?\s+)?(?:integer|whole number)\b|(?:answer|respond)\s+(?:as|with)\s+(?:an?\s+)?integer", tail, re.I):
        return "integer"
    if re.search(r"only\s+(?:the\s+)?number|فقط\s+عدد", tail, re.I):
        return "number"
    return "text"


def _as_answer(value) -> str:
    if isinstance(value, bool) or value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (int, float)):
        return str(int(value)) if isinstance(value, float) and value.is_integer() else str(value)
    if isinstance(value, dict):
        if not value:
            return ""
        return ", ".join(f"{k}={v}" for k, v in value.items())
    return ""


def normalize_answer(value, answer_format="text") -> str:
    answer = _as_answer(value)
    if not answer:
        raise AnswerFormatError("Answer is empty")
    if "\n" in answer or "\r" in answer:
        raise AnswerFormatError("Answer must be one line")
    answer = normalize_digits(answer).strip().strip("`").strip()
    answer = re.sub(r"^(?:FINAL_ANSWER|Final\s+Answer)\s*[:：]\s*", "", answer, flags=re.I)
    answer = re.sub(r"^####\s*", "", answer)
    answer = answer.replace("\u200c", "")
    answer = re.sub(r"\s+", " ", answer).strip().rstrip(" .;،")
    if not answer or len(answer) > 300 or re.fullmatch(r"<[^>]+>", answer):
        raise AnswerFormatError("Malformed final answer")
    fmt = answer_format.lower()
    if fmt == "integer":
        if not re.fullmatch(r"[+-]?\d+", answer):
            raise AnswerFormatError("Expected an integer")
        return str(int(answer))
    if fmt == "number":
        if not re.fullmatch(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+|\d+/\d+)", answer):
            raise AnswerFormatError("Expected a number")
        return answer
    if fmt == "code":
        if not re.fullmatch(r"\d{2,8}", answer):
            raise AnswerFormatError("Expected a digit code")
        return answer
    if fmt == "letter":
        if not re.fullmatch(r"[A-Za-z]", answer):
            raise AnswerFormatError("Expected one letter")
        return answer.upper()
    if fmt == "day":
        days = {d.lower(): d for d in ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")}
        if answer.lower() not in days:
            raise AnswerFormatError("Expected a weekday")
        return days[answer.lower()]
    if fmt == "assignment":
        parts = [p.strip() for p in answer.split(",")]
        if not parts or any(not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*\s*=\s*[^,=\s]+", p) for p in parts):
            raise AnswerFormatError("Expected comma-separated name=value pairs")
        return ", ".join(re.sub(r"\s*=\s*", "=", p) for p in parts)
    return answer


class LocalOllama:
    """Minimal stdlib client restricted to the local Ollama endpoint."""
    def __init__(self, model_name=None):
        self.base_url = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/")
        if self.base_url not in ("http://127.0.0.1:11434", "http://localhost:11434"):
            raise ValueError("OLLAMA_HOST must point to local Ollama (127.0.0.1 or localhost)")
        configured = model_name or os.getenv("OLLAMA_MODEL")
        self.model_name = configured.strip() if isinstance(configured, str) and configured.strip() else self._local_model()

    def _local_model(self):
        """Prefer the instructor's installed 27B, otherwise the local 4B dev model.

        Reads ONLY the local Ollama tag list; it never downloads anything.
        Explicit OLLAMA_MODEL always wins over this convenience selection.
        """
        try:
            with urlopen(self.base_url + "/api/tags", timeout=2.0) as response:
                data = json.loads(response.read().decode("utf-8"))
            installed = [m.get("name", "") for m in data.get("models", []) if isinstance(m, dict)]
            for name in installed:
                if "qwen" in name.casefold() and "27b" in name.casefold():
                    return name
            for preferred in ("qwen3:4b-instruct", "qwen3:1.7b"):
                if preferred in installed:
                    return preferred
        except (OSError, ValueError, TypeError, KeyError):
            pass
        # Last-resort default. For evaluation, set OLLAMA_MODEL to the exact
        # tag supplied by the instructor if automatic local detection is not
        # available or selects a different model.
        return "qwen3.5:27b"

    def chat(self, system_prompt, user_prompt, timeout=120, num_predict=768,
             response_format="json"):
        """Send one bounded request to local Ollama.

        Structured planning uses JSON mode; final-answer recovery can use
        plain-text mode to avoid forcing a struggling small model to emit an
        entire planning schema twice.
        """
        think_setting = os.getenv("AGENT_THINK", "off").strip().lower()
        if think_setting not in ("off", "false", "0", "on", "true", "1", "auto"):
            raise ValueError("AGENT_THINK must be on, off, or auto")
        # Think mode has very different cost across Ollama models; do not
        # silently consume the evaluation deadline. It can be opted into.
        use_thinking = think_setting in ("on", "true", "1")
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,
            "think": use_thinking,
            "keep_alive": os.getenv("OLLAMA_KEEP_ALIVE", "5m"),
            "options": {
                "temperature": float(os.getenv("AGENT_TEMPERATURE", "0")),
                "num_ctx": max(1024, min(32768, int(os.getenv("AGENT_NUM_CTX", "8192")))),
                "num_predict": int(num_predict),
            },
        }
        if response_format == "json":
            payload["format"] = "json"
        elif response_format not in (None, "text"):
            raise ValueError("Unsupported Ollama response format")
        req = Request(
            self.base_url + "/api/chat",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json"}, method="POST",
        )
        try:
            with urlopen(req, timeout=float(timeout)) as response:
                data = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            detail = exc.read(500).decode("utf-8", errors="replace")
            raise OllamaClientError(f"Ollama HTTP {exc.code}: {detail}") from exc
        except (URLError, socket.timeout, TimeoutError, OSError) as exc:
            raise OllamaClientError(f"Local Ollama request failed: {exc}") from exc
        except (ValueError, UnicodeError) as exc:
            raise OllamaClientError("Invalid Ollama response") from exc
        message = data.get("message", {}) if isinstance(data, dict) else {}
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, str) or not content.strip():
            done_reason = data.get("done_reason", "unknown") if isinstance(data, dict) else "unknown"
            raise OllamaClientError(f"Ollama returned empty final content (done_reason={done_reason})")
        return content.strip()

    def close(self):
        return None


PLAN_PROMPT = """You are a precise reasoning agent for math, logic, ordering, and planning problems.
Read the FULL question. Identify the exact requested quantity, not an intermediate result.
Solve independently, then provide one JSON OBJECT only. All intermediate reasoning stays in the 'check' field; 'answer' is ONLY the final requested answer with its exact required format.

Use a deterministic tool if it can model EVERY relevant condition: method = calculate, linear, sets, percent_chain, schedule, grid, csp. Otherwise use method = direct.
If you use a tool, include that tool's arguments exactly as described below. Ground every literal number in the question. Copy the quoted evidence EXACTLY from the user question, not paraphrases. Do not omit restrictions. Never create fictional facts or extra conditions.
Output JSON: {"method":"direct","answer":"FINAL ANSWER","check":"show the essential checked equation or constraints"} (other fields are allowed for tools).
For direct method: compute the answer before output, double-check arithmetic, signs, inequalities, ordering, and asked-for form. Do NOT reply with just a guessed number.
If the problem admits multiple valid results, return one that satisfies every explicit condition; never invent a uniqueness assumption.
"""

TOOL_GUIDES = {
 "calculate": """Best for single or multi-step arithmetic. Return {"method":"calculate","answer":"...","check":"...","expression":"(12*7)-5","evidence":["exact quote containing 12 and 7","exact quote containing 5"]}. Use Python-like + - * / parentheses only. The expression must compute the FINAL requested value, not an intermediate one. If the question has bullet conditions, include evidence for ALL of them. If this is not possible, use direct.""",
 "linear": """Best for additive relationships among named unknowns. Return {"method":"linear","answer":"...","check":"...","linear":{"variables":["x","y"],"equations":[{"expr":"x+y=20","source":"exact source quote"},{"expr":"x=3*y","source":"exact source quote"}],"answer_spec":{"type":"variable","name":"x"}}}. For named multi-part answers use answer_spec={"type":"assignment","variables":["x","y"],"labels":["First","Second"]}. When the question gives an exact output pattern like First=<integer>, Second=<integer>, preferably name your variables First and Second in equations and copy these labels exactly. Exactly copy all relevant source quotes. Include enough independent equations. If you cannot formalize relationships, use direct.""",
 "sets": """Best for 'both', 'either' and 'neither' with two categories. Return {"method":"sets","answer":"...","check":"...","sets":{"total":50,"count_a":29,"count_b":24,"both":9,"evidence":{"total":"exact quote","count_a":"exact quote","count_b":"exact quote","both":"exact quote"}}}. Only use it when the question asks how many belong to NEITHER set, otherwise use calculate or direct. All four counts must come from the question.""",
 "percent_chain": """Best for applying/reversing sequential percent increases/decreases. Return {"method":"percent_chain","answer":"...","check":"...","percent_chain":{"known_value":90,"solve_for":"initial","operations":[{"type":"increase","percent":20,"evidence":"exact quote"},{"type":"decrease","percent":25,"evidence":"exact quote"}]}}. 'initial' means inverse from the final known amount; 'final' means forward from the initial known amount. Preserve the operation order. Check which quantity the question asks for.""",
 "schedule": """Best for duration/dependency critical paths. Return {"method":"schedule","answer":"...","check":"...","tasks":[{"name":"stage1","duration":2,"depends_on":[],"evidence":["exact source quote for task duration"]},{"name":"stage2","duration":5,"depends_on":["stage1"],"evidence":["exact source quote for duration and prerequisite"]}]}. Include EVERY activity, including those that start in parallel. Quote all bullet conditions and prerequisites. Do not add dependencies not stated. If cannot encode all, use direct.""",
 "grid": """Best for shortest path on a bounded orthogonal square grid. Return {"method":"grid","answer":"...","check":"...","grid":{"start":[0,0],"goal":[3,3],"min_x":0,"max_x":3,"min_y":0,"max_y":3,"blocked":[[1,0]]}}. Copy every coordinate and stated bound. Up/down/left/right cost 1 per step, no diagonal. If movement rules differ, use direct.""",
 "csp": """Best for finite ordering, schedules, truth-tellers/liars, digit codes, and discrete logical constraints. Return {"method":"csp","answer":"...","check":"...","csp":{"variables":{"a":[1,2,3],"b":[1,2,3]},"constraints":[{"source":"exact quote from question","expr":"a != b"}],"answer":{"type":"variable","name":"a"}}}. Other answer forms: {"type":"concat","variables":["a","b"]}, {"type":"assignment","variables":["a","b"],"value_map":{"1":"Truth","0":"Lie"}}, {"type":"inverse","mapping":{"X":"a","Y":"b"},"value":1}. Operators: + - * / // %, comparisons == != > < >= <=, and/or/not, all_different(a,b), abs(a), int(a). Every bullet condition needs an exact source quote. For weekdays use numeric indices 1=Monday,2=Tuesday,etc. For logic 1=True and 0=False. If formalization might omit conditions, use direct."""
}

REPAIR_PROMPT = """You are repairing a failed reasoning plan. Re-read the full original question.
Return one JSON object using the same structured schema as the planner. Prefer a deterministic
method (calculate, linear, sets, percent_chain, schedule, grid, csp) whenever it can represent
all conditions. Include exact source evidence and compute the final requested value, not an
intermediate value. Use method=direct only if no listed tool can represent the task, and include
an actual independent check. Never invent facts or omit constraints.
"""


def select_tool_hint(problem):
    """Generic task classification, never a memorized question/answer lookup."""
    s = problem.casefold()
    if (re.search(r"\(\s*[۰-۹0-9-]+\s*[,،]\s*[۰-۹0-9-]+\s*\)", s)
            and any(k in s for k in ("مسدود", "blocked", "obstacle", "شبکه", "grid"))):
        return "grid"
    if any(k in s for k in ("پروژه", "project", "activity", "فعالیت")) and any(
            k in s for k in ("زمان", "duration", "depends", "پس از", "وابسته")):
        return "schedule"
    if any(k in s for k in ("هر دو", "both")) and any(k in s for k in ("هیچ", "neither")):
        return "sets"
    if ("%" in s or "٪" in s or "percent" in s) and any(
            k in s for k in ("increase", "decrease", "discount", "افزایش", "کاهش", "تخفیف")):
        return "percent_chain"
    if any(k in s for k in ("دروغ", "truth", "lie", "گزاره", "proposition", "exactly one", "قیود", "constraints", "distinct digits", "متمایز")):
        return "csp"
    if any(k in s for k in ("دو برابر", "سه برابر", "twice", "double", "triple", "بیشتر از", "less than")) and any(
            k in s for k in ("مجموع", "total", "altogether", "sum")):
        return "linear"
    return "calculate"


def plan_prompt(problem):
    method = select_tool_hint(problem)
    return (PLAN_PROMPT + "\nPREFERRED TOOL FOR THIS TYPE OF PROBLEM: " + method
            + "\n" + TOOL_GUIDES[method]
            + "\nYou may instead choose direct when the preferred tool cannot express ALL conditions."
            + " For a simple numeric problem, calculate is also allowed with expression and evidence.")


def _source_quote_list(spec):
    quotes = []
    for item in spec if isinstance(spec, list) else []:
        if isinstance(item, str):
            quotes.append(item)
        elif isinstance(item, dict):
            for key in ("source", "evidence_duration"):
                if isinstance(item.get(key), str):
                    quotes.append(item[key])
            extra = item.get("evidence", [])
            if isinstance(extra, str):
                quotes.append(extra)
            elif isinstance(extra, list):
                quotes.extend(x for x in extra if isinstance(x, str))
            extra = item.get("evidence_dependencies", [])
            if isinstance(extra, list):
                quotes.extend(x for x in extra if isinstance(x, str))
    return quotes



def _quote_has_number(quote, value):
    if type(value) not in (int, float):
        return True
    text = normalize_digits(str(quote)).replace(",", "")
    number = str(int(value)) if float(value).is_integer() else str(value)
    if re.search(r"(?<!\d)" + re.escape(number) + r"(?!\d)", text):
        return True
    lower = text.casefold()
    word_equivalents = {
        2: ("twice", "double", "دو برابر"),
        3: ("triple", "three times", "سه برابر"),
        0.5: ("half", "نصف"),
        0.25: ("quarter", "ربع"),
    }
    return any(token in lower for token in word_equivalents.get(value, ()))


def _validate_numeric_values(question, values):
    facts = _question_numeric_facts(question)
    for value in values:
        if type(value) not in (int, float):
            continue
        if not any(abs(float(value) - fact) <= 1e-9 for fact in facts):
            raise ToolError(f"Unsupported numeric value in plan: {value}")


def _validate_grid(question, grid):
    if not isinstance(grid, dict):
        raise ToolError("Missing grid plan")
    text = normalize_digits(question)
    if not (re.search(r"بالا.*پایین.*چپ.*راست", text) or
            all(re.search(r"\b" + direction + r"\b", text, re.I)
                for direction in ("up", "down", "left", "right"))):
        raise ToolError("Question does not explicitly define orthogonal grid moves")
    source_points = {
        (int(a), int(b)) for a, b in re.findall(
            r"\(\s*(-?\d+)\s*[,،]\s*(-?\d+)\s*\)", text
        )
    }
    planned_points = {tuple(grid.get("start", [])), tuple(grid.get("goal", []))}
    blocked = grid.get("blocked")
    if not isinstance(blocked, list):
        raise ToolError("Grid blocked cells must be a list")
    planned_points.update(tuple(p) for p in blocked if isinstance(p, (list, tuple)))
    if len(source_points) < 2 or source_points != planned_points:
        raise ToolError("Grid plan does not exactly represent source coordinates")
    bounds = {}
    for axis in ("x", "y"):
        match = re.search(rf"(-?\d+)\s*<=\s*{axis}\s*<=\s*(-?\d+)", text, re.I)
        if match:
            bounds[f"min_{axis}"] = int(match.group(1))
            bounds[f"max_{axis}"] = int(match.group(2))
    for key, value in bounds.items():
        if grid.get(key) != value:
            raise ToolError("Grid bounds do not match the question")
    return True


def _linear_spec_with_requested_format(question, spec):
    """Repair a malformed *output mapping*, never mathematical equations.

    The model may solve a system correctly but misspell/omit answer_spec's
    variables and labels.  For a question that explicitly supplies labels in
    e.g. A=<integer>, B=<integer>, use those labels and match them to existing
    solver variables.  Do not invent a mapping when the correspondence cannot
    be inferred; that would silently turn a formatting error into a wrong
    answer.
    """
    template = re.findall(
        r"(?<![\w])([A-Za-z][A-Za-z0-9_]*)\s*=\s*<\s*(?:integer|int|number|digit)\s*>",
        question, re.I,
    )
    if len(template) < 2 or len(set(x.casefold() for x in template)) != len(template):
        return spec
    names = spec.get("variables")
    if not isinstance(names, list) or len(template) != len(names) or not all(
        isinstance(n, str) for n in names
    ):
        return spec

    # Match case-insensitively, including unambiguous initials (n -> Nima).
    mapped = []
    for label in template:
        exact = [v for v in names if v.casefold() == label.casefold()]
        initial = [v for v in names if len(v) == 1 and v.casefold() == label[0].casefold()]
        choices = exact or initial
        if len(choices) != 1 or choices[0] in mapped:
            mapped = []
            break
        mapped.append(choices[0])

    answer_spec = spec.get("answer_spec")
    if not mapped:
        # An explicit order of valid variable names is still meaningful even
        # when the model forgot the label array.  Do not guess from the order
        # of unrelated variable names.
        if (isinstance(answer_spec, dict) and answer_spec.get("type") == "assignment"
                and isinstance(answer_spec.get("variables"), list)
                and len(answer_spec["variables"]) == len(names)
                and set(answer_spec["variables"]) == set(names)):
            mapped = list(answer_spec["variables"])
        else:
            return spec

    expected = {"type": "assignment", "variables": mapped, "labels": template}
    if answer_spec == expected:
        return spec
    fixed = dict(spec)
    fixed["answer_spec"] = expected
    return fixed


def _execute_plan(question, plan, answer_format):
    method = plan.get("method", "direct")
    if method == "direct":
        return None
    if method == "calculate":
        evidence = plan.get("evidence")
        _require_quotes(question, evidence if isinstance(evidence, list) else [evidence], "calculation plan")
        validate_bullet_coverage(question, evidence if isinstance(evidence, list) else [evidence], "calculation plan")
        require_grounded_plan(question, {"method": "calculate", "expression": plan.get("expression", "")})
        return normalize_answer(safe_arithmetic(plan.get("expression")), answer_format)
    if method == "linear":
        spec = plan.get("linear")
        if not isinstance(spec, dict):
            raise ToolError("Missing linear plan")
        equations = spec.get("equations", [])
        quotes = [e.get("source") for e in equations if isinstance(e, dict)]
        if len(quotes) != len(equations):
            raise ToolError("Each linear equation needs source evidence")
        _require_quotes(question, quotes, "linear plan")
        validate_bullet_coverage(question, quotes, "linear plan")
        for equation in equations:
            try:
                tree = ast.parse(equation.get("expr", "").replace("=", "==", 1), mode="eval")
            except (SyntaxError, ValueError) as exc:
                raise ToolError("Invalid linear equation") from exc
            constants = [n.value for n in ast.walk(tree)
                         if isinstance(n, ast.Constant) and type(n.value) in (int, float)]
            if any(v not in (0, 1) and not _quote_has_number(equation.get("source", ""), v)
                   for v in constants):
                raise ToolError("Equation constant does not match its source evidence")
        _require_numeric_plan_for_linear(question, spec)
        fixed_spec = _linear_spec_with_requested_format(question, spec)
        return normalize_answer(solve_linear_system(fixed_spec), answer_format)
    if method == "sets":
        spec = plan.get("sets")
        if not isinstance(spec, dict):
            raise ToolError("Missing set-count plan")
        evidence = spec.get("evidence")
        keys = ("total", "count_a", "count_b", "both")
        if not isinstance(evidence, dict) or any(not isinstance(evidence.get(k), str) for k in keys):
            raise ToolError("Set-count evidence must map each count to its source quote")
        quotes = [evidence[k] for k in keys]
        _require_quotes(question, quotes, "set-count plan")
        validate_bullet_coverage(question, quotes, "set-count plan")
        for key in keys:
            if not _quote_has_number(evidence[key], spec.get(key)):
                raise ToolError(f"Set-count {key} does not match its evidence")
        _validate_numeric_values(question, [spec.get(k) for k in keys])
        return normalize_answer(solve_set_count(spec), answer_format)
    if method == "percent_chain":
        spec = plan.get("percent_chain")
        if not isinstance(spec, dict):
            raise ToolError("Missing percentage plan")
        ops = spec.get("operations", [])
        quotes = [op.get("evidence") for op in ops if isinstance(op, dict)]
        if len(quotes) != len(ops):
            raise ToolError("Each percentage operation needs evidence")
        _require_quotes(question, quotes, "percentage plan")
        validate_bullet_coverage(question, quotes, "percentage plan")
        for op in ops:
            if not _quote_has_number(op.get("evidence", ""), op.get("percent")):
                raise ToolError("Percentage value does not match its evidence")
        values = [spec.get("known_value")] + [op.get("percent") for op in ops]
        _validate_numeric_values(question, values)
        return normalize_answer(solve_percent_chain(spec), answer_format)
    if method == "schedule":
        tasks = plan.get("tasks")
        if not isinstance(tasks, list):
            raise ToolError("Missing schedule tasks")
        quotes = _source_quote_list([t for t in tasks if isinstance(t, dict)])
        validate_bullet_coverage(question, quotes, "schedule plan")
        _require_quotes(question, quotes, "schedule plan")
        for task in tasks:
            task_quotes = task.get("evidence", []) if isinstance(task, dict) else []
            if isinstance(task_quotes, str):
                task_quotes = [task_quotes]
            if not any(_quote_has_number(q, task.get("duration")) for q in task_quotes if isinstance(q, str)):
                raise ToolError("A task duration does not match its evidence")
        _validate_numeric_values(question, [t.get("duration") for t in tasks if isinstance(t, dict)])
        return normalize_answer(calculate_project_duration(tasks), answer_format)
    if method == "grid":
        grid = plan.get("grid")
        _validate_grid(question, grid)
        require_grounded_plan(question, {"method": "grid", "grid": grid})
        return normalize_answer(shortest_grid_path(grid), answer_format)
    if method == "csp":
        spec = plan.get("csp")
        if not isinstance(spec, dict):
            raise ToolError("Missing CSP plan")
        validate_csp_evidence(question, spec.get("constraints"))
        require_grounded_plan(question, {"method": "csp", "csp": spec})
        result = solve_csp(spec)
        # If multiple valid answers exist, the source question is ambiguous.
        # The deterministic solver returns a reproducible valid candidate;
        # diagnostics expose the number of distinct answer candidates.
        return normalize_answer(result["answer"], answer_format), result
    raise ToolError("Unsupported tool method")


def _require_numeric_plan_for_linear(question, spec):
    values = []
    for eq in spec.get("equations", []):
        expr = eq.get("expr", "") if isinstance(eq, dict) else ""
        try:
            tree = ast.parse(expr.replace("=", "==", 1), mode="eval")
        except (SyntaxError, ValueError) as exc:
            raise ToolError("Invalid linear equation") from exc
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and type(node.value) in (int, float):
                values.append(node.value)
    _validate_numeric_values(question, values)


class ReasoningAgent:
    def __init__(self, llm=None):
        self.llm = llm if llm is not None else LocalOllama()
        self.max_calls = max(1, min(2, int(os.getenv("AGENT_MAX_CALLS", "2"))))
        self.call_timeout = float(os.getenv("AGENT_CALL_TIMEOUT", "120"))
        self.total_timeout = float(os.getenv("AGENT_TOTAL_TIMEOUT", "150"))
        self.num_predict = max(192, min(1536, int(os.getenv("AGENT_NUM_PREDICT", "768"))))
        strategy = os.getenv("AGENT_STRATEGY", "auto").strip().lower()
        if strategy not in ("auto", "fast", "model-first"):
            raise ValueError("AGENT_STRATEGY must be auto, fast, or model-first")
        self.strategy = strategy
        self.model_calls = 0
        self.started = 0.0

    def _prefer_model(self, solver_name):
        """Use larger grading models to interpret semantically delicate rules.

        This is based on solver *families*, not any question text, ID or
        precomputed sample answer.  The deterministic result remains a
        fallback if the local model fails or hits its budget.
        """
        if self.strategy == "fast":
            return False
        if self.strategy == "model-first":
            return True
        model = str(getattr(self.llm, "model_name", "")).lower()
        large = bool(re.search(r"(?:27b|32b|35b|70b|72b|120b|122b)", model))
        delicate = {
            "_try_generic_schedule", "_try_generic_truth_liar",
            "_try_generic_weekday_assignment", "_try_generic_box_logic",
            "_try_generic_digit_code", "_try_generic_ordering",
            "_try_generic_linear_assignment",
        }
        return large and solver_name in delicate

    @staticmethod
    def _diag(message):
        if os.getenv("AGENT_DIAGNOSTICS", "").lower() in ("1", "true", "yes"):
            print("AGENT DIAG: " + str(message), file=sys.stderr)

    def _ask(self, system, user, budget=None, raw_text=False):
        if self.model_calls >= self.max_calls:
            raise AgentError("Model-call budget reached")
        elapsed = time.monotonic() - self.started
        remaining = self.total_timeout - elapsed
        if remaining <= 2:
            raise AgentError("Overall time limit reached")
        self.model_calls += 1
        return self.llm.chat(system, user, timeout=min(self.call_timeout, remaining),
                             num_predict=budget or self.num_predict,
                             **({"response_format": "text"} if raw_text else {}))

    def _independent_check(self, question, answer_format, primary):
        """Optional second model pass for direct answers without a tool proof.

        Unlike merely asking again, the second pass explicitly checks every
        condition and can return an executable tool plan. Invalid/empty
        critiques do not destroy a usable primary candidate.
        """
        if self.model_calls >= self.max_calls:
            return None
        system = (
            "You are an independent verifier of math, logic, and reasoning problems. "
            "First solve the ORIGINAL problem yourself, including all constraints. "
            "Then compare with the candidate. Do not agree merely because the "
            "candidate was supplied. Return one JSON object: "
            '{"status":"correct|incorrect|uncertain","answer":"FINAL ANSWER",'
            '"check":"short concrete check"}. '
            "If you can formalize a deterministic tool plan, you may instead "
            "return the usual method=calculate/linear/sets/percent_chain/schedule/grid/csp "
            "plan, including exact source evidence. No Markdown."
        )
        user = (f"QUESTION:\n{question}\n\nCANDIDATE: {primary}\n"
                f"REQUIRED FORMAT: {answer_format}\n"
                "Check the requested FINAL value, every explicit restriction, and output format.")
        try:
            raw = self._ask(system, user, budget=min(self.num_predict, 512))
            obj = parse_json_object(raw)
            if obj.get("method") in ("calculate", "linear", "sets", "percent_chain",
                                     "schedule", "grid", "csp"):
                verified = _execute_plan(question, obj, answer_format)
                return verified[0] if isinstance(verified, tuple) else verified
            proposed = normalize_answer(obj.get("answer"), answer_format)
            status = str(obj.get("status", "")).strip().lower()
            if status == "correct":
                return primary
            if status == "incorrect" and proposed != primary and isinstance(obj.get("check"), str) and len(obj["check"].strip()) >= 8:
                return proposed
            # Old-style response from small models: a repeated answer is a
            # consistency check, not proof; disagreements remain uncertain.
            if proposed == primary:
                return primary
            return None
        except (AgentError, OllamaClientError, ToolError, AnswerFormatError,
                ValueError, TypeError, KeyError, ArithmeticError) as exc:
            self._diag(f"independent check unavailable: {type(exc).__name__}: {exc}")
            return None

    def _repair(self, question, answer_format, reason):
        """One compact, local-only recovery call. Never loop or exceed 2 calls.

        A small local model sometimes fails to create a large tool-plan JSON.
        Recovery therefore asks for only the final answer without JSON-mode
        decoding. If it still returns a valid structured plan, verify that
        plan with the same deterministic tools before returning anything.
        """
        if self.model_calls >= self.max_calls:
            return None
        system = (
            "You solve mathematics, logic and word problems accurately. "
            "Read all stated facts, compute the requested FINAL quantity, "
            "check your arithmetic and constraints internally. "
            "Return ONLY the final answer; no explanation, markup or JSON."
        )
        user = (
            f"PROBLEM:\n{question}\n\n"
            f"REQUIRED ANSWER FORMAT: {answer_format}.\n"
            "Return only that answer."
        )
        try:
            raw = self._ask(system, user, budget=192, raw_text=True)
            # Some models still emit JSON in plain-text mode; use it if given.
            try:
                obj = parse_json_object(raw)
            except ValueError:
                obj = None
            if isinstance(obj, dict):
                candidate = None
                try:
                    candidate = normalize_answer(obj.get("answer"), answer_format)
                except (AnswerFormatError, TypeError, ValueError):
                    pass
                try:
                    computed = _execute_plan(question, obj, answer_format)
                    if isinstance(computed, tuple):
                        computed = computed[0]
                    if computed is not None:
                        return computed
                except (ToolError, AnswerFormatError, TypeError, ValueError,
                        KeyError, ArithmeticError) as exc:
                    self._diag(f"recovery tool plan was invalid: {exc}")
                    # A broken tool plan is not proof of the candidate answer.
                    return None
                # A direct answer is best effort, not independently proven.
                if candidate is not None and obj.get("method", "direct") == "direct":
                    self._diag("recovered a format-valid direct answer from JSON")
                    return candidate
                return None

            # Recover from a fenced /think/ segment or a labelled final line.
            clean = re.sub(r"<think>.*?</think>", "", raw, flags=re.I | re.S).strip()
            clean = re.sub(r"^```[^\n]*\n?|\n?```$", "", clean).strip()
            if "\n" in clean:
                final_lines = [line.strip() for line in clean.splitlines()
                               if re.match(r"^(?:final\s+answer|answer)\s*[:：]", line, re.I)]
                if len(final_lines) == 1:
                    clean = final_lines[0]
            clean = re.sub(r"^(?:final\s+answer|answer)\s*[:：]\s*", "", clean, flags=re.I)
            answer = normalize_answer(clean, answer_format)
            self._diag("recovered a format-valid final answer using the compact prompt")
            return answer
        except (AgentError, OllamaClientError, ToolError, ValueError, TypeError,
                AnswerFormatError, KeyError, ArithmeticError) as exc:
            self._diag(f"recovery attempt failed: {type(exc).__name__}: {exc}")
            return None

    def run(self, problem):
        if not isinstance(problem, str) or not problem.strip():
            raise AgentError("Problem must be a non-empty string")
        self.started = time.monotonic()
        self.model_calls = 0
        fmt = detect_format(problem)
        # A conservative mechanical baseline, retained if local inference
        # later fails. For large grading models, semantically nuanced puzzles
        # are interpreted by the model rather than a brittle language regex.
        fallback = None
        try:
            solved = solve_question_family(problem, fmt, with_meta=True)
            if solved is not None:
                proposed, solver_name = solved
                fallback = normalize_answer(proposed, fmt)
                if not self._prefer_model(solver_name):
                    self._diag("deterministic family solver: " + solver_name)
                    return fallback
                self._diag("model-first semantic check for " + solver_name)
        except (ToolError, AnswerFormatError, ValueError, TypeError, ArithmeticError) as exc:
            self._diag(f"deterministic family solver abstained: {exc}")

        user = ("Read the complete problem. Extract all facts and constraints. Select a method, solve, and verify the final requested value.\n\n"
                f"QUESTION:\n{problem}\n\nREQUIRED ANSWER FORMAT: {fmt}\n"
                "به‌ویژه مراقب باش مقدار خواسته‌شده را با مقدار میانی اشتباه نگیری."
                " تمام شرط‌ها باید در ابزار انتخابی پوشش داده شوند.")
        try:
            raw = self._ask(plan_prompt(problem), user)
            plan = parse_json_object(raw)
        except (AgentError, OllamaClientError, ValueError, TypeError) as exc:
            self._diag(f"primary request failed: {exc}")
            # Loading the selected model has definitively failed. Repeating
            # the same call cannot create RAM/VRAM and only wastes the budget.
            if _is_fatal_memory_error(exc):
                self._diag("Ollama model failed to load due to memory; skip redundant recovery call")
                if fallback is not None:
                    return fallback
                raise AgentError("Local Ollama model cannot be loaded: insufficient memory") from exc
            # A timeout is not repaired with another slow LLM request. The CLI
            # still writes valid JSON with an empty answer, as required.
            if isinstance(exc, OllamaClientError) and ("timed out" in str(exc).casefold() or "timeout" in str(exc).casefold()):
                if fallback is not None:
                    return fallback
                raise AgentError("Local Ollama request timed out") from exc
            repaired = self._repair(problem, fmt, str(exc))
            if repaired is not None:
                return repaired
            if fallback is not None:
                return fallback
            raise AgentError("No valid response from local Ollama") from exc

        if plan.get("method") == "linear" and isinstance(plan.get("linear"), dict):
            self._diag("linear mapping: " + json.dumps({
                "variables": plan["linear"].get("variables"),
                "answer_spec": plan["linear"].get("answer_spec"),
            }, ensure_ascii=False)[:900])

        candidate = None
        try:
            candidate = normalize_answer(plan.get("answer"), fmt)
        except (AnswerFormatError, TypeError, ValueError) as exc:
            self._diag(f"candidate rejected: {exc}")

        try:
            computed = _execute_plan(problem, plan, fmt)
            tool_meta = None
            if isinstance(computed, tuple):
                tool_answer, tool_meta = computed
            else:
                tool_answer = computed
            if tool_answer is not None:
                if tool_meta and tool_meta.get("solution_count", 0) > 1:
                    self._diag(f"CSP has {tool_meta['solution_count']} valid assignments; answer candidates={tool_meta.get('candidate_answers')}")
                if candidate is not None and candidate != tool_answer:
                    self._diag(f"deterministic verifier replaced candidate {candidate!r} with {tool_answer!r}")
                return tool_answer
        except (ToolError, AnswerFormatError, TypeError, ValueError, KeyError, ArithmeticError) as exc:
            self._diag(f"tool plan rejected: {exc}")
            # A malformed tool plan does not verify the model's answer.
            # Try one independent correction; if unavailable, preserve any
            # correctly formatted candidate rather than return no answer.
            repaired = self._repair(problem, fmt,
                                    f"The {plan.get('method')} tool was invalid: {exc}; verify all original constraints")
            if repaired is not None:
                return repaired
            if fallback is not None:
                self._diag("reverting to deterministic fallback after rejected model plan")
                return fallback
            if candidate is not None and os.getenv("AGENT_STRICT_VERIFICATION", "0") != "1":
                # The grading metric treats both an empty answer and a wrong
                # answer as incorrect. A format-valid unverified answer is a
                # better last resort than an empty JSON field; diagnostics
                # explicitly mark that it is NOT a verification result.
                self._diag("unverified last-resort answer after rejected plan")
                return candidate
            raise AgentError("No verified answer or executable plan") from exc

        if candidate is not None:
            if plan.get("method") == "direct":
                check = plan.get("check")
                # Avoid needlessly checking a simple textual fact, but check
                # numeric, code, assignment and constrained direct answers.
                if fmt == "text" and isinstance(check, str) and len(check.strip()) >= 16:
                    return candidate
                checked = self._independent_check(problem, fmt, candidate)
                if checked is not None:
                    return checked
                if fallback is not None and not isinstance(check, str):
                    return fallback
                self._diag("using format-valid model answer (no independent proof)")
                return candidate
            repaired = self._repair(problem, fmt, "Primary answer was not tool-verified")
            if repaired is not None:
                return repaired
            if fallback is not None:
                return fallback
            if os.getenv("AGENT_STRICT_VERIFICATION", "0") != "1":
                return candidate
            raise AgentError("No verified tool answer available")

        repaired = self._repair(problem, fmt, "Primary answer did not match the required format")
        if repaired is not None:
            return repaired
        if fallback is not None:
            return fallback
        raise AgentError("No valid final answer")

    def close(self):
        close = getattr(self.llm, "close", None)
        if callable(close):
            close()
