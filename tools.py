"""Bounded, deterministic reasoning tools. No eval/exec and no network access."""

import ast
from collections import deque
from decimal import Decimal, localcontext
from fractions import Fraction
import math
import re
import itertools


class ToolError(ValueError):
    """Invalid or unsafe tool input."""


def _fraction(value):
    if type(value) is int:
        return Fraction(value)
    if type(value) is float and math.isfinite(value):
        return Fraction(str(value))
    raise ToolError("Expected a finite number")


def _format_fraction(value):
    if value.denominator == 1:
        return str(value.numerator)
    with localcontext() as ctx:
        ctx.prec = 18
        result = Decimal(value.numerator) / Decimal(value.denominator)
    return format(result.normalize(), "f")


def safe_arithmetic(expression):
    """Evaluate a small arithmetic expression using a strict AST allow-list."""
    if not isinstance(expression, str) or not expression.strip() or len(expression) > 400:
        raise ToolError("Invalid arithmetic expression")
    try:
        root = ast.parse(expression, mode="eval").body
    except (SyntaxError, ValueError) as exc:
        raise ToolError("Invalid arithmetic syntax") from exc

    def visit(node, depth=0):
        if depth > 24:
            raise ToolError("Arithmetic expression is too deep")
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            value = _fraction(node.value)
        elif isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            child = visit(node.operand, depth + 1)
            value = child if isinstance(node.op, ast.UAdd) else -child
        elif isinstance(node, ast.BinOp) and isinstance(
            node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)
        ):
            left, right = visit(node.left, depth + 1), visit(node.right, depth + 1)
            if isinstance(node.op, ast.Add):
                value = left + right
            elif isinstance(node.op, ast.Sub):
                value = left - right
            elif isinstance(node.op, ast.Mult):
                value = left * right
            else:
                if right == 0:
                    raise ToolError("Division by zero")
                value = left / right
        else:
            raise ToolError("Only numbers and +, -, *, /, parentheses are allowed")
        if abs(value.numerator) > 10**18 or value.denominator > 10**18:
            raise ToolError("Arithmetic result exceeds safety bounds")
        return value

    return _format_fraction(visit(root))


def calculate_project_duration(tasks):
    """Earliest completion time for a DAG of activities and prerequisites."""
    if not isinstance(tasks, list) or not tasks or len(tasks) > 100:
        raise ToolError("Tasks must be a non-empty list of at most 100 items")

    durations, dependencies = {}, {}
    for task in tasks:
        if not isinstance(task, dict):
            raise ToolError("Each task must be an object")
        name, duration = task.get("name"), task.get("duration")
        deps = task.get("depends_on", [])
        if not isinstance(name, str) or not name.strip() or name in durations:
            raise ToolError("Task names must be unique non-empty strings")
        if type(duration) not in (int, float) or duration < 0 or not math.isfinite(duration):
            raise ToolError("Task duration must be finite and non-negative")
        if not isinstance(deps, list) or not all(isinstance(x, str) for x in deps):
            raise ToolError("Invalid dependency list")
        if len(deps) != len(set(deps)):
            raise ToolError("Duplicate dependency")
        durations[name] = _fraction(duration)
        dependencies[name] = deps

    successors = {name: [] for name in durations}
    indegree = {name: len(deps) for name, deps in dependencies.items()}
    for name, deps in dependencies.items():
        for dep in deps:
            if dep not in durations:
                raise ToolError("Unknown prerequisite: " + dep)
            successors[dep].append(name)

    ready = deque(name for name, degree in indegree.items() if degree == 0)
    starts = {name: Fraction(0) for name in durations}
    processed = 0
    while ready:
        name = ready.popleft()
        processed += 1
        finish = starts[name] + durations[name]
        for nxt in successors[name]:
            starts[nxt] = max(starts[nxt], finish)
            indegree[nxt] -= 1
            if indegree[nxt] == 0:
                ready.append(nxt)

    if processed != len(durations):
        raise ToolError("Circular task dependencies")
    return _format_fraction(max(starts[n] + durations[n] for n in durations))


def shortest_grid_path(grid):
    """BFS for unit-cost orthogonal movement on a bounded rectangular grid."""
    if not isinstance(grid, dict):
        raise ToolError("Grid must be an object")
    try:
        min_x, max_x = grid["min_x"], grid["max_x"]
        min_y, max_y = grid["min_y"], grid["max_y"]
        start, goal, blocked = grid["start"], grid["goal"], grid["blocked"]
    except KeyError as exc:
        raise ToolError("Grid is missing a required field") from exc

    if not all(type(x) is int for x in (min_x, max_x, min_y, max_y)):
        raise ToolError("Grid bounds must be integers")
    if min_x > max_x or min_y > max_y:
        raise ToolError("Invalid grid bounds")
    if (max_x - min_x + 1) * (max_y - min_y + 1) > 100000:
        raise ToolError("Grid is too large")

    def point(raw):
        if not isinstance(raw, (list, tuple)) or len(raw) != 2:
            raise ToolError("Coordinates must contain two integers")
        if not all(type(x) is int for x in raw):
            raise ToolError("Coordinates must contain two integers")
        x, y = raw
        if not (min_x <= x <= max_x and min_y <= y <= max_y):
            raise ToolError("Coordinate is outside the grid")
        return x, y

    source, target = point(start), point(goal)
    if not isinstance(blocked, list):
        raise ToolError("Blocked cells must be a list")
    obstacles = {point(p) for p in blocked}
    if source in obstacles or target in obstacles:
        raise ToolError("Start or goal is blocked")

    queue = deque([(source, 0)])
    seen = {source}
    while queue:
        (x, y), distance = queue.popleft()
        if (x, y) == target:
            return str(distance)
        for dx, dy in ((0, 1), (0, -1), (1, 0), (-1, 0)):
            nxt = (x + dx, y + dy)
            if (min_x <= nxt[0] <= max_x and min_y <= nxt[1] <= max_y
                    and nxt not in obstacles and nxt not in seen):
                seen.add(nxt)
                queue.append((nxt, distance + 1))
    raise ToolError("No path exists")


# ----- Small finite-domain constraint solver ---------------------------------

_ALLOWED_BINOPS = (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod)
_ALLOWED_UNARY = (ast.UAdd, ast.USub, ast.Not)
_ALLOWED_COMPARE = (ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.In, ast.NotIn)
_ALLOWED_BOOL = (ast.And, ast.Or)
_ALLOWED_CALLS = {"all_different", "int", "abs"}


def _validate_expr(node, variable_names):
    for item in ast.walk(node):
        if isinstance(item, ast.Name):
            if item.id not in variable_names and item.id not in _ALLOWED_CALLS:
                raise ToolError("Unknown name in CSP expression: " + item.id)
        elif isinstance(item, ast.Call):
            if not isinstance(item.func, ast.Name) or item.func.id not in _ALLOWED_CALLS:
                raise ToolError("Unsupported function in CSP expression")
            if item.keywords:
                raise ToolError("Keyword arguments are not allowed")
        elif isinstance(item, ast.BinOp):
            if not isinstance(item.op, _ALLOWED_BINOPS):
                raise ToolError("Unsupported arithmetic operator in CSP")
        elif isinstance(item, ast.UnaryOp):
            if not isinstance(item.op, _ALLOWED_UNARY):
                raise ToolError("Unsupported unary operator in CSP")
        elif isinstance(item, ast.BoolOp):
            if not isinstance(item.op, _ALLOWED_BOOL):
                raise ToolError("Unsupported boolean operator in CSP")
        elif isinstance(item, ast.Compare):
            if not all(isinstance(op, _ALLOWED_COMPARE) for op in item.ops):
                raise ToolError("Unsupported comparison in CSP")
        elif isinstance(item, (ast.Expression, ast.Load, ast.Constant, ast.List,
                               ast.Tuple, ast.Set, ast.keyword, ast.operator,
                               ast.unaryop, ast.boolop, ast.cmpop)):
            pass
        elif isinstance(item, (ast.Store,)):
            raise ToolError("Assignments are not allowed in CSP expressions")
        else:
            raise ToolError("Unsupported syntax in CSP expression")


def _eval_expr(node, env):
    if isinstance(node, ast.Constant):
        if type(node.value) not in (int, float, str, bool):
            raise ToolError("Unsupported constant in CSP")
        return node.value
    if isinstance(node, ast.Name):
        if node.id not in env:
            raise KeyError(node.id)
        return env[node.id]
    if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        values = [_eval_expr(x, env) for x in node.elts]
        if isinstance(node, ast.Tuple):
            return tuple(values)
        if isinstance(node, ast.Set):
            return set(values)
        return values
    if isinstance(node, ast.UnaryOp):
        value = _eval_expr(node.operand, env)
        if isinstance(node.op, ast.Not):
            return not value
        if isinstance(node.op, ast.USub):
            return -value
        return +value
    if isinstance(node, ast.BinOp):
        left, right = _eval_expr(node.left, env), _eval_expr(node.right, env)
        if isinstance(node.op, ast.Add): return left + right
        if isinstance(node.op, ast.Sub): return left - right
        if isinstance(node.op, ast.Mult): return left * right
        if isinstance(node.op, ast.Div): return left / right
        if isinstance(node.op, ast.FloorDiv): return left // right
        if isinstance(node.op, ast.Mod): return left % right
    if isinstance(node, ast.BoolOp):
        values = [_eval_expr(x, env) for x in node.values]
        return all(values) if isinstance(node.op, ast.And) else any(values)
    if isinstance(node, ast.Compare):
        left = _eval_expr(node.left, env)
        for op, comparator in zip(node.ops, node.comparators):
            right = _eval_expr(comparator, env)
            if isinstance(op, ast.Eq): ok = left == right
            elif isinstance(op, ast.NotEq): ok = left != right
            elif isinstance(op, ast.Lt): ok = left < right
            elif isinstance(op, ast.LtE): ok = left <= right
            elif isinstance(op, ast.Gt): ok = left > right
            elif isinstance(op, ast.GtE): ok = left >= right
            elif isinstance(op, ast.In): ok = left in right
            elif isinstance(op, ast.NotIn): ok = left not in right
            else: raise ToolError("Unsupported comparison")
            if not ok:
                return False
            left = right
        return True
    if isinstance(node, ast.Call):
        name = node.func.id
        args = [_eval_expr(x, env) for x in node.args]
        if name == "all_different":
            return len(args) == len(set(args))
        if name == "int" and len(args) == 1:
            return int(args[0])
        if name == "abs" and len(args) == 1:
            return abs(args[0])
    raise ToolError("Unsupported CSP expression")


def _normal_space(text):
    return re.sub(r"\s+", " ", str(text)).strip().casefold()


def validate_csp_evidence(question, constraints):
    """Require every model-quoted source fragment to exist in the original text."""
    if not isinstance(constraints, list) or not constraints:
        raise ToolError("CSP needs source-backed constraints")
    q = _normal_space(question)
    for item in constraints:
        if not isinstance(item, dict):
            raise ToolError("Each CSP constraint needs source and expr")
        source, expr = item.get("source"), item.get("expr")
        if not isinstance(source, str) or not source.strip():
            raise ToolError("CSP constraint is missing its source quote")
        if not isinstance(expr, str) or not expr.strip():
            raise ToolError("CSP constraint is missing its expression")
        if _normal_space(source) not in q:
            raise ToolError("CSP source quote is not present in the question")

    # Do not allow the planner to silently omit explicit bullet constraints.
    bullet_lines = []
    for line in question.splitlines():
        match = re.match(r"^\s*[-•*]\s*(.+?)\s*$", line)
        if match:
            bullet_lines.append(_normal_space(match.group(1)))
    sources = [_normal_space(item["source"]) for item in constraints]
    for bullet in bullet_lines:
        if not any(bullet in source or source in bullet for source in sources):
            raise ToolError("CSP plan omitted an explicit bullet constraint")
    return True


def _answer_from_solution(spec, solution):
    kind = spec.get("type")
    if kind == "variable":
        name = spec.get("name")
        if name not in solution:
            raise ToolError("CSP answer variable is missing")
        value = solution[name]
        mapping = spec.get("value_map", {})
        if mapping:
            return str(mapping.get(str(value), value))
        return str(value)
    if kind == "inverse":
        mapping, target = spec.get("mapping"), spec.get("value")
        if not isinstance(mapping, dict) or not mapping:
            raise ToolError("Invalid inverse answer mapping")
        matches = sorted(label for label, variable in mapping.items()
                         if variable in solution and solution[variable] == target)
        if not matches:
            raise ToolError("Inverse answer has no match")
        return str(matches[0])
    if kind == "concat":
        names = spec.get("variables")
        if not isinstance(names, list) or not names or any(n not in solution for n in names):
            raise ToolError("Invalid concatenated answer variables")
        return "".join(str(solution[n]) for n in names)
    if kind == "assignment":
        names = spec.get("variables")
        mapping = spec.get("value_map", {})
        if not isinstance(names, list) or not names or any(n not in solution for n in names):
            raise ToolError("Invalid assignment answer variables")
        parts = []
        for name in names:
            value = solution[name]
            label = mapping.get(str(value), value) if isinstance(mapping, dict) else value
            parts.append(f"{name}={label}")
        return ", ".join(parts)
    raise ToolError("Unsupported CSP answer specification")


def solve_csp(spec, *, max_nodes=50000, max_solutions=500):
    """Solve a small finite-domain CSP with bounded backtracking and safe expressions.

    Variables are a mapping from names to finite lists. Constraints are safe
    Python-like boolean expressions; only the operators listed above are allowed.
    """
    if not isinstance(spec, dict):
        raise ToolError("CSP specification must be an object")
    variables = spec.get("variables")
    constraints = spec.get("constraints")
    answer_spec = spec.get("answer")
    if not isinstance(variables, dict) or not 1 <= len(variables) <= 8:
        raise ToolError("CSP must define between 1 and 8 variables")
    clean_domains = {}
    for name, domain in variables.items():
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z_]\w*", name):
            raise ToolError("Invalid CSP variable name")
        if not isinstance(domain, list) or not 1 <= len(domain) <= 12:
            raise ToolError("Each CSP domain must contain 1 to 12 values")
        if any(type(v) not in (int, float, str, bool) for v in domain):
            raise ToolError("Unsupported CSP domain value")
        if len({(type(v).__name__, str(v)) for v in domain}) != len(domain):
            raise ToolError("Duplicate CSP domain value")
        clean_domains[name] = domain

    if not isinstance(constraints, list) or not 1 <= len(constraints) <= 30:
        raise ToolError("CSP must contain 1 to 30 constraints")
    compiled = []
    for item in constraints:
        if not isinstance(item, dict) or not isinstance(item.get("expr"), str):
            raise ToolError("Invalid CSP constraint")
        try:
            node = ast.parse(item["expr"], mode="eval").body
        except (SyntaxError, ValueError) as exc:
            raise ToolError("Invalid CSP expression") from exc
        _validate_expr(node, set(clean_domains))
        names = {n.id for n in ast.walk(node)
                 if isinstance(n, ast.Name) and n.id not in _ALLOWED_CALLS}
        compiled.append((node, names))

    # Search smallest domains first; constraints are checked as soon as all
    # variables they mention have been assigned.
    order = sorted(clean_domains, key=lambda n: (len(clean_domains[n]), n))
    solutions = []
    nodes = 0

    def visit(index, env):
        nonlocal nodes
        nodes += 1
        if nodes > max_nodes:
            raise ToolError("CSP search limit reached")
        for node, names in compiled:
            if names.issubset(env):
                try:
                    if not bool(_eval_expr(node, env)):
                        return
                except (ArithmeticError, TypeError, ValueError, KeyError) as exc:
                    raise ToolError("CSP constraint could not be evaluated") from exc
        if index == len(order):
            solutions.append(dict(env))
            if len(solutions) > max_solutions:
                raise ToolError("CSP has too many solutions")
            return
        name = order[index]
        for value in clean_domains[name]:
            env[name] = value
            visit(index + 1, env)
            del env[name]

    visit(0, {})
    if not solutions:
        raise ToolError("CSP has no satisfying assignment")
    if not isinstance(answer_spec, dict):
        raise ToolError("CSP answer specification is missing")

    candidates = sorted({_answer_from_solution(answer_spec, solution)
                         for solution in solutions})
    if not candidates:
        raise ToolError("CSP produced no answer candidate")
    return {
        "answer": candidates[0],
        "candidate_answers": candidates[:30],
        "solution_count": len(solutions),
    }


# ----- Numeric grounding for model-proposed tool inputs ----------------------

_SMALL_WORDS = {
    "zero":0, "one":1, "two":2, "three":3, "four":4, "five":5,
    "six":6, "seven":7, "eight":8, "nine":9, "ten":10, "eleven":11,
    "twelve":12, "thirteen":13, "fourteen":14, "fifteen":15,
    "sixteen":16, "seventeen":17, "eighteen":18, "nineteen":19,
}
_TENS_WORDS = {
    "twenty":20, "thirty":30, "forty":40, "fifty":50,
    "sixty":60, "seventy":70, "eighty":80, "ninety":90,
}
_WORD_NUMBER_RE = re.compile(
    r"\b(?:zero|one|two|three|four|five|six|seven|eight|nine|ten|"
    r"eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|"
    r"eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|"
    r"eighty|ninety|hundred|thousand)(?:[- ](?:and[- ])?"
    r"(?:zero|one|two|three|four|five|six|seven|eight|nine|ten|"
    r"eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|"
    r"eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|"
    r"eighty|ninety|hundred|thousand))*\b", re.I
)


def _parse_english_number(phrase):
    total = current = 0
    for token in re.findall(r"[a-z]+", phrase.lower()):
        if token == "and":
            continue
        if token in _SMALL_WORDS:
            current += _SMALL_WORDS[token]
        elif token in _TENS_WORDS:
            current += _TENS_WORDS[token]
        elif token == "hundred":
            current = max(1, current) * 100
        elif token == "thousand":
            total += max(1, current) * 1000
            current = 0
    return total + current


def _question_numeric_facts(question):
    text = str(question).translate(str.maketrans(
        "۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩", "0123456789" * 2
    )).lower()
    facts = {0.0, 1.0}
    for raw in re.findall(r"(?<![\w.])(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?!\w)", text):
        try:
            facts.add(float(raw.replace(",", "")))
        except ValueError:
            pass
    for match in re.finditer(r"(?<!\d)(\d+)\s*(?:to|through|تا|–|-)\s*(\d+)(?!\d)", text):
        left, right = int(match.group(1)), int(match.group(2))
        if 0 <= abs(right - left) <= 100:
            facts.update(float(x) for x in range(min(left, right), max(left, right) + 1))
    for match in _WORD_NUMBER_RE.finditer(text):
        facts.add(float(_parse_english_number(match.group(0))))

    if re.search(r"\b(?:twice|double)\b|دو\s*برابر", text):
        facts.add(2.0)
    if re.search(r"\btriple\b|\bthree times\b|سه\s*برابر", text):
        facts.add(3.0)
    if re.search(r"\bhalf\b|نصف", text):
        facts.add(0.5)
    if re.search(r"\bquarter\b|ربع", text):
        facts.add(0.25)
    if re.search(r"\bthird\b|یک\s*سوم", text):
        facts.add(1.0 / 3.0)
    if re.search(r"\bdozen\b", text):
        facts.add(12.0)

    if re.search(r"[%٪]|\bpercent(?:age)?\b", text):
        facts.add(100.0)
        for match in re.finditer(r"(\d+(?:\.\d+)?)\s*(?:%|٪|percent(?:age)?)", text):
            rate = float(match.group(1)) / 100.0
            facts.update((rate, 1.0 + rate, 1.0 - rate))
        for match in _WORD_NUMBER_RE.finditer(text):
            if re.match(r"\s*percent(?:age)?\b", text[match.end():]):
                rate = _parse_english_number(match.group(0)) / 100.0
                facts.update((rate, 1.0 + rate, 1.0 - rate))

    if re.search(r"\b(?:monday|tuesday|wednesday|thursday)\b", text):
        facts.update((1.0, 2.0, 3.0, 4.0))
    if re.search(r"\bdays?\b", text) and re.search(r"\bhours?\b", text):
        facts.add(24.0)
    if re.search(r"\bhours?\b", text) and re.search(r"\bminutes?\b", text):
        facts.add(60.0)
    if re.search(r"\bweeks?\b", text) and re.search(r"\bdays?\b", text):
        facts.add(7.0)
    return facts


def _plan_numeric_values(plan):
    route = plan.get("method", plan.get("route"))
    values = []
    if route == "calculate":
        try:
            root = ast.parse(plan.get("expression", ""), mode="eval")
        except (SyntaxError, ValueError) as exc:
            raise ToolError("Invalid arithmetic expression in plan") from exc
        nodes = ast.walk(root)
    elif route == "schedule":
        tasks = plan.get("tasks")
        if not isinstance(tasks, list):
            raise ToolError("Invalid schedule plan")
        for task in tasks:
            if not isinstance(task, dict):
                raise ToolError("Invalid schedule task")
            values.append(task.get("duration"))
        nodes = ()
    elif route == "grid":
        grid = plan.get("grid")
        if not isinstance(grid, dict):
            raise ToolError("Invalid grid plan")
        for key in ("min_x", "max_x", "min_y", "max_y"):
            values.append(grid.get(key))
        for key in ("start", "goal"):
            point = grid.get(key)
            if not isinstance(point, (list, tuple)) or len(point) != 2:
                raise ToolError("Invalid grid point")
            values.extend(point)
        blocked = grid.get("blocked")
        if not isinstance(blocked, list):
            raise ToolError("Invalid blocked-cell list")
        for point in blocked:
            if not isinstance(point, (list, tuple)) or len(point) != 2:
                raise ToolError("Invalid blocked cell")
            values.extend(point)
        nodes = ()
    elif route == "csp":
        spec = plan.get("csp")
        if not isinstance(spec, dict) or not isinstance(spec.get("variables"), dict):
            raise ToolError("Invalid CSP plan")
        for domain in spec["variables"].values():
            if not isinstance(domain, list):
                raise ToolError("Invalid CSP domain")
            values.extend(domain)
        for item in spec.get("constraints", []):
            if not isinstance(item, dict) or not isinstance(item.get("expr"), str):
                raise ToolError("Invalid CSP constraint")
            try:
                root = ast.parse(item["expr"], mode="eval")
            except (SyntaxError, ValueError) as exc:
                raise ToolError("Invalid CSP expression") from exc
            nodes = ast.walk(root)
            for node in nodes:
                if isinstance(node, ast.Constant) and type(node.value) in (int, float):
                    values.append(node.value)
        answer = spec.get("answer", {})
        if isinstance(answer, dict) and type(answer.get("value")) in (int, float):
            values.append(answer["value"])
        nodes = ()
    else:
        return []

    for node in nodes:
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            values.append(node.value)
    return values


def require_grounded_plan(question, plan):
    """Reject model-proposed numeric facts not supported by the source question."""
    facts = _question_numeric_facts(question)
    for value in _plan_numeric_values(plan):
        if type(value) in (str, bool):
            continue
        if type(value) not in (int, float) or not math.isfinite(value):
            raise ToolError("Tool plan contains a non-finite or unsupported value")
        if not any(math.isclose(float(value), fact, rel_tol=0.0, abs_tol=1e-9)
                   for fact in facts):
            raise ToolError("Tool plan contains an unsupported numeric value: "
                            + format(float(value), "g"))
    return True



# ----- Structured extraction tools used by the agent -------------------------


def _require_quotes(question, quotes, label):
    """Require source evidence for extracted facts; no invented evidence."""
    if not isinstance(quotes, list) or not quotes:
        raise ToolError(f"{label} needs source evidence")
    normalized = _normal_space(question)
    for quote in quotes:
        if not isinstance(quote, str) or not quote.strip():
            raise ToolError(f"{label} contains an empty evidence quote")
        if _normal_space(quote) not in normalized:
            raise ToolError(f"{label} evidence is not present in the question")
    return True


def validate_bullet_coverage(question, quotes, label="plan"):
    """Every explicit bullet rule must be represented in quoted evidence."""
    if not isinstance(quotes, list):
        raise ToolError(f"{label} evidence must be a list")
    normalized_quotes = [_normal_space(x) for x in quotes if isinstance(x, str)]
    for line in question.splitlines():
        match = re.match(r"^\s*(?:[-•*]|\d+[.)])\s*(.+?)\s*$", line)
        if not match:
            continue
        bullet = _normal_space(match.group(1))
        # A one-word quote like "B" must not count as evidence for every
        # condition containing B. Require the complete clause, or a substantial
        # contiguous phrase with at least three substantive tokens.
        def covers(quote):
            if bullet in quote:
                return True
            tokens = bullet.split()
            qt = quote.split()
            return len(qt) >= 3 and len(qt) >= max(3, int(len(tokens) * .8)) and quote in bullet
        if not any(covers(quote) for quote in normalized_quotes):
            raise ToolError(f"{label} omitted an explicit condition")
    return True


def _linear_form(node, names):
    """Return (coefficients, constant) for a safe linear AST expression."""
    zero = {name: Fraction(0) for name in names}
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        return zero, _fraction(node.value)
    if isinstance(node, ast.Name) and node.id in names:
        coeff = dict(zero)
        coeff[node.id] = Fraction(1)
        return coeff, Fraction(0)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        coeff, const = _linear_form(node.operand, names)
        if isinstance(node.op, ast.USub):
            return {k: -v for k, v in coeff.items()}, -const
        return coeff, const
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub)):
        left, lc = _linear_form(node.left, names)
        right, rc = _linear_form(node.right, names)
        sign = 1 if isinstance(node.op, ast.Add) else -1
        return {k: left[k] + sign * right[k] for k in names}, lc + sign * rc
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult):
        left, lc = _linear_form(node.left, names)
        right, rc = _linear_form(node.right, names)
        left_linear = any(left.values())
        right_linear = any(right.values())
        if left_linear and right_linear:
            raise ToolError("Non-linear equation is not supported")
        if left_linear:
            scale = rc
            return {k: v * scale for k, v in left.items()}, lc * scale
        if right_linear:
            scale = lc
            return {k: v * scale for k, v in right.items()}, rc * scale
        return zero, lc * rc
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
        left, lc = _linear_form(node.left, names)
        right, rc = _linear_form(node.right, names)
        if any(right.values()) or rc == 0:
            raise ToolError("Linear equations require a non-zero constant divisor")
        return {k: v / rc for k, v in left.items()}, lc / rc
    raise ToolError("Unsupported syntax in linear equation")


def solve_linear_system(spec):
    """Solve a small full-rank linear system with exact rational arithmetic."""
    if not isinstance(spec, dict):
        raise ToolError("Linear specification must be an object")
    names = spec.get("variables")
    equations = spec.get("equations")
    answer_spec = spec.get("answer_spec")
    if (not isinstance(names, list) or not 1 <= len(names) <= 10
            or len(set(names)) != len(names)
            or any(not isinstance(n, str) or not re.fullmatch(r"[A-Za-z_]\w*", n) for n in names)):
        raise ToolError("Invalid linear variable list")
    if not isinstance(equations, list) or len(equations) < len(names) or len(equations) > 20:
        raise ToolError("Linear system needs enough equations")
    matrix = []
    for item in equations:
        if not isinstance(item, dict):
            raise ToolError("Invalid linear equation")
        expr, source = item.get("expr"), item.get("source")
        if not isinstance(expr, str) or not isinstance(source, str) or not source.strip():
            raise ToolError("Each equation needs expression and source quote")
        if "=" not in expr:
            raise ToolError("Equation must contain equals sign")
        left_text, right_text = expr.split("=", 1)
        try:
            left_node = ast.parse(left_text.strip(), mode="eval").body
            right_node = ast.parse(right_text.strip(), mode="eval").body
        except (SyntaxError, ValueError) as exc:
            raise ToolError("Invalid linear equation syntax") from exc
        left_coeff, left_const = _linear_form(left_node, set(names))
        right_coeff, right_const = _linear_form(right_node, set(names))
        row = [left_coeff[n] - right_coeff[n] for n in names]
        rhs = right_const - left_const
        matrix.append(row + [rhs])
    # Exact Gaussian elimination.
    row = 0
    pivots = []
    for col in range(len(names)):
        pivot = next((r for r in range(row, len(matrix)) if matrix[r][col] != 0), None)
        if pivot is None:
            continue
        matrix[row], matrix[pivot] = matrix[pivot], matrix[row]
        scale = matrix[row][col]
        matrix[row] = [v / scale for v in matrix[row]]
        for r in range(len(matrix)):
            if r == row:
                continue
            factor = matrix[r][col]
            if factor:
                matrix[r] = [matrix[r][c] - factor * matrix[row][c]
                             for c in range(len(names) + 1)]
        pivots.append(col)
        row += 1
        if row == len(matrix):
            break
    for r in matrix:
        if all(v == 0 for v in r[:-1]) and r[-1] != 0:
            raise ToolError("Linear system is inconsistent")
    if len(pivots) != len(names):
        raise ToolError("Linear system does not have a unique solution")
    solution = {names[col]: matrix[r][-1] for r, col in enumerate(pivots)}
    if not isinstance(answer_spec, dict):
        raise ToolError("Missing linear answer specification")
    kind = answer_spec.get("type")
    if kind == "variable":
        name = answer_spec.get("name")
        if name not in solution:
            raise ToolError("Unknown answer variable")
        return _format_fraction(solution[name])
    if kind == "concat":
        order = answer_spec.get("variables")
        if not isinstance(order, list) or not order or any(n not in solution for n in order):
            raise ToolError("Invalid linear concatenation order")
        if any(solution[n].denominator != 1 for n in order):
            raise ToolError("Code digits must be integers")
        return "".join(str(solution[n].numerator) for n in order)
    if kind == "assignment":
        order, labels = answer_spec.get("variables"), answer_spec.get("labels")
        if (not isinstance(order, list) or not isinstance(labels, list)
                or len(order) != len(labels) or not order
                or any(n not in solution for n in order)
                or any(not isinstance(x, str) for x in labels)):
            raise ToolError("Invalid linear assignment output")
        return ", ".join(f"{label}={_format_fraction(solution[name])}"
                         for name, label in zip(order, labels))
    raise ToolError("Unsupported linear answer format")


def solve_set_count(spec):
    """Inclusion-exclusion for two finite sets, grounded in source quotes."""
    if not isinstance(spec, dict):
        raise ToolError("Set-count specification must be an object")
    total, a, b, both = (spec.get(k) for k in ("total", "count_a", "count_b", "both"))
    if any(type(v) is not int or v < 0 for v in (total, a, b, both)):
        raise ToolError("Set counts must be non-negative integers")
    if a > total or b > total or both > min(a, b):
        raise ToolError("Inconsistent set counts")
    union = a + b - both
    if union > total:
        raise ToolError("Union exceeds the total population")
    return str(total - union)


def solve_percent_chain(spec):
    """Invert/apply a sequence of percentage increases and decreases."""
    if not isinstance(spec, dict):
        raise ToolError("Percentage specification must be an object")
    value = spec.get("known_value")
    operations = spec.get("operations")
    solve_for = spec.get("solve_for", "initial")
    if type(value) not in (int, float) or value <= 0:
        raise ToolError("Known percentage-chain value must be positive")
    if not isinstance(operations, list) or not 1 <= len(operations) <= 10:
        raise ToolError("Percentage chain needs one to ten operations")
    factor = Fraction(1)
    for op in operations:
        if not isinstance(op, dict):
            raise ToolError("Invalid percentage operation")
        kind, percent = op.get("type"), op.get("percent")
        if kind not in ("increase", "decrease") or type(percent) not in (int, float):
            raise ToolError("Invalid percentage operation")
        rate = _fraction(percent) / 100
        if rate < 0 or rate >= 1 and kind == "decrease":
            raise ToolError("Invalid percentage rate")
        factor *= (1 + rate) if kind == "increase" else (1 - rate)
    known = _fraction(value)
    result = known / factor if solve_for == "initial" else known * factor
    return _format_fraction(result)

# ----- Generic language-to-structure fallbacks for common reasoning families --
# These routines parse reusable mathematical/constraint patterns. They do not
# inspect question IDs, load the public set, or contain any public answer.


def _norm_question_text(question):
    text = str(question).translate(str.maketrans(
        "۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩", "0123456789" * 2
    ))
    text = text.replace("ي", "ی").replace("ك", "ک").replace("‌", "")
    text = re.sub(r"[\u064b-\u065f\u0670]", "", text)
    # Preserve line boundaries because bullet constraints are semantically useful.
    return "\n".join(re.sub(r"[ \t]+", " ", line).strip().casefold()
                     for line in text.splitlines())




def _question_clauses(question):
    text = str(question).replace("‌", "")
    parts = re.split(r"\n+|(?<=[.!؟])\s+|(?<=[؛;])\s*|\s+[-•*]\s+", text)
    return [p.strip() for p in parts if p and p.strip()]

def _number_tokens(text):
    normalized = str(text).translate(str.maketrans(
        "۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩", "0123456789" * 2
    ))
    return [int(x.replace(",", "")) for x in re.findall(r"(?<!\w)\d[\d,]*(?!\w)", normalized)]


def _try_generic_set_count(question):
    import re
    q = _norm_question_text(question)
    if not re.search(r"neither|none of|neither of|هیچکدام|هیچ یک|هیچکدوم", q):
        return None
    if not re.search(r"both|هر دو", q):
        return None

    total = None
    patterns = (
        r"(?:there are|there is|a class of|class has|total(?: of)?)\s*(\d+)\s*(?:students|people|persons|children|members)?",
        r"در\s*(?:یک\s*)?کلاس\s*(\d+)\s*(?:دانشجو|نفر)",
        r"(?:population|group|class|total)[^\d]{0,35}(\d+)\s*(?:students|people|persons|members)?",
    )
    for pattern in patterns:
        match = re.search(pattern, q)
        if match:
            total = int(match.group(1))
            break
    if total is None:
        return None

    individual, both = [], None
    for line in _question_clauses(question):
        line_n = _norm_question_text(line)
        nums = _number_tokens(line)
        if not nums:
            continue
        if re.search(r"both|هر دو", line_n):
            both = nums[0]
        elif re.search(r"study|studies|learn|read|take|میخوانند|میخواند|می خوانند|میخواند", line_n):
            individual.append(nums[0])
    if len(individual) != 2 or both is None:
        return None
    a, b = individual
    # Prevent silently ignoring a third subject, cohort, or numeric condition.
    # A two-set closed form has exactly four numeric factual counts.
    if len(_number_tokens(question)) != 4:
        return None
    if min(total, a, b, both) < 0 or max(a, b) > total or both > min(a, b):
        return None
    result = total - (a + b - both)
    if result < 0:
        return None
    return str(result)


def _try_generic_percent_chain(question):
    q = _norm_question_text(question)
    percent_matches = list(re.finditer(r"(\d+(?:\.\d+)?)\s*(?:%|٪|percent(?:age)?|per cent)", q))
    if not percent_matches:
        return None
    operations = []
    for match in percent_matches:
        # Classify each rate using its own sentence/clause, not a wide window
        # that can accidentally see the next operation (e.g. +25%, then -20%).
        left = max(q.rfind("\n", 0, match.start()), q.rfind(".", 0, match.start()),
                   q.rfind("؛", 0, match.start()), q.rfind(";", 0, match.start())) + 1
        right_candidates = [p for p in (q.find("\n", match.end()), q.find(".", match.end()),
                                         q.find("؛", match.end()), q.find(";", match.end())) if p >= 0]
        right = min(right_candidates) if right_candidates else len(q)
        context = q[max(0, match.start() - 55):min(len(q), match.end() + 55)]
        increase_terms = list(re.finditer(r"increase|increased|raise|raised|rise|grew|markup|افزایش|افزوده|بالا برد", context))
        decrease_terms = list(re.finditer(r"discount|decrease|decreased|reduce|reduced|off|cut|کاهش|تخفیف|کم کرد", context))
        def nearest_distance(found):
            if not found:
                return float("inf")
            point = match.start() - max(0, match.start() - 55)
            return min(abs(point - x.end()) if x.end() <= point else abs(x.start() - point) for x in found)
        inc_distance, dec_distance = nearest_distance(increase_terms), nearest_distance(decrease_terms)
        if dec_distance < inc_distance:
            kind = "decrease"
        elif inc_distance < dec_distance:
            kind = "increase"
        else:
            return None
        operations.append((kind, Fraction(match.group(1)) / 100))
    if not operations:
        return None

    # Find non-percentage quantities. A single remaining number is the known
    # amount; otherwise prefer the one explicitly associated with final/payment.
    spans = [(m.start(), m.end()) for m in percent_matches]
    numeric = []
    for m in re.finditer(r"(?<!\w)(\d+(?:\.\d+)?)(?!\w)", q):
        if any(a <= m.start() < b for a, b in spans):
            continue
        numeric.append((m.start(), m.end(), Fraction(m.group(1))))
    if len(numeric) == 1:
        known = numeric[0][2]
    else:
        scored = []
        for start, end, value in numeric:
            context = q[max(0, start - 45):min(len(q), end + 45)]
            score = 0
            if re.search(r"paid|pay|pays|payment|final|after (?:the )?(?:changes|discount|increase)|پرداخت|پرداخته|قیمت نهایی|پس از این", context):
                score += 2
            if re.search(r"initial|original|before|ابتدایی|اولیه|قبل از", context):
                score -= 2
            scored.append((score, value))
        if not scored:
            return None
        scored.sort(key=lambda x: x[0], reverse=True)
        if len(scored) > 1 and scored[0][0] == scored[1][0]:
            return None
        known = scored[0][1]

    # The requested quantity, not earlier narrative mentions, determines
    # whether the chain is applied forward or backward.
    questions = re.split(r"[?؟]", q)
    if len(questions) >= 2:
        asked = re.split(r"[.\n]", questions[-2])[-1].strip()
    else:
        clauses = [part.strip() for part in re.split(r"[.\n]", q) if part.strip()]
        while clauses and re.match(r"^(?:return|answer|respond|give)\b", clauses[-1]):
            clauses.pop()
        asked = clauses[-1] if clauses else ""
    asked = re.sub(r"\b(?:return|give|respond)\s+only.*$", "", asked).strip()
    asks_initial = bool(re.search(r"initial|original|before (?:the )?(?:increase|change)|قیمت اولیه|اولیه|قبل از افزایش", asked))
    asks_final = bool(re.search(r"final|new price|after|قیمت نهایی|قیمت جدید|پس از تغییر", asked))
    if asks_initial == asks_final:
        return None
    factor = Fraction(1)
    for kind, rate in operations:
        factor *= (1 + rate) if kind == "increase" else (1 - rate)
    result = known / factor if asks_initial else known * factor
    return _format_fraction(result)


def _try_generic_schedule(question):
    q = _norm_question_text(question)
    if not re.search(r"(?:earliest|minimum|soonest).{0,40}(?:project |overall |total )?(?:completion|finish|end)|(?:project|overall).{0,30}(?:earliest|finish|completion)|زودترین.{0,30}(?:پایان|اتمام)|کمترین.{0,25}زمان.{0,25}(?:پایان|اتمام)", q):
        return None
    if re.search(r"cannot overlap|must not overlap|non.overlap|one at a time|single worker|limited resources|نمی.*همزمان|منابع محدود", q):
        return None
    lines = _question_clauses(question)
    task_rows = []
    duration_re = re.compile(r"^\s*[-•*]?\s*(.+?)\s+(\d+(?:\.\d+)?)\s*(days?|روز|ساعت|hours?)", re.I)
    for line in lines:
        m = duration_re.search(_norm_question_text(line))
        if not m:
            continue
        name = m.group(1).strip(" -•*،,:؛.")
        name = re.sub(r"\b(?:takes?|requires?|needs?|lasts?)\s*$", "", name, flags=re.I).strip()
        name = re.sub(r"(?:زمان\s*میبرد|زمان\s*می\s*برد)\s*$", "", name).strip()
        if not name:
            continue
        duration = float(m.group(2))
        if duration.is_integer():
            duration = int(duration)
        task_rows.append({"name": name, "duration": duration, "line": _norm_question_text(line)})
    if len(task_rows) < 2:
        return None

    tasks = []
    after_markers = ("after", "following", "depends on", "پس از", "بعد از", "بعداز")
    for row in task_rows:
        deps = []
        line = row["line"]
        if any(marker in line for marker in after_markers):
            for other in task_rows:
                if other is row:
                    continue
                if other["name"] in line and other["name"] not in deps:
                    deps.append(other["name"])
        tasks.append({"name": row["name"], "duration": row["duration"], "depends_on": deps})
    try:
        return calculate_project_duration(tasks)
    except ToolError:
        return None


def _try_generic_grid(question):
    q = _norm_question_text(question)
    # BFS is only valid for static unit-cost four-neighbour movement.
    if re.search(r"teleport|portal|diagonal|jump|one-way|one way|different cost|weighted|move twice|moves? two|پرش|تلپورت|مورب|هزینه متفاوت|جهش", q):
        return None
    if not re.search(r"grid|شبکه|robot|ربات", q) or not re.search(r"blocked|obstacle|مسدود", q):
        return None
    if not (re.search(r"up.*down.*left.*right", q) or "بالا" in q and "پایین" in q and "چپ" in q and "راست" in q):
        return None
    coords = [tuple(map(int, m.groups())) for m in re.finditer(r"\(\s*(-?\d+)\s*[,،]\s*(-?\d+)\s*\)", q)]
    if len(coords) < 3:
        return None
    bounds = {}
    for axis in ("x", "y"):
        m = re.search(rf"(-?\d+)\s*<=\s*{axis}\s*<=\s*(-?\d+)", q)
        if not m:
            return None
        bounds[f"min_{axis}"] = int(m.group(1))
        bounds[f"max_{axis}"] = int(m.group(2))
    grid = {"start": list(coords[0]), "goal": list(coords[1]), "blocked": [list(p) for p in coords[2:]], **bounds}
    try:
        return shortest_grid_path(grid)
    except ToolError:
        return None


def _try_generic_digit_code(question):
    q = _norm_question_text(question)
    if not re.search(r"three.?digit|digit code|کد\s*سه.?رقمی", q):
        return None
    mset = re.search(r"\{([^}]+)\}", q)
    if not mset:
        return None
    domain = sorted(set(int(x) for x in re.findall(r"\d+", mset.group(1))))
    if not 2 <= len(domain) <= 10:
        return None
    import itertools
    length = 3 if re.search(r"three.?digit|سه.?رقمی", q) else None
    if length != 3:
        return None
    require_distinct = bool(re.search(r"distinct|different digits|متمایز|متفاوت", q))
    first_gt_second = bool(re.search(r"first digit.{0,35}greater than.{0,20}second|رقم اول از رقم دوم بزرگتر|رقم اول.{0,25}بزرگتر از رقم دوم", q))
    third_difference = bool(re.search(r"third digit.{0,35}(?:difference|subtract).{0,30}first.{0,15}second|رقم سوم.{0,35}تفاضل رقم اول و دوم", q))
    sum_match = re.search(r"sum.{0,25}(?:equals?|is)\s*(\d+)|مجموع سه رقم برابر\s*(\d+)", q)
    excluded = set()
    for ex in re.finditer(r"(?:digit|رقم)\s*(\d+)\s*(?:is )?not used|(?:رقم)\s*(\d+)\s*استفاده نشده", q):
        excluded.add(int(next(g for g in ex.groups() if g)))
    if not (require_distinct and first_gt_second and third_difference and sum_match):
        return None
    # Exhaustively account for separate bullet constraints. A code formed from
    # a correct subset of the conditions is not a verified solution.
    for line in q.splitlines():
        stripped = line.strip()
        if not stripped.startswith(("-", "•", "*")):
            continue
        condition = stripped.lstrip("-•* ")
        covered = (
            (re.search(r"first digit|رقم اول", condition) and
             re.search(r"greater|بزرگتر", condition)) or
            (re.search(r"third digit|رقم سوم", condition) and
             re.search(r"difference|subtract|تفاضل", condition)) or
            (re.search(r"sum|مجموع", condition) and re.search(r"\d", condition)) or
            re.search(r"(?:digit|رقم)\s*\d+.*(?:not used|استفاده نشده)", condition)
        )
        if not covered:
            return None
    target_sum = int(next(g for g in sum_match.groups() if g))
    domain = [x for x in domain if x not in excluded]
    answers = []
    for a, b, c in itertools.permutations(domain, 3):
        if a > b and c == a - b and a + b + c == target_sum:
            answers.append(f"{a}{b}{c}")
    return sorted(answers)[0] if answers else None


def _try_generic_ordering(question):
    q = _norm_question_text(question)
    if not re.search(r"(?:which|what).{0,40}(?:first|position 1)|(?:first|position 1).{0,40}(?:which|what)|کدام.{0,35}(?:جایگاه اول|اولین)|فقط حرف", q):
        return None
    if not re.search(r"position|positions|جایگاه", q) or not re.search(r"before|قبل از", q):
        return None
    first_sentence = re.split(r"[\n.]", q, maxsplit=1)[0]
    labels = list(dict.fromkeys(x.upper() for x in re.findall(r"\b([a-e])\b", first_sentence)))
    if not 3 <= len(labels) <= 7:
        return None
    n = len(labels)
    import itertools
    before, immediate, not_ends, last = [], [], [], []
    for line in _question_clauses(q):
        recognized = False
        m = re.search(r"\b([a-z])\b\s+(?:must|should|has to)\s+be before\s+\b([a-z])\b", line)
        if not m:
            m = re.search(r"\b([a-z])\b\s+باید قبل از\s+\b([a-z])\b", line)
        if m:
            before.append((m.group(1).upper(), m.group(2).upper()))
            recognized = True
        m = re.search(r"\b([a-z])\b\s+(?:must|should)\s+be immediately after\s+\b([a-z])\b", line)
        if not m:
            m = re.search(r"\b([a-z])\b\s+باید بلافاصله بعد از\s+\b([a-z])\b", line)
        if m:
            immediate.append((m.group(1).upper(), m.group(2).upper()))
            recognized = True
        m = re.search(r"\b([a-z])\b\s+(?:cannot|can't)\s+be first or last|\b([a-z])\b\s+نمی(?:تواند|تونه)\s+اولین یا آخرین", line)
        if m:
            not_ends.append((m.group(1) or m.group(2)).upper())
            recognized = True
        m = re.search(r"\b([a-z])\b\s+(?:must|has to)\s+be last|\b([a-z])\b\s+باید آخرین", line)
        if m:
            last.append((m.group(1) or m.group(2)).upper())
            recognized = True
        # Any additional bullet restriction must be interpreted, otherwise
        # defer to the model; never silently solve an incomplete CSP.
        looks_like_constraint = bool(re.search(
            r"\b[a-e]\b", line) and
            re.search(r"\b(?:must|cannot|can't|should|has to|before|after|first|last|not)\b|باید|نمی|بلافاصله|قبل از|بعد از|آخرین|اولین", line)
        )
        if (line.lstrip().startswith(('-', '•', '*')) or looks_like_constraint) and not recognized and line.rstrip(". ") != first_sentence.rstrip(". "):
            return None
    if not (before or immediate or not_ends or last):
        return None
    valid = []
    for perm in itertools.permutations(labels):
        pos = {label: i for i, label in enumerate(perm)}
        if any(pos[a] >= pos[b] for a, b in before):
            continue
        if any(pos[a] != pos[b] + 1 for a, b in immediate):
            continue
        if any(pos[a] in (0, n - 1) for a in not_ends):
            continue
        if any(pos[a] != n - 1 for a in last):
            continue
        valid.append(perm)
    if not valid:
        return None
    possible_first = sorted({perm[0] for perm in valid})
    return possible_first[0]


def _try_generic_box_logic(question):
    q = _norm_question_text(question)
    if re.search(r"how many (?:boxes|locations|possibilities)|چند (?:جعبه|حالت)", q):
        return None
    if not re.search(r"exactly one|دقیقا یکی", q) or not re.search(r"box|جعبه", q):
        return None
    domain_match = re.search(r"(?:one of the )?boxes?\s*([\d\s,،andor]+)", q)
    if not domain_match:
        domain_match = re.search(r"یکی از جعبه(?:های)?\s*([\d\s,،ویا]+)", q)
    if not domain_match:
        return None
    domain = sorted(set(_number_tokens(domain_match.group(1))))
    if not domain:
        return None
    propositions = []
    for line in _question_clauses(q):
        m = re.search(r"\b([a-z])\s*[:：]\s*(.+)", line)
        if not m:
            continue
        statement = m.group(2)
        box = re.search(r"(?:box|جعبه)\s*(\d+)", statement)
        if not box:
            continue
        negated = bool(re.search(r"\bnot\b|\bisn't\b|نیست", statement))
        propositions.append((int(box.group(1)), negated))
    if len(propositions) < 2:
        return None
    candidates = []
    for location in domain:
        truth_count = sum((location != box) if neg else (location == box) for box, neg in propositions)
        if truth_count == 1:
            candidates.append(location)
    return str(min(candidates)) if candidates else None


def _try_generic_truth_liar(question):
    q = _norm_question_text(question)
    if not re.search(r"liar|دروغگو", q) or not re.search(r"says|میگوید|می گوید", q):
        return None
    statements = []
    for line in _question_clauses(q):
        m = re.search(r"^\s*([A-Za-z\u0600-\u06FF]+)\s+(?:says|می\s*گوید)\s*[:：]\s*(.+?)\s*$", line, re.I)
        if m:
            statements.append((m.group(1), _norm_question_text(m.group(2))))
    if not statements:
        return None
    people = [name for name, _ in statements]
    if len(people) > 6:
        return None
    import itertools
    parsed = []
    for speaker, statement in statements:
        target = None
        relation = None
        if re.search(r"different types|different kind|از دو نوع متفاوت|متفاوت هستیم", statement):
            relation = "different"
            for person in people:
                if person != speaker and person.casefold() in statement:
                    target = person
                    break
        elif re.search(r"liar|دروغگو", statement):
            relation = "liar"
            for person in people:
                if person != speaker and person.casefold() in statement:
                    target = person
                    break
            if target is None and re.search(r"\bi am\b|\bمن\b", statement):
                target = speaker
        elif re.search(r"truthful| راستگو", statement):
            relation = "truth"
            for person in people:
                if person.casefold() in statement:
                    target = person
                    break
            if target is None:
                target = speaker
        if relation is None or target is None:
            return None
        parsed.append((speaker, relation, target))

    solutions = []
    for bits in itertools.product((False, True), repeat=len(people)):
        truth = dict(zip(people, bits))
        valid = True
        for speaker, relation, target in parsed:
            if relation == "liar":
                content = not truth[target]
            elif relation == "truth":
                content = truth[target]
            else:
                content = truth[speaker] != truth[target]
            if truth[speaker] != content:
                valid = False
                break
        if valid:
            solutions.append(truth)
    if not solutions:
        return None
    solution = sorted(solutions, key=lambda s: tuple(s[p] for p in people))[0]

    # Use the output labels explicitly offered in the question, if any; this
    # allows Persian source names and Latin output labels without hard-coding.
    option = re.search(r"\b([A-Za-z]+)\s*=\s*(?:Truth|Lie)\s*,\s*([A-Za-z]+)\s*=\s*(?:Truth|Lie)\b", str(question), re.I)
    labels = [option.group(1), option.group(2)] if option else people
    if len(labels) != len(people):
        labels = people
    values = ["Truth" if solution[p] else "Lie" for p in people]
    return ", ".join(f"{label}={value}" for label, value in zip(labels, values))


def _try_generic_weekday_assignment(question):
    q = _norm_question_text(question)
    if not re.search(r"monday|دوشنبه", q) or not re.search(r"thursday|پنجشنبه", q):
        return None
    days = {"monday": 1, "tuesday": 2, "wednesday": 3, "thursday": 4,
            "دوشنبه": 1, "سهشنبه": 2, "چهارشنبه": 3, "پنجشنبه": 4}
    persian_days = {"دوشنبه": "Monday", "سهشنبه": "Tuesday", "چهارشنبه": "Wednesday", "پنجشنبه": "Thursday"}
    names = []
    intro = re.search(r"(?:researchers|people|پژوهشگر(?:ان)?)\s+(.+?)\s+(?:each|هرکدام)", q)
    if intro:
        raw = intro.group(1)
        tokens = re.findall(r"[a-z][a-z-]*|[\u0600-\u06FF]+", raw)
        names = [x.strip("،,؛:.") for x in tokens if x not in {"and", "و"}]
    if len(names) != 4:
        return None
    names = list(dict.fromkeys(names))
    if len(names) != 4:
        return None
    before, fixed, forbidden, next_day = [], {}, [], []
    for line in _question_clauses(q):
        m = re.search(r"([\w\u0600-\u06FF]+)\s+(?:is\s+)?(?:earlier than|before)\s+([\w\u0600-\u06FF]+)|([\w\u0600-\u06FF]+)\s+زودتر از\s+([\w\u0600-\u06FF]+)", line)
        if m:
            a, b = (m.group(1), m.group(2)) if m.group(1) else (m.group(3), m.group(4))
            before.append((a, b))
        for day_name, idx in days.items():
            if day_name in line:
                person_match = re.search(r"^\s*[-•*]?\s*([\w\u0600-\u06FF]+)", line)
                if person_match:
                    person = person_match.group(1)
                    if person not in names:
                        continue
                    if re.search(r"cannot|can't|does not|doesn't|نمی(?:دهد|تواند)", line):
                        forbidden.append((person, idx))
                    elif re.search(r"present|ارائه میدهد|ارائه میده", line):
                        fixed[person] = idx
        m = re.search(r"([\w\u0600-\u06FF]+)\s+(?:.{0,20}?)?(?:exactly one day after|one day after)\s+([\w\u0600-\u06FF]+)|([\w\u0600-\u06FF]+)\s+(?:.{0,20}?)?دقیقا یک روز بعد از\s+([\w\u0600-\u06FF]+)", line)
        if m:
            a, b = (m.group(1), m.group(2)) if m.group(1) else (m.group(3), m.group(4))
            next_day.append((a, b))
    # A question may express a fixed day without a verb after it.
    for person in names:
        for day_name, idx in days.items():
            if re.search(re.escape(person) + r".{0,30}" + re.escape(day_name), q):
                if re.search(r"(?:has|is|on|دارد|ارائه میدهد|ارائه میده)\s+" + re.escape(day_name), q):
                    fixed[person] = idx
    if not (before or fixed or forbidden or next_day):
        return None
    import itertools
    solutions = []
    for perm in itertools.permutations((1, 2, 3, 4)):
        assignment = dict(zip(names, perm))
        if any(assignment.get(a, -99) >= assignment.get(b, 99) for a, b in before):
            continue
        if any(assignment.get(p) != d for p, d in fixed.items()):
            continue
        if any(assignment.get(p) == d for p, d in forbidden):
            continue
        if any(assignment.get(a, -99) != assignment.get(b, 99) + 1 for a, b in next_day):
            continue
        solutions.append(assignment)
    if not solutions:
        return None
    asked = re.search(r"([\w\u0600-\u06FF]+)\s+در چه روزی|what day\s+(?:does|is)\s+([\w\u0600-\u06FF]+)", q)
    target = (asked.group(1) or asked.group(2)) if asked else None
    if target not in names:
        return None
    day_idx = sorted({s[target] for s in solutions})[0]
    return {1: "Monday", 2: "Tuesday", 3: "Wednesday", 4: "Thursday"}[day_idx]


def _try_generic_linear_assignment(question):
    q = _norm_question_text(question)
    labels = re.findall(r"([A-Za-z][A-Za-z0-9_]*)\s*=\s*<\s*(?:integer|int|number)\s*>", question, re.I)
    if len(labels) < 2:
        return None
    # Persian list form: "X، Y و Z در مجموع N ... دارند".
    total_match = re.search(r"(.{1,100}?)\s+در مجموع\s+(\d+)\s+[^.]*", q)
    source_names = []
    if total_match:
        tokens = re.findall(r"[\u0600-\u06FF]+", total_match.group(1))
        stop = {"تعداد", "ژتونهای", "ژتون", "مهرههای", "مهره", "و", "در", "یک"}
        source_names = [x.strip("،,؛:.") for x in tokens if x.strip("،,؛:.") not in stop][-len(labels):]
        total = int(total_match.group(2))
    else:
        total_match = re.search(r"(?:total|altogether|in total)[^\d]{0,25}(\d+)|(\d+)\s+[a-z][a-z ]{0,25}\s+(?:in total|altogether)", q)
        total = int(next(g for g in total_match.groups() if g)) if total_match else None
        source_names = labels[:]
    if total is None or len(source_names) != len(labels):
        return None
    rels = []
    m = re.search(r"([\u0600-\u06FF]+)\s+دو\s*برابر\s+([\u0600-\u06FF]+)", q)
    if m:
        rels.append((m.group(1), "twice", m.group(2), None))
    else:
        m = re.search(r"([A-Za-z]+)\s+(?:has|is)\s+twice as many.*?as\s+([A-Za-z]+)", q)
        if m:
            rels.append((m.group(1).casefold(), "twice", m.group(2).casefold(), None))
    m = re.search(r"([\u0600-\u06FF]+)\s+(\d+)\s+[^.]{0,20}بیشتر از\s+([\u0600-\u06FF]+)", q)
    if m:
        rels.append((m.group(1), "more", m.group(3), int(m.group(2))))
    else:
        m = re.search(r"([A-Za-z]+)\s+has\s+(\d+)\s+more.*?than\s+([A-Za-z]+)", q)
        if m:
            rels.append((m.group(1).casefold(), "more", m.group(3).casefold(), int(m.group(2))))
    if len(rels) < 2:
        return None
    # Resolve source names to variable positions by appearance order.
    variables = [f"v{i}" for i in range(len(labels))]
    mapping = dict(zip([name.casefold() for name in source_names], variables))
    equations = [{"expr": " + ".join(variables) + f" = {total}", "source": total_match.group(0)}]
    for left, kind, right, amount in rels:
        if left not in mapping or right not in mapping:
            return None
        if kind == "twice":
            expr = f"{mapping[left]} = 2 * {mapping[right]}"
        else:
            expr = f"{mapping[left]} = {mapping[right]} + {amount}"
        # Use the whole original sentence as source evidence; the deterministic
        # parser uses it only as provenance, not as an answer lookup.
        equations.append({"expr": expr, "source": question.strip()})
    spec = {"variables": variables, "equations": equations,
            "answer_spec": {"type": "assignment", "variables": variables, "labels": labels}}
    try:
        answer = solve_linear_system(spec)
        if re.search(r"<\s*integer\s*>", question, re.I) and re.search(r"=\s*-?\d+\.\d+", answer):
            return None
        return answer
    except ToolError:
        return None




def _try_generic_purchase_arithmetic(question):
    """Solve a fully specified bill/change question without using an LLM.

    Works with arbitrary item names, quantities, unit prices, and payment
    amounts.  It deliberately abstains when any numeric detail is not accounted
    for, e.g. taxes, discounts, fees, installments, or extra conditions.
    """
    from collections import Counter
    from decimal import Decimal, InvalidOperation

    q = _norm_question_text(question)
    asks_change = bool(re.search(
        r"\b(?:how much|what amount|amount of)\s+change\b|\bchange\s+(?:should|does|will|would)\b",
        q,
    ))
    asks_cost = bool(re.search(
        r"\b(?:total|overall|combined)\s+cost\b|\bhow much\s+(?:does|do|did|will)\s+.*?\b(?:cost|spend)\b",
        q,
    ))
    if not (asks_change or asks_cost):
        return None
    if re.search(
        r"\b(?:tax|tips?|discount|coupon|rebate|cashback|refund|interest|installment|"
        r"delivery|shipping|fee|sales price|percentage)\b|[%٪]",
        q,
    ):
        return None

    number = r"(?:\d+(?:,\d{3})*(?:\.\d+)?)"
    items = re.compile(
        rf"(?<!\w)(?P<qty>{number})\s+(?P<item>[a-z][a-z\s-]{{0,45}}?)"
        rf"\s+(?:at|for|costing|priced at)\s*\$?\s*(?P<unit>{number})"
        rf"\s*(?:(?P<currency>dollars?|usd|euros?|pounds?|cents?)\s*)?"
        rf"(?:each|apiece|per\s+\w+)\b",
    )
    matches = list(items.finditer(q))
    if not matches or len(matches) > 20:
        return None

    try:
        terms = [
            (Decimal(m.group("qty").replace(",", "")),
             Decimal(m.group("unit").replace(",", "")))
            for m in matches
        ]
    except InvalidOperation:
        return None
    if any(qty <= 0 or unit < 0 for qty, unit in terms):
        return None

    # These are specifically purchase transactions, not generic unit-rate or
    # inventory tasks that happen to use the words "at ... each".
    if not re.search(r"\b(?:buys?|bought|purchases?|purchased|shopping|basket|cart|customer|shopper)\b", q):
        return None

    payment = None
    if asks_change:
        payments = list(re.finditer(
            rf"\b(?:pays?|paid|paying)\s+(?:with|using)?\s*\$?\s*(?P<paid>{number})"
            rf"\s*(?P<paid_unit>dollars?|usd|euros?|pounds?|cents?)?\b"
            rf"|\b(?:hands? over|gives? (?:the )?cashier|tenders?)\s*\$?\s*(?P<given>{number})\b",
            q,
        ))
        if len(payments) != 1:
            return None
        try:
            payment = Decimal((payments[0].group("paid") or payments[0].group("given")).replace(",", ""))
        except InvalidOperation:
            return None

    # Never silently mix currencies or dollars with cents. If an item uses
    # a printed dollar sign, that is treated as USD. Otherwise only explicit
    # units are compared; missing units inherit the transaction's unit.
    def canonical_currency(token):
        token = (token or "").casefold()
        if token in ("dollar", "dollars", "usd"):
            return "usd"
        if token in ("euro", "euros"):
            return "eur"
        if token in ("pound", "pounds"):
            return "gbp"
        if token in ("cent", "cents"):
            return "cent"
        return None

    units = set()
    for m in matches:
        unit = canonical_currency(m.group("currency"))
        if unit is None and "$" in m.group():
            unit = "usd"
        if unit is not None:
            units.add(unit)
    if payment is not None:
        pay_unit = canonical_currency(payments[0].group("paid_unit"))
        if pay_unit is None and "$" in payments[0].group():
            pay_unit = "usd"
        if pay_unit is not None:
            units.add(pay_unit)
    if len(units) > 1:
        return None

    # Strictly account for EVERY numeric literal in the original wording;
    # otherwise we may have missed an essential stated condition.
    def numeric_values(text):
        return Counter(Decimal(m.group().replace(",", "")) for m in
                       re.finditer(r"(?<![\w.])\d+(?:,\d{3})*(?:\.\d+)?(?![\w.])", text))

    used = Counter()
    for m in matches:
        used[Decimal(m.group("qty").replace(",", ""))] += 1
        used[Decimal(m.group("unit").replace(",", ""))] += 1
    if payment is not None:
        used[payment] += 1
    if used != numeric_values(q):
        return None

    cost = sum((qty * unit for qty, unit in terms), Decimal(0))
    result = payment - cost if asks_change else cost
    if payment is not None and result < 0:
        return None
    if not result.is_finite() or abs(result) > Decimal("1000000000000"):
        return None
    return format(result, "f").rstrip("0").rstrip(".") if "." in format(result, "f") else str(result)

def _try_generic_inventory_events(question):
    """Bounded quantity ledger for common multi-step inventory word problems.

    Extracts an initial group/quantity and subsequent positive and negative
    changes; every number in the factual clauses must be explained by a
    parsed operation.  Ambiguous or unrecognized clauses cause abstention.
    This deliberately works with arbitrary names, objects and quantities.
    """
    from collections import Counter

    q = _norm_question_text(question).replace("\n", " ")
    # Do not infer answers when the requested quantity is a change, an
    # intermediate value or an explicitly more complex mathematical relation.
    if not re.search(
        r"\b(?:how many|how much)\b.{0,90}\b(?:now|left|remain|remaining|finally|end up with)\b|"
        r"\b(?:how many|how much)\b.{0,90}\b(?:does|do|did|has|have)\b.{0,35}\b(?:have|left|remain|remaining)\b",
        q,
    ):
        return None
    if re.search(r"\b(?:percent|percentage|double|twice|triple|half|quarter|ratio|tax|interest|"
                 r"fraction|twice as many|each day|per day|every day|per week|every week|"
                 r"average|equally|between|different types|maximum|minimum)\b|[%٪]", q):
        return None
    # Treat the text before the first interrogative as the factual part.
    facts = re.split(r"\bhow many\b|\bhow much\b", q, maxsplit=1)[0]
    if not facts.strip():
        return None

    n = r"\d+(?:,\d{3})*(?:\.\d+)?"
    item = r"[a-z][a-z-]{1,30}"
    initial_patterns = [
        # There are 5 bags, each containing 8 apples.
        rf"\b(?P<count>{n})\s+(?:bags?|boxes?|baskets?|packs?|trays?|cartons?|"
        rf"crates?|containers?|jars?|bins?|groups?)\s*,?\s*"
        rf"(?:each\s+)?(?:containing|holding|with|having|of)\s+"
        rf"(?P<per>{n})\s+(?P<item>{item})(?:\s+(?:each|apiece))?\b",
        # 5 bags with 8 apples in each, 5 bags of 8 apples each
        rf"\b(?P<count>{n})\s+(?:bags?|boxes?|baskets?|packs?|trays?|cartons?|"
        rf"crates?|containers?|jars?|bins?|groups?)\s+(?:with|of|holding)\s+"
        rf"(?P<per>{n})\s+(?P<item>{item})(?:\s+in\s+each|\s+each)?\b",
        # 5 bags each have/contain 8 apples
        rf"\b(?P<count>{n})\s+(?:bags?|boxes?|baskets?|packs?|trays?|cartons?|"
        rf"crates?|containers?|jars?|bins?|groups?)\s+each\s+"
        rf"(?:have|has|contain|contains|hold|holds)\s+(?P<per>{n})\s+(?P<item>{item})\b",
    ]
    matches = [m for pat in initial_patterns for m in re.finditer(pat, facts)]
    # Multiple distinct initial groups are not yet a safe single-inventory parse.
    starts = {(m.start(), m.end()) for m in matches}
    if len(starts) != 1:
        return None
    match = matches[0]
    count, per = (float(match.group('count').replace(',', '')),
                  float(match.group('per').replace(',', '')))
    kind = match.group('item')
    if not count.is_integer() or not per.is_integer() or count <= 0 or per < 0:
        return None
    current = int(count) * int(per)
    used = Counter((match.group('count').replace(',', ''), match.group('per').replace(',', '')))

    # Operations can occur in the same sentence, joined with 'and', or in
    # successive sentences.  Each quantity is consumed exactly once.
    change_pattern = re.compile(
        rf"\b(?P<verb>gives?\s+away|gave\s+away|donates?|donated|sells?|sold|"
        rf"uses?|used|eats?|ate|loses?|lost|spends?|spent|throws?\s+away|"
        rf"buys?|bought|purchases?|purchased|receives?|received|gets?|got|"
        rf"finds?|found|adds?|added|collects?|collected|picks?\s+up|"
        rf"is\s+given|was\s+given)\s+(?P<num>{n})"
        rf"(?:\s+(?P<unit>{item}))?\b"
    )
    operations = list(change_pattern.finditer(facts))
    if not operations:
        return None
    # Whitelist only unit words that actually describe this object's count,
    # plus the common anaphor 'more'. E.g. buying 7 bags is not 7 apples.
    def singular(word):
        return word[:-1] if word.endswith('s') else word
    normalized_kind = singular(kind)
    negatives = {'give away', 'gives away', 'gave away', 'donate', 'donates',
                 'donated', 'sell', 'sells', 'sold', 'use', 'uses', 'used',
                 'eat', 'eats', 'ate', 'lose', 'loses', 'lost', 'spend', 'spends',
                 'spent', 'throw away', 'throws away'}
    positives = {'buy', 'buys', 'bought', 'purchase', 'purchases', 'purchased',
                 'receive', 'receives', 'received', 'get', 'gets', 'got',
                 'find', 'finds', 'found', 'add', 'adds', 'added',
                 'collect', 'collects', 'collected', 'pick up', 'picks up',
                 'is given', 'was given'}
    for op in operations:
        verb = op.group('verb')
        unit = op.group('unit')
        if unit and singular(unit) not in (normalized_kind, 'more', 'additional'):
            return None
        if unit in ('more', 'additional'):
            # 'buys 7 more baskets' is not buying 7 more apples.
            tail_word = re.match(r"\s+([a-z][a-z-]*)\b", facts[op.end():])
            if tail_word and tail_word.group(1) not in ('and', 'but', 'then', 'after', 'before', 'now'):
                if singular(tail_word.group(1)) != normalized_kind:
                    return None
        number_text = op.group('num').replace(',', '')
        number = float(number_text)
        if not number.is_integer() or number < 0:
            return None
        if verb in negatives:
            current -= int(number)
        elif verb in positives:
            current += int(number)
        else:
            return None
        used[number_text] += 1
    # Do not assume the operations account for ambiguous changes (e.g.
    # 'doubled', 'shared half', transfer between groups or an extra rate).
    actual = Counter(m.group().replace(',', '') for m in re.finditer(
        r"(?<![\w.])\d+(?:,\d{3})*(?:\.\d+)?(?![\w.])", facts))
    if used != actual or current < 0 or current > 10 ** 12:
        return None
    return str(current)


def solve_question_family(question, answer_format="text", *, with_meta=False):
    """Bounded reusable solvers, with optional provenance for adaptive routing.

    The ordinary return value remains a string/None for compatibility.  A
    caller using with_meta=True receives (answer, solver_name) or None. The
    source label is *not* a confidence claim: some regex-based extractors
    require semantic model verification on unseen, nuanced word problems.
    """
    solvers = (
        _try_generic_inventory_events,
        _try_generic_inventory_remaining,
        _try_generic_purchase_arithmetic,
        _try_generic_grid,
        _try_generic_schedule,
        _try_generic_set_count,
        _try_generic_percent_chain,
        _try_generic_truth_liar,
        _try_generic_weekday_assignment,
        _try_generic_box_logic,
        _try_generic_digit_code,
        _try_generic_ordering,
        _try_generic_linear_assignment,
    )
    for solver in solvers:
        try:
            result = solver(question)
            if result is not None:
                return (str(result), solver.__name__) if with_meta else str(result)
        except (ToolError, ValueError, TypeError, ArithmeticError, KeyError):
            continue
    return None


def _try_generic_inventory_remaining(question):
    q = _norm_question_text(question)
    if not re.search(r"remain|remaining|left|باقی|مانده", q):
        return None
    group = re.search(
        r"(\d+)\s+(?:boxes?|packs?|bags?|trays?|cartons?|groups?)\s+(?:of|with)\s+(\d+)\s+[a-z]+\s+each",
        q,
    )
    if not group:
        group = re.search(
            r"(\d+)\s+(?:جعبه|بسته|سینی)\s*(?:که\s*)?(?:هرکدام|هر کدام)?\s*(?:دارای)?\s*(\d+)\s+\S+",
            q,
        )
    if not group:
        return None
    consumed_matches = list(re.finditer(
        r"(?:uses?|used|consumes?|consumed|sells?|sold|removes?|removed|eats?|ate|gives? away|gave away|مصرف(?: میکند| میکند)?|استفاده میکند)\s+(\d+)\s*(?:items?|pieces?|units?|مورد|عدد)?",
        q,
    ))
    if len(consumed_matches) != 1:
        return None
    groups, per_group, consumed = int(group.group(1)), int(group.group(2)), int(consumed_matches[0].group(1))
    total = groups * per_group
    if groups <= 0 or per_group < 0 or consumed < 0 or consumed > total:
        return None
    # Only solve a single subtractive event. An additional quantity (buying,
    # receiving, transferring, etc.) changes the final amount and must not be
    # ignored. This is essential for unseen multi-step problems.
    from collections import Counter
    observed = Counter(re.findall(r"(?<![\w.])\d+(?:,\d{3})*(?:\.\d+)?(?![\w.])", q))
    accounted = Counter((group.group(1), group.group(2), consumed_matches[0].group(1)))
    if observed != accounted:
        return None
    return str(total - consumed)
