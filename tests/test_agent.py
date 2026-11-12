"""Offline regression tests. These tests do not contact Ollama or the internet."""
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agent import (ReasoningAgent, LocalOllama, normalize_answer, detect_format,
                   parse_json_object, OllamaClientError, AgentError, _execute_plan, select_tool_hint, plan_prompt)
from tools import (safe_arithmetic, calculate_project_duration, shortest_grid_path,
                   solve_csp, solve_linear_system, solve_set_count,
                   solve_percent_chain, solve_question_family, ToolError)
import main


class FakeLLM:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = []
    def chat(self, system, user, **kwargs):
        self.calls.append((system, user, kwargs))
        if not self.responses:
            raise AssertionError("Unexpected model call")
        result = self.responses.pop(0)
        if isinstance(result, BaseException):
            raise result
        return json.dumps(result, ensure_ascii=False) if isinstance(result, dict) else result
    def close(self):
        pass


class FormatTests(unittest.TestCase):
    def test_detect_assignment(self):
        self.assertEqual(detect_format("Nima=<integer>, Sara=<integer>, Reza=<integer>"), "assignment")
    def test_detect_integer_persian(self):
        self.assertEqual(detect_format("فقط عدد صحیح را برگردانید."), "integer")
    def test_detect_day(self):
        self.assertEqual(detect_format("فقط یکی از این چهار عبارت: Monday Tuesday Wednesday Thursday"), "day")
    def test_detect_code(self):
        self.assertEqual(detect_format("فقط کد سه‌رقمی را برگردانید."), "code")
    def test_detect_truth_lie_assignment_options(self):
        q = "Choose one:\nX=Truth, Y=Truth\nX=Truth, Y=Lie\nX=Lie, Y=Truth\nX=Lie, Y=Lie"
        self.assertEqual(detect_format(q), "assignment")
    def test_persian_digits(self):
        self.assertEqual(normalize_answer("۱۲۳", "integer"), "123")
    def test_preserve_code_zero(self):
        self.assertEqual(normalize_answer("007", "code"), "007")
    def test_reject_multiline(self):
        with self.assertRaises(ValueError):
            normalize_answer("42\nExplanation", "integer")
    def test_json_duplicate_key(self):
        with self.assertRaises(ValueError):
            parse_json_object('{"answer":"1","answer":"2"}')
    def test_json_fence(self):
        self.assertEqual(parse_json_object('```json\n{"answer":"7"}\n```')["answer"], "7")


class ToolTests(unittest.TestCase):
    def test_safe_arithmetic(self):
        self.assertEqual(safe_arithmetic("(72-12)/4"), "15")
    def test_reject_code_execution(self):
        with self.assertRaises(ToolError):
            safe_arithmetic("__import__('os').system('whoami')")
    def test_division_zero(self):
        with self.assertRaises(ToolError):
            safe_arithmetic("4/0")
    def test_linear_system(self):
        spec = {
            "variables": ["a", "b", "c"],
            "equations": [
                {"expr": "a+b+c=93", "source": "The three amounts total 93."},
                {"expr": "a=3*c", "source": "A is three times C."},
                {"expr": "b=c+8", "source": "B has 8 more than C."},
            ],
            "answer_spec": {"type": "assignment", "variables": ["a", "b", "c"],
                            "labels": ["A", "B", "C"]},
        }
        self.assertEqual(solve_linear_system(spec), "A=51, B=25, C=17")

    def test_linear_assignment_recovers_from_missing_labels(self):
        question = ("There are 93 objects in total. A is triple C. B is 8 more than C. "
                    "Answer exactly: A=<integer>, B=<integer>, C=<integer>")
        plan = {"method": "linear", "answer": "A=3, B=9, C=81", "linear": {
            "variables": ["a", "b", "c"],
            "equations": [
                {"expr": "a+b+c=93", "source": "There are 93 objects in total."},
                {"expr": "a=3*c", "source": "A is triple C."},
                {"expr": "b=c+8", "source": "B is 8 more than C."},
            ],
            "answer_spec": {"type": "assignment", "variables": ["A", "B", "C"]},
        }}
        llm = FakeLLM(plan)
        self.assertEqual(ReasoningAgent(llm).run(question), "A=51, B=25, C=17")
        self.assertEqual(len(llm.calls), 1)

    def test_linear_assignment_does_not_invent_ambiguous_mapping(self):
        from agent import _linear_spec_with_requested_format
        question = "Return X=<integer>, Y=<integer>"
        spec = {"variables": ["a", "b"], "answer_spec": {"type": "assignment", "labels": ["X", "Y"]}}
        self.assertEqual(_linear_spec_with_requested_format(question, spec), spec)

    def test_linear_rejects_nonlinear(self):
        spec = {"variables": ["x", "y"], "equations": [{"expr": "x*y=4", "source": "x times y is 4"}],
                "answer_spec": {"type": "variable", "name": "x"}}
        with self.assertRaises(ToolError):
            solve_linear_system(spec)
    def test_set_count(self):
        self.assertEqual(solve_set_count({"total":60,"count_a":32,"count_b":25,"both":12}), "15")
    def test_set_count_inconsistent(self):
        with self.assertRaises(ToolError):
            solve_set_count({"total":12,"count_a":10,"count_b":9,"both":1})
    def test_percent_chain(self):
        self.assertEqual(solve_percent_chain({"known_value":99,"solve_for":"initial",
            "operations":[{"type":"increase","percent":10},{"type":"decrease","percent":10}]}), "100")
    def test_schedule_parallel(self):
        tasks = [
            {"name":"design","duration":2,"depends_on":[]},
            {"name":"build","duration":6,"depends_on":["design"]},
            {"name":"legal","duration":3,"depends_on":[]},
            {"name":"deploy","duration":1,"depends_on":["build","legal"]},
        ]
        self.assertEqual(calculate_project_duration(tasks), "9")
    def test_schedule_cycle(self):
        with self.assertRaises(ToolError):
            calculate_project_duration([{"name":"a","duration":1,"depends_on":["b"]},
                                        {"name":"b","duration":1,"depends_on":["a"]}])
    def test_grid_bfs(self):
        grid = {"start":[0,0],"goal":[3,2],"min_x":0,"max_x":3,"min_y":0,"max_y":2,
                "blocked":[[1,0],[1,1]]}
        self.assertEqual(shortest_grid_path(grid), "5")
    def test_csp_digit_code(self):
        spec = {"variables":{"a":[2,3,4,5,6],"b":[2,3,4,5,6],"c":[2,3,4,5,6]},
                "constraints":[{"expr":"all_different(a,b,c)"},{"expr":"a>b"},
                                {"expr":"c==a-b"},{"expr":"a+b+c==12"}],
                "answer":{"type":"concat","variables":["a","b","c"]}}
        result = solve_csp(spec)
        self.assertIn(result["answer"], ("624", "642"))
        self.assertEqual(set(result["candidate_answers"]), {"624", "642"})


class GenericFamilyTests(unittest.TestCase):
    """Tests use paraphrases and changed values, not the instructor's public answers."""
    def test_generic_inventory_remainder(self):
        q = "A bakery packs 9 trays with 14 muffins each. It sells 37 muffins. How many muffins remain? Return integer."
        self.assertEqual(solve_question_family(q, "integer"), "89")

    def test_generic_set_count(self):
        q = "A class has 52 students. 31 study Biology. 24 study Chemistry. 9 study both. How many study neither? Return integer."
        self.assertEqual(solve_question_family(q, "integer"), "6")

    def test_generic_percentage_chain(self):
        q = "A price is increased by 10% and then discounted by 10%. The final price is 99. Find the original price. Return integer."
        self.assertEqual(solve_question_family(q, "integer"), "100")

    def test_generic_parallel_schedule(self):
        q = ("Design takes 2 days. Build takes 4 days and can start only after Design. "
             "Legal review takes 5 days and can start concurrently with Design. "
             "Deploy takes 1 day and starts only after Build and Legal review finish. "
             "What is the earliest project completion time? Return integer.")
        self.assertEqual(solve_question_family(q, "integer"), "7")

    def test_generic_grid_search(self):
        q = ("A robot starts at (0,0) and must reach (3,2). Moves are up, down, left, right. "
             "Blocked cells are (1,0) and (1,1). Bounds are 0<=x<=3 and 0<=y<=2. Return integer.")
        self.assertEqual(solve_question_family(q, "integer"), "5")

    def test_generic_digit_constraints(self):
        q = ("A three-digit code uses three distinct digits from {1,2,3,4,5}. "
             "The first digit is greater than the second. The third digit equals the difference "
             "between the first and second. The sum is 8. Digit 5 is not used. Return code.")
        self.assertEqual(solve_question_family(q, "code"), "413")

    def test_generic_box_propositions(self):
        q = ("Exactly one proposition is true. A: The key is in box 3. B: The key is not in box 3. "
             "C: The key is in box 1. The key is in one of boxes 1, 3, or 4. Which box? Return integer.")
        self.assertEqual(solve_question_family(q, "integer"), "3")

    def test_generic_truth_liar(self):
        q = ("In an island, each person always tells the truth or always lies. "
             "Ali says: Bita is a liar. Bita says: I and Ali are different types. "
             "Choose: Ali=Truth, Bita=Truth; Ali=Truth, Bita=Lie; "
             "Ali=Lie, Bita=Truth; Ali=Lie, Bita=Lie.")
        self.assertEqual(solve_question_family(q, "assignment"), "Ali=Lie, Bita=Truth")

    def test_generic_weekday_assignment(self):
        q = ("Four researchers Arman, Bahar, Kourosh and Darya each present Monday through Thursday. "
             "Arman is earlier than Kourosh. Bahar presents Thursday. Darya does not present Monday. "
             "Kourosh presents exactly one day after Darya. What day does Arman present? "
             "Monday Tuesday Wednesday Thursday.")
        self.assertEqual(solve_question_family(q, "day"), "Monday")

    def test_generic_order_constraints(self):
        q = ("Five tasks A, B, C, D and E run once in positions 1 to 5.\n"
             "- B must be before D.\n- A must be immediately after C.\n"
             "- E cannot be first or last.\n- D must be last. Which task is first?")
        self.assertEqual(solve_question_family(q, "letter"), "B")

    def test_generic_linear_assignment(self):
        q = ("Nima, Sara and Reza have 98 tokens in total. Nima has twice as many as Reza. "
             "Sara has 18 more than Reza. Return exactly Nima=<integer>, Sara=<integer>, Reza=<integer>.")
        self.assertEqual(solve_question_family(q, "assignment"), "Nima=40, Sara=38, Reza=20")

    def test_family_solver_precedes_model_for_supported_task(self):
        llm = FakeLLM()
        q = "A class has 52 students. 31 study Biology. 24 study Chemistry. 9 study both. How many study neither? Return integer."
        # This test asserts the default/auto fast path, regardless of the
        # caller's shell-level AGENT_STRATEGY used for live-model experiments.
        with patch.dict(os.environ, {"AGENT_STRATEGY": "auto"}):
            self.assertEqual(ReasoningAgent(llm).run(q), "6")
        self.assertEqual(len(llm.calls), 0)


class AgentTests(unittest.TestCase):
    def test_ollama_oom_skips_redundant_recovery_with_fallback(self):
        question = ("A class has 52 students. 31 study Biology. 24 study Chemistry. "
                    "9 study both. How many study neither? Return integer.")
        llm = FakeLLM(OllamaClientError(
            'Ollama HTTP 500: {"error":"model requires more system memory '
            '(3.8 GiB) than is available (2.5 GiB)"}'))
        with patch.dict(os.environ, {"AGENT_STRATEGY": "model-first"}):
            agent = ReasoningAgent(llm)
            self.assertEqual(agent.run(question), "6")
        self.assertEqual(len(llm.calls), 1)
        self.assertEqual(agent.model_calls, 1)

    def test_ollama_oom_skips_recovery_without_fallback(self):
        llm = FakeLLM(OllamaClientError(
            'Ollama HTTP 500: {"error":"model requires more system memory"}'))
        with patch.dict(os.environ, {"AGENT_STRATEGY": "model-first"}):
            agent = ReasoningAgent(llm)
            with self.assertRaisesRegex(AgentError, "insufficient memory"):
                agent.run("Solve a completely new unknown type of complex reasoning problem.")
        self.assertEqual(len(llm.calls), 1)

    def test_tool_verified_set_answer(self):
        plan = {"answer":"2","method":"sets","sets":{"total":60,"count_a":32,"count_b":25,"both":12,
                 "evidence":{"total":"60 students total","count_a":"32 study A","count_b":"25 study B","both":"12 study both"}}}
        llm = FakeLLM(plan)
        agent = ReasoningAgent(llm)
        q = "60 students total. 32 study A. 25 study B. 12 study both. How many study neither? Return only integer."
        self.assertEqual(agent.run(q), "15")
        self.assertEqual(agent.model_calls, 1)
    def test_tool_verified_linear_answer(self):
        plan = {"answer":"A=1, B=2, C=3","method":"linear","linear":{
            "variables":["a","b","c"],
            "equations":[{"expr":"a+b+c=93","source":"The three amounts total 93."},
                         {"expr":"a=3*c","source":"A is three times C."},
                         {"expr":"b=c+8","source":"B has 8 more than C."}],
            "answer_spec":{"type":"assignment","variables":["a","b","c"],"labels":["A","B","C"]}}}
        q = "The three amounts total 93. A is three times C. B has 8 more than C. A=<integer>, B=<integer>, C=<integer>"
        self.assertEqual(ReasoningAgent(FakeLLM(plan)).run(q), "A=51, B=25, C=17")
    def test_tool_verified_percentage(self):
        plan = {"answer":"99","method":"percent_chain","percent_chain":{
            "known_value":99,"solve_for":"initial","operations":[
                {"type":"increase","percent":10,"evidence":"increased by 10 percent"},
                {"type":"decrease","percent":10,"evidence":"discounted by 10 percent"}]}}
        q = "The price increased by 10 percent then discounted by 10 percent. Final price 99. Find initial price. Return only integer."
        self.assertEqual(ReasoningAgent(FakeLLM(plan)).run(q), "100")
    def test_grid_tool_overrides_bad_candidate(self):
        plan = {"answer":"4","method":"grid","grid":{"start":[0,0],"goal":[3,2],"min_x":0,"max_x":3,
                "min_y":0,"max_y":2,"blocked":[[1,0],[1,1]]}}
        q = "Move one cell up, down, left or right. Start (0,0), goal (3,2). Blocked (1,0), (1,1). Bounds 0<=x<=3 and 0<=y<=2. Return only integer."
        self.assertEqual(ReasoningAgent(FakeLLM(plan)).run(q), "5")
    def test_direct_text_answer_requires_check(self):
        llm = FakeLLM({"answer":"Paris","method":"direct","check":"The question asks for a capital city; Paris is the requested city."})
        self.assertEqual(ReasoningAgent(llm).run("What is the capital of France?"), "Paris")

    def test_numeric_direct_answer_has_bounded_best_effort_fallback(self):
        direct = {"answer":"42","method":"direct","check":"Independent check says the answer is 42."}
        llm = FakeLLM(direct, direct)
        agent = ReasoningAgent(llm)
        self.assertEqual(agent.run("What is the answer? Return only integer."), "42")
        self.assertEqual(agent.model_calls, 2)
    def test_invalid_primary_uses_structured_repair(self):
        repair = {"method":"calculate", "answer":"42", "check":"40 plus 2 equals 42",
                  "expression":"40+2", "evidence":["40 plus 2"]}
        llm = FakeLLM("not json", repair)
        self.assertEqual(ReasoningAgent(llm).run("40 plus 2. Return only integer."), "42")
        self.assertEqual(len(llm.calls), 2)
    def test_timeout_does_not_trigger_a_second_slow_request(self):
        llm = FakeLLM(OllamaClientError("timeout"), {"answer":"42"})
        with self.assertRaises(AgentError):
            ReasoningAgent(llm).run("Return only integer.")
        self.assertEqual(len(llm.calls), 1)
    def test_max_two_calls(self):
        repair = {"method":"calculate", "answer":"42", "check":"40 plus 2 equals 42",
                  "expression":"40+2", "evidence":["40 plus 2"]}
        llm = FakeLLM({"answer":"bad","method":"direct"}, repair)
        agent = ReasoningAgent(llm)
        self.assertEqual(agent.run("40 plus 2. Return only integer."), "42")
        self.assertLessEqual(agent.model_calls, 2)


class ClientAndCliTests(unittest.TestCase):
    def test_local_host_only(self):
        with patch.dict(os.environ, {"OLLAMA_HOST":"https://example.com"}):
            with self.assertRaises(ValueError):
                LocalOllama()
    def test_default_prefers_locally_installed_4b(self):
        class FakeResponse:
            def __enter__(self): return self
            def __exit__(self, *args): return False
            def read(self):
                return json.dumps({"models":[{"name":"qwen3:4b-instruct"}]}).encode()
        with patch.dict(os.environ, {}, clear=True), patch("agent.urlopen", return_value=FakeResponse()):
            self.assertEqual(LocalOllama().model_name, "qwen3:4b-instruct")
    def test_default_prefers_instructor_27b_if_present(self):
        class FakeResponse:
            def __enter__(self): return self
            def __exit__(self, *args): return False
            def read(self):
                return json.dumps({"models":[{"name":"qwen3:4b-instruct"}, {"name":"qwen3.8:27b"}]}).encode()
        with patch.dict(os.environ, {}, clear=True), patch("agent.urlopen", return_value=FakeResponse()):
            self.assertEqual(LocalOllama().model_name, "qwen3.8:27b")
    def test_model_can_be_overridden_for_local_development(self):
        with patch.dict(os.environ, {"OLLAMA_MODEL":"qwen3:1.7b"}, clear=True):
            self.assertEqual(LocalOllama().model_name, "qwen3:1.7b")
    def test_ollama_payload_is_local_json_and_thinking_off(self):
        class FakeResponse:
            def __enter__(self): return self
            def __exit__(self, *args): return False
            def read(self): return b'{"message":{"content":"{\\"answer\\":\\"8\\"}"}}'
        captured = []
        def fake_urlopen(req, timeout=None):
            captured.append((req.full_url, json.loads(req.data.decode("utf-8"))))
            return FakeResponse()
        with patch("agent.urlopen", side_effect=fake_urlopen):
            self.assertTrue(LocalOllama("qwen3:1.7b").chat("system", "user"))
        url, payload = captured[0]
        self.assertEqual(url, "http://127.0.0.1:11434/api/chat")
        self.assertIs(payload["think"], False)
        self.assertEqual(payload["format"], "json")
    def test_official_input_output_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            inp, out = Path(tmp)/"input.json", Path(tmp)/"output.json"
            inp.write_text(json.dumps({"id":"x1","problem":"Return integer."}), encoding="utf-8")
            with patch.object(main, "ReasoningAgent") as MockAgent:
                inst = MockAgent.return_value
                inst.run.return_value = "42"
                self.assertEqual(main.main(["--input",str(inp),"--output",str(out)]), 0)
            result = json.loads(out.read_text(encoding="utf-8"))
            self.assertEqual(result, {"id":"x1","answer":"42"})
    def test_invalid_input_still_writes_valid_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            inp, out = Path(tmp)/"input.json", Path(tmp)/"output.json"
            inp.write_text("{bad json", encoding="utf-8")
            out.write_text("stale output", encoding="utf-8")
            self.assertEqual(main.main(["--input", str(inp), "--output", str(out)]), 2)
            self.assertEqual(json.loads(out.read_text(encoding="utf-8")), {"id":"", "answer":""})

    def test_agent_error_still_writes_valid_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            inp, out = Path(tmp)/"input.json", Path(tmp)/"output.json"
            inp.write_text(json.dumps({"id":"x2","problem":"Return integer."}), encoding="utf-8")
            with patch.object(main, "ReasoningAgent") as MockAgent:
                MockAgent.return_value.run.side_effect = RuntimeError("simulated timeout")
                self.assertEqual(main.main(["--input",str(inp),"--output",str(out)]), 0)
            result = json.loads(out.read_text(encoding="utf-8"))
            self.assertEqual(result, {"id":"x2","answer":""})


class RoutingAndReliabilityTests(unittest.TestCase):
    def test_generic_route_hints(self):
        queries = {
            "grid": "A robot on a grid starts (0,0), goal (4,2). Blocked (2,1).",
            "schedule": "A project has tasks that depend on one another. Find the duration.",
            "sets": "There are 37 people; 19 study A, 15 study B and 8 study both. How many neither?",
            "percent_chain": "An item gets a 15% increase and then a 10% discount. Find its original price.",
            "linear": "Three amounts total 87. One is twice the other. Find the amounts.",
            "csp": "There are four statements. Exactly one is true. Which proposition holds?",
            "calculate": "Find the number of remaining packages after 7 are used from 25.",
        }
        for expected, question in queries.items():
            with self.subTest(expected=expected):
                self.assertEqual(select_tool_hint(question), expected)
                self.assertIn(expected, plan_prompt(question))

    def test_invalid_tool_plan_is_repaired_once(self):
        first = {"method": "calculate", "answer": "11", "check": "11 is the computed total",
                 "expression": "8+5", "evidence": ["10 marbles"]}
        repair = {"method": "calculate", "answer": "13", "check": "8 plus 5 equals 13",
                  "expression": "8+5", "evidence": ["8 marbles plus 5 marbles"]}
        model = FakeLLM(first, repair)
        result = ReasoningAgent(model).run("8 marbles plus 5 marbles. Return only integer.")
        self.assertEqual(result, "13")
        self.assertEqual(len(model.calls), 2)

    def test_strict_mode_rejects_unverified_invalid_tool_plan(self):
        first = {"method": "calculate", "answer": "13", "check": "8 plus 5 equals 13",
                 "expression": "8+5", "evidence": ["nonexistent text"]}
        model = FakeLLM(first, OllamaClientError("offline timeout"))
        with patch.dict(os.environ, {"AGENT_STRICT_VERIFICATION": "1"}):
            with self.assertRaises(AgentError):
                ReasoningAgent(model).run("8 marbles plus 5 marbles. Return only integer.")
        self.assertLessEqual(len(model.calls), 2)

    def test_accuracy_mode_preserves_format_valid_fallback(self):
        first = {"method": "calculate", "answer": "13", "check": "8 plus 5 equals 13",
                 "expression": "8+5", "evidence": ["nonexistent text"]}
        model = FakeLLM(first, OllamaClientError("offline timeout"))
        with patch.dict(os.environ, {"AGENT_STRICT_VERIFICATION": "0"}):
            self.assertEqual(ReasoningAgent(model).run("8 marbles plus 5 marbles. Return only integer."), "13")
        self.assertLessEqual(len(model.calls), 2)

    def test_multiple_label_format(self):
        self.assertEqual(normalize_answer("Alpha=10, Beta=6", "assignment"), "Alpha=10, Beta=6")




class PurchaseArithmeticRegressionTests(unittest.TestCase):
    """Fresh wordings/quantities: no ID-dependent sample lookups."""

    def test_official_sample_purchase_change(self):
        q = ("A shop buys 5 notebooks at 4 dollars each and 3 pens at 2 dollars each. "
             "It pays with 30 dollars. How much change should it receive? Return only the integer.")
        llm = FakeLLM()
        # Pin the strategy because this is a deterministic-family regression,
        # not a test of model-first routing.
        with patch.dict(os.environ, {"AGENT_STRATEGY": "auto"}):
            self.assertEqual(ReasoningAgent(llm).run(q), "4")
        self.assertEqual(len(llm.calls), 0)

    def test_unseen_items_and_numbers(self):
        q = ("A customer buys 4 apples at 3 dollars each and 7 oranges at 2 dollars each. "
             "He pays with 40 dollars. How much change should he receive?")
        self.assertEqual(solve_question_family(q, "integer"), "14")

    def test_unseen_packs_and_toys(self):
        q = ("A customer buys 2 packs at 5 dollars each and 3 toys at 4 dollars each. "
             "They pay with 35 dollars. How much change should they receive?")
        self.assertEqual(solve_question_family(q, "integer"), "13")

    def test_only_total_cost(self):
        q = "A customer buys 3 packs at 5 dollars each. How much is the total cost?"
        self.assertEqual(solve_question_family(q, "integer"), "15")

    def test_unknown_tax_abstains(self):
        q = ("A customer buys 2 packs at 5 dollars each with a 10% tax. "
             "Pays with 15 dollars. How much change?")
        from tools import _try_generic_purchase_arithmetic
        self.assertIsNone(_try_generic_purchase_arithmetic(q))

    def test_mixed_currency_abstains(self):
        q = ("A customer buys 2 apples at 5 dollars each and 3 oranges at 2 euros each. "
             "Pays with 50 dollars. How much change?")
        from tools import _try_generic_purchase_arithmetic
        self.assertIsNone(_try_generic_purchase_arithmetic(q))

    def test_extra_number_not_silently_ignored(self):
        q = ("A customer buys 3 apples at 4 dollars each. They pay with 30 dollars "
             "and also owe a separate 5 dollar delivery charge. How much change?")
        from tools import _try_generic_purchase_arithmetic
        self.assertIsNone(_try_generic_purchase_arithmetic(q))

    def test_bad_model_json_uses_single_repair(self):
        repair = {"method":"direct","answer":"73",
                  "check":"73 is obtained by subtracting the two stated totals."}
        agent = ReasoningAgent(FakeLLM("unparseable", repair))
        self.assertEqual(agent.run("What is the unknown number? Return only integer."), "73")
        self.assertEqual(agent.model_calls, 2)

    def test_timeout_still_uses_one_call(self):
        agent = ReasoningAgent(FakeLLM(OllamaClientError("timed out")))
        with self.assertRaises(AgentError):
            agent.run("What is the unknown number? Return only integer.")
        self.assertEqual(agent.model_calls, 1)



class UnseenInventoryRegressionTests(unittest.TestCase):
    """Held-out-style arithmetic variants, never match public IDs/answers."""
    def test_user_probe_general_event_ledger(self):
        q = ("There are 5 bags, each containing 8 apples. Ava gives away 13 apples "
             "and buys 7 more. How many apples does she have now? Return only the integer.")
        llm = FakeLLM()  # never contact a model for a fully specified arithmetic task
        with patch.dict(os.environ, {"AGENT_STRATEGY": "auto"}):
            self.assertEqual(ReasoningAgent(llm).run(q), "34")
        self.assertEqual(len(llm.calls), 0)

    def test_changed_values_and_items(self):
        q = ("There are 9 boxes, each containing 11 coins. Maya gives away 12 coins "
             "and receives 6 more. How many coins does she have now?")
        self.assertEqual(solve_question_family(q, "integer"), "93")

    def test_different_group_style(self):
        q = ("There are 4 baskets with 10 pears each. Sam sold 9 pears "
             "and bought 2 more. How many pears remain?")
        self.assertEqual(solve_question_family(q, "integer"), "33")

    def test_extra_group_purchase_is_not_misread_as_item(self):
        q = ("There are 5 boxes with 8 cakes each. Mara gives away 13 cakes "
             "and buys 7 boxes. How many cakes remain?")
        self.assertIsNone(solve_question_family(q, "integer"))

    def test_more_boxes_is_not_misread_as_more_cakes(self):
        q = ("There are 5 boxes with 8 cakes each. Mara gives away 13 cakes "
             "and buys 7 more boxes. How many cakes remain?")
        self.assertIsNone(solve_question_family(q, "integer"))

    def test_additional_numerical_condition_forces_abstention(self):
        q = ("There are 5 boxes with 8 cakes each. Mara gives away 13 cakes "
             "and buys 7 more. Each cake costs 2 dollars. How many cakes remain?")
        self.assertIsNone(solve_question_family(q, "integer"))

    def test_original_simple_inventory_remains_supported(self):
        q = ("A bakery packs 9 trays with 14 muffins each. It sells 37 muffins. "
             "How many muffins remain? Return integer.")
        self.assertEqual(solve_question_family(q, "integer"), "89")

    def test_unknown_problem_compact_plain_fallback(self):
        q = "How many different objects are left? Return only integer."
        model = FakeLLM("not JSON", "17")
        agent = ReasoningAgent(model)
        self.assertEqual(agent.run(q), "17")
        self.assertEqual(len(model.calls), 2)
        self.assertEqual(model.calls[1][2].get("response_format"), "text")

    def test_ollama_empty_content_compact_fallback(self):
        q = "How many objects does the collection contain? Return only integer."
        model = FakeLLM(OllamaClientError("Ollama returned empty final content"), "27")
        self.assertEqual(ReasoningAgent(model).run(q), "27")
        self.assertEqual(len(model.calls), 2)

    def test_model_only_timeout_does_not_make_many_calls(self):
        q = "How many unknown widgets? Return only integer."
        model = FakeLLM(OllamaClientError("timed out"))
        with self.assertRaises(AgentError):
            ReasoningAgent(model).run(q)
        self.assertEqual(len(model.calls), 1)

    def test_local_ollama_raw_fallback_omits_json_constraint(self):
        class Response:
            def __init__(self, reply): self.reply = reply
            def __enter__(self): return self
            def __exit__(self, *args): return False
            def read(self): return json.dumps({"message": {"content": self.reply}}).encode()
        calls = []
        def fake_urlopen(req, timeout=None):
            payload = json.loads(req.data.decode())
            calls.append(payload)
            return Response("13")
        with patch("agent.urlopen", side_effect=fake_urlopen):
            self.assertEqual(LocalOllama("qwen3:4b-instruct").chat(
                "Solve", "Problem", response_format="text"), "13")
        self.assertNotIn("format", calls[0])
        self.assertEqual(calls[0]["model"], "qwen3:4b-instruct")
        self.assertFalse(calls[0]["think"])

    def test_official_cli_for_unseen_problem_without_ollama(self):
        q = ("There are 5 bags, each containing 8 apples. Ava gives away 13 apples "
             "and buys 7 more. How many apples does she have now? Return only the integer.")
        with tempfile.TemporaryDirectory() as tmp:
            inp, out = Path(tmp)/"probe.json", Path(tmp)/"output.json"
            inp.write_text(json.dumps({"id":"new_01", "problem":q}), encoding="utf-8")
            with patch.dict(os.environ, {"OLLAMA_MODEL":"model-not-installed"}):
                self.assertEqual(main.main(["--input",str(inp),"--output",str(out)]), 0)
            self.assertEqual(json.loads(out.read_text(encoding="utf-8")),
                             {"id":"new_01","answer":"34"})

if __name__ == "__main__":
    unittest.main(verbosity=2)
