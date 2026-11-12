"""Regression tests with generated inputs and additional edge cases.

The tests use new values and wording variations. They are not a substitute
for evaluation on the instructor's hidden question set.
"""
import json
import os
import random
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agent import ReasoningAgent, OllamaClientError
from tools import solve_question_family, validate_bullet_coverage, ToolError


class FakeLLM:
    def __init__(self, *replies, model_name="qwen3.5:27b"):
        self.model_name = model_name
        self.replies = list(replies)
        self.calls = 0

    def chat(self, *args, **kwargs):
        self.calls += 1
        response = self.replies.pop(0)
        if isinstance(response, BaseException):
            raise response
        return json.dumps(response) if isinstance(response, dict) else response


class MetamorphicArithmeticTests(unittest.TestCase):
    def test_two_set_inclusion_exclusion_randomized(self):
        r = random.Random(345)
        for _ in range(80):
            both = r.randrange(1, 20)
            only_a = r.randrange(1, 30)
            only_b = r.randrange(1, 30)
            neither = r.randrange(1, 30)
            n = both + only_a + only_b + neither
            q = (f"A class has {n} students. {only_a + both} study Physics. "
                 f"{only_b + both} study Literature. {both} study both. "
                 "How many study neither? Return integer.")
            with self.subTest(n=n, both=both):
                self.assertEqual(solve_question_family(q, "integer"), str(neither))

    def test_percent_forward_and_reverse_randomized(self):
        r = random.Random(70)
        for _ in range(70):
            original = r.randint(1, 999) * 8
            final = original * 9 // 8  # +25%, then -10% = 9/8.
            forward = (f"A price increased by 25% and then discounted by 10%. "
                       f"The initial price was {original}. What is the final price?")
            reverse = (f"A price increased by 25% and then discounted by 10%. "
                       f"The final price is {final}. What was the original price?")
            with self.subTest(original=original):
                self.assertEqual(solve_question_family(forward), str(final))
                self.assertEqual(solve_question_family(reverse), str(original))

    def test_purchase_change_randomized(self):
        r = random.Random(11)
        for _ in range(70):
            count_a, unit_a = r.randint(1, 20), r.randint(1, 20)
            count_b, unit_b = r.randint(1, 20), r.randint(1, 20)
            change = r.randint(1, 100)
            paid = count_a * unit_a + count_b * unit_b + change
            q = (f"A customer buys {count_a} pencils at {unit_a} dollars each and "
                 f"{count_b} rulers at {unit_b} dollars each. "
                 f"They pay with {paid} dollars. How much change should they receive?")
            with self.subTest(change=change):
                self.assertEqual(solve_question_family(q), str(change))

    def test_multi_event_inventory_randomized(self):
        r = random.Random(8)
        for _ in range(60):
            bags, each = r.randint(3, 20), r.randint(5, 30)
            given = r.randint(1, bags * each - 1)
            added = r.randint(1, 25)
            expected = bags * each - given + added
            q = (f"There are {bags} bags, each containing {each} apples. "
                 f"Ava gives away {given} apples and buys {added} more. "
                 "How many apples does she have now? Return only the integer.")
            with self.subTest(expected=expected):
                self.assertEqual(solve_question_family(q), str(expected))


class AdversarialCoverageTests(unittest.TestCase):
    def test_schedule_does_not_answer_wrong_target(self):
        q = ("Design takes 2 days. Build takes 4 days and can start only after Design. "
             "Legal review takes 5 days and can start concurrently with Design. "
             "Deploy takes 1 day and starts only after Build and Legal review finish. "
             "When can Build start? Return the integer.")
        self.assertIsNone(solve_question_family(q))

    def test_ordering_does_not_answer_count_question(self):
        q = ("Five tasks A, B, C, D and E run once in positions 1 to 5. "
             "B must be before D. A must be immediately after C. "
             "E cannot be first or last. D must be last. "
             "How many valid orders are there?")
        self.assertIsNone(solve_question_family(q))

    def test_unknown_ordering_constraint_abstains(self):
        q = ("Five tasks A, B, C, D and E run once in positions 1 to 5. "
             "B must be before D. A must be immediately after C. "
             "E cannot be first or last. D must be last. "
             "C cannot be second. Which task is first?")
        self.assertIsNone(solve_question_family(q))

    def test_grid_with_teleport_abstains(self):
        q = ("A robot on a grid starts (0,0) and must reach (3,3). "
             "Blocked (1,0), (1,1). Bounds 0<=x<=3, 0<=y<=3. "
             "Moves are up, down, left, or right, but a teleport is also allowed. "
             "Find the shortest path length.")
        self.assertIsNone(solve_question_family(q))

    def test_extra_set_count_abstains(self):
        q = ("A class has 50 students. 25 study A. 20 study B. "
             "10 study both. Also, 5 students study C. "
             "How many study neither A nor B?")
        self.assertIsNone(solve_question_family(q))

    def test_bullet_quote_must_not_be_one_character(self):
        q = "Constraints:\n- B must be before D.\n- A must be after C."
        with self.assertRaises(ToolError):
            validate_bullet_coverage(q, ["B", "A"])


class AdaptiveAgentTests(unittest.TestCase):
    def test_large_model_semantic_route_uses_model(self):
        q = ("Design takes 2 days. Build takes 4 days and can start only after Design. "
             "Legal review takes 5 days and can start concurrently with Design. "
             "Deploy takes 1 day and starts only after Build and Legal review finish. "
             "What is the earliest project completion time? Return integer.")
        llm = FakeLLM({"method": "direct", "answer": "7", "check": "Build finishes at 6; deploy at 7."},
                      {"status": "correct", "answer": "7", "check": "7 is the earliest completion."})
        with patch.dict(os.environ, {"AGENT_STRATEGY": "auto"}):
            self.assertEqual(ReasoningAgent(llm).run(q), "7")
        self.assertEqual(llm.calls, 2)

    def test_large_model_timeout_returns_computed_baseline(self):
        q = ("Design takes 2 days. Build takes 4 days and can start only after Design. "
             "Legal review takes 5 days and can start concurrently with Design. "
             "Deploy takes 1 day and starts only after Build and Legal review finish. "
             "What is the earliest project completion time? Return integer.")
        llm = FakeLLM(OllamaClientError("timed out"))
        with patch.dict(os.environ, {"AGENT_STRATEGY": "auto"}):
            self.assertEqual(ReasoningAgent(llm).run(q), "7")
        self.assertEqual(llm.calls, 1)

    def test_verifier_can_correct_direct_candidate(self):
        first = {"method": "direct", "answer": "17", "check": "My computed value is 17."}
        review = {"status": "incorrect", "answer": "18", "check": "The sum of 9 and 9 is 18."}
        llm = FakeLLM(first, review)
        q = "What is the sum of 9 and 9? Return only the integer."
        with patch.dict(os.environ, {"AGENT_STRATEGY": "model-first"}):
            self.assertEqual(ReasoningAgent(llm).run(q), "18")
        self.assertEqual(llm.calls, 2)

    def test_fast_mode_avoids_llm_for_recognized_family(self):
        llm = FakeLLM(model_name="qwen3.5:27b")
        q = ("Design takes 2 days. Build takes 4 days and can start only after Design. "
             "Legal review takes 5 days and can start concurrently with Design. "
             "Deploy takes 1 day and starts only after Build and Legal review finish. "
             "What is the earliest project completion time? Return integer.")
        with patch.dict(os.environ, {"AGENT_STRATEGY": "fast"}):
            self.assertEqual(ReasoningAgent(llm).run(q), "7")
        self.assertEqual(llm.calls, 0)


if __name__ == '__main__':
    unittest.main()
