#!/usr/bin/env python3

import json
import unittest
from unittest import mock

from evaluation import run_benchmark
from evaluation.run_benchmark import (
    JudgeError,
    dataset_sha256,
    evaluate_case,
    parse_decisions,
    score_case,
    stored_answers,
    summarize,
)


class ScoreCaseTest(unittest.TestCase):
    def test_points_rubric_normalizes_and_applies_penalty(self):
        case = {
            "grading_method": "points_rubric",
            "rubric_items": [
                {"item_id": "a", "points": 2},
                {"item_id": "b", "points": 1},
                {"item_id": "c", "points": -1},
            ],
        }
        score, components = score_case(case, {"a": True, "b": False, "c": True})
        self.assertAlmostEqual(score, 1 / 3)
        self.assertEqual(components["matched_signed_points"], 1)
        self.assertEqual(components["available_positive_points"], 3)

    def test_points_rubric_clamps_below_zero(self):
        case = {
            "grading_method": "points_rubric",
            "rubric_items": [
                {"item_id": "a", "points": 1},
                {"item_id": "b", "points": -2},
            ],
        }
        score, _ = score_case(case, {"a": False, "b": True})
        self.assertEqual(score, 0)

    def test_f1_weighted_rubric(self):
        case = {
            "grading_method": "f1_weighted_rubric",
            "rubric_items": [
                {"item_id": "a", "points": 3},
                {"item_id": "b", "points": 1},
                {"item_id": "c", "points": -2},
                {"item_id": "d", "points": 0},
            ],
        }
        score, components = score_case(
            case, {"a": True, "b": False, "c": True, "d": True}
        )
        # +3/+1/−2/0 → classes 9/7/2/5 → weights 72/3/24/1.
        # Recall 72/75; precision 72/97.
        recall = 72 / 75
        precision = 72 / 97
        expected = 2 * precision * recall / (precision + recall)
        self.assertAlmostEqual(score, round(expected, 4))
        self.assertEqual(components["severe_rate"], 0.0)
        self.assertAlmostEqual(components["recall_weighted"], round(recall, 4))
        self.assertAlmostEqual(components["precision_weighted"], round(precision, 4))

    def test_f1_weighted_rubric_severe_omission_is_reported_not_gated(self):
        case = {
            "grading_method": "f1_weighted_rubric",
            "rubric_items": [
                {"item_id": "a", "points": 3},
                {"item_id": "b", "points": 2},
            ],
        }
        score, components = score_case(case, {"a": False, "b": True})
        # +3/+2 → classes 9/8 → weights 72/24. Missing "a" is a severe
        # omission: it is reported, but the weighted F1 stands on its own.
        recall = 24 / 96
        precision = 1.0
        expected = 2 * precision * recall / (precision + recall)
        self.assertAlmostEqual(score, round(expected, 4))
        self.assertEqual(components["severe_rate"], 1.0)

    def test_f1_weighted_rubric_severe_commission_is_reported_not_gated(self):
        case = {
            "grading_method": "f1_weighted_rubric",
            "rubric_items": [
                {"item_id": "a", "points": 2},
                {"item_id": "b", "points": -3},
            ],
        }
        score, components = score_case(case, {"a": True, "b": True})
        # +2/−3 → classes 8/1 → weights 24/72. Asserting "b" is a severe
        # commission: it lands in the precision denominator, not a gate.
        recall = 1.0
        precision = 24 / 96
        expected = 2 * precision * recall / (precision + recall)
        self.assertAlmostEqual(score, round(expected, 4))
        self.assertEqual(components["severe_rate"], 1.0)

    def test_judge_decisions_must_cover_exact_item_ids(self):
        case = {
            "rubric_items": [
                {"item_id": "a"},
                {"item_id": "b"},
            ]
        }
        with self.assertRaises(ValueError):
            parse_decisions(
                case,
                '{"decisions":[{"item_id":"a","matched":true}]}',
            )

    def test_dataset_digest_uses_canonical_case_content(self):
        first = [{"case_id": "a", "prompt": "test", "rubric_items": []}]
        reordered = [{"rubric_items": [], "prompt": "test", "case_id": "a"}]
        self.assertEqual(dataset_sha256(first), dataset_sha256(reordered))

    def test_summary_includes_reproducibility_metadata(self):
        rows = [
            {
                "case_id": "a",
                "benchmark": "example",
                "score": 0.5,
                "answer_model": "answer",
                "judge_model": "judge",
                "answer_temperature": 0,
                "judge_temperature": None,
                "started_at": "2026-01-01T00:00:00+00:00",
                "finished_at": "2026-01-01T00:01:00+00:00",
            }
        ]
        summary = summarize(rows, 10, 42, {"a"}, "digest")
        self.assertEqual(summary["dataset_sha256"], "digest")
        self.assertEqual(summary["answer_temperature"], 0)
        self.assertEqual(summary["judge_passes_per_answer"], 1)
        self.assertEqual(summary["errors"], 0)

    def test_summary_counts_truncated_answers(self):
        rows = [
            {"case_id": "a", "benchmark": "example", "score": 1.0, "answer_stop_reason": "stop"},
            {"case_id": "b", "benchmark": "example", "score": 0.0, "answer_stop_reason": "length"},
        ]
        summary = summarize(rows, 10, 42, {"a", "b"}, "digest")
        self.assertEqual(summary["truncated_answers"], 1)


CASE = {
    "case_id": "c1",
    "benchmark": "example",
    "grading_method": "points_rubric",
    "prompt": "question",
    "rubric_items": [{"item_id": "a", "text": "criterion", "points": 1}],
}
VALID_JUDGE = json.dumps({"decisions": [{"item_id": "a", "matched": True}]})
TRUNCATED_JUDGE = '{"decisions":[{"item_id":"a","matched":tr'


def fake_post_chat(judge_replies):
    """Stand-in for post_chat: answers in plain mode, pops judge replies in JSON mode."""
    calls = {"answer": 0, "judge": 0}

    def post_chat(token, model, messages, **kwargs):
        if kwargs["json_mode"]:
            calls["judge"] += 1
            return judge_replies.pop(0), 0, "stop"
        calls["answer"] += 1
        return f"answer {calls['answer']}", 1, "stop"

    return post_chat, calls


class JudgeFailureTest(unittest.TestCase):
    def test_invalid_judge_reply_is_retried(self):
        post_chat, calls = fake_post_chat([TRUNCATED_JUDGE, VALID_JUDGE])
        with mock.patch.object(run_benchmark, "post_chat", post_chat):
            row = evaluate_case("token", CASE, "answer-model", "judge-model", 1)
        self.assertEqual(row["score"], 1.0)
        self.assertEqual(calls, {"answer": 1, "judge": 2})

    def test_judge_failure_keeps_the_answer(self):
        replies = [TRUNCATED_JUDGE] * run_benchmark.JUDGE_ATTEMPTS
        post_chat, _ = fake_post_chat(replies)
        with mock.patch.object(run_benchmark, "post_chat", post_chat):
            with self.assertRaises(JudgeError) as raised:
                evaluate_case("token", CASE, "answer-model", "judge-model", 1)
        self.assertEqual(raised.exception.answer["answer"], "answer 1")
        self.assertEqual(raised.exception.judge_content, TRUNCATED_JUDGE)

    def test_stored_answer_is_rejudged_not_resampled(self):
        stored = {"answer": "kept answer", "answer_temperature": 1, "answer_stop_reason": "stop"}
        post_chat, calls = fake_post_chat([VALID_JUDGE])
        with mock.patch.object(run_benchmark, "post_chat", post_chat):
            row = evaluate_case(
                "token", CASE, "answer-model", "judge-model", 1, stored_answer=stored
            )
        self.assertEqual(row["answer"], "kept answer")
        self.assertEqual(calls, {"answer": 0, "judge": 1})

    def test_stored_answers_come_from_failed_rows_with_an_answer(self):
        rows = [
            {"case_id": "a", "error": "JudgeError: ...", "answer": "kept", "answer_temperature": 1},
            {"case_id": "b", "error": "RuntimeError: answer request failed"},
            {"case_id": "c", "score": 1.0, "answer": "done"},
        ]
        stored = stored_answers(rows)
        self.assertEqual(set(stored), {"a"})
        self.assertEqual(stored["a"]["answer"], "kept")


if __name__ == "__main__":
    unittest.main()
