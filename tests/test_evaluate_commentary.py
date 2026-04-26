import tempfile
import unittest
from pathlib import Path

from evaluate_commentary import CommentaryEvaluator


FIXTURES = Path(__file__).resolve().parent / "fixtures"


class CommentaryEvaluationScriptTests(unittest.TestCase):
    def test_evaluator_writes_expected_research_outputs(self) -> None:
        gm_path = FIXTURES / "grandmaster_eval_fixture.pgn"
        cdc_path = FIXTURES / "chessdotcom_eval_fixture.pgn"

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir) / "evaluation"
            evaluator = CommentaryEvaluator(semantic_backend="tfidf")
            summary = evaluator.evaluate(
                grandmaster_path=gm_path,
                chessdotcom_path=cdc_path,
                output_dir=output_dir,
                max_games=1,
            )

            self.assertEqual(summary["reference"]["games"], 1)
            self.assertEqual(summary["baseline"]["games"], 1)
            self.assertEqual(summary["baseline"]["shared_moves_with_reference"], 8)
            self.assertTrue(summary["fair_baseline_available"])
            self.assertEqual(len(summary["benchmark"]), 2)
            self.assertTrue(summary["system_by_phase"])
            self.assertTrue(summary["system_by_quality"])
            self.assertTrue(summary["source_profiles"])

            expected_files = [
                output_dir / "README.md",
                output_dir / "report.md",
                output_dir / "benchmark_summary.csv",
                output_dir / "paper_tables.md",
                output_dir / "paper_tables.tex",
                output_dir / "figures" / "main_benchmark.svg",
                output_dir / "figures" / "system_phase_breakdown.svg",
                output_dir / "figures" / "system_quality_breakdown.svg",
                output_dir / "figures" / "metric_heatmap.svg",
            ]
            for path in expected_files:
                self.assertTrue(path.exists(), path)


if __name__ == "__main__":
    unittest.main()
