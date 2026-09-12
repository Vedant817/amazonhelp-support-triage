import os
import unittest
from unittest.mock import patch

from main import run_benchmark, run_serve


class TestMainEntrypoint(unittest.TestCase):
    def test_benchmark_forwards_contrastive_ablation_size(self):
        with patch("subprocess.call", return_value=0) as call:
            run_benchmark(contrastive_pairs=600)
        command = call.call_args.args[0]
        self.assertIn("--contrastive-pairs", command)
        self.assertIn("600", command)

    def test_serve_generation_mode_reaches_runtime_environment(self):
        with patch.dict(os.environ, {"SUPPORT_GENERATION_MODE": "offline"}):
            with patch("uvicorn.run") as uvicorn_run:
                run_serve(generation_mode="local")
                uvicorn_run.assert_called_once()
                self.assertEqual(os.environ["SUPPORT_GENERATION_MODE"], "local")


if __name__ == "__main__":
    unittest.main()
