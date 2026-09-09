import os
import unittest
from unittest.mock import patch

from main import run_serve


class TestMainEntrypoint(unittest.TestCase):
    def test_serve_generation_mode_reaches_runtime_environment(self):
        with patch.dict(os.environ, {"SUPPORT_GENERATION_MODE": "offline"}):
            with patch("uvicorn.run") as uvicorn_run:
                run_serve(generation_mode="local")
                uvicorn_run.assert_called_once()
                self.assertEqual(os.environ["SUPPORT_GENERATION_MODE"], "local")


if __name__ == "__main__":
    unittest.main()
