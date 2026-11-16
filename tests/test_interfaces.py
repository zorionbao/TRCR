"""Execution interfaces and module initialization contracts."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


PACKAGE = Path(__file__).resolve().parents[1]


class ExecutionInterfaces(unittest.TestCase):
    def test_eight_reusable_panel_modules_per_group(self):
        for figure in range(1, 5):
            folder = PACKAGE / "figures" / f"figure{figure}"
            self.assertEqual(
                sorted(p.name for p in folder.glob("panel_*.py")),
                [f"panel_{letter}.py" for letter in "abcdefgh"],
            )

    def test_imports_leave_execution_directory_unchanged(self):
        with tempfile.TemporaryDirectory() as directory:
            env = os.environ.copy()
            env["PYTHONPATH"] = str(PACKAGE)
            env["PYTHONDONTWRITEBYTECODE"] = "1"
            code = (
                "import importlib, pkgutil; import core, figures; "
                "[importlib.import_module(m.name) for p in (core, figures) "
                "for m in pkgutil.walk_packages(p.__path__, p.__name__ + '.')]; "
                "from pathlib import Path; assert list(Path('.').iterdir()) == []"
            )
            result = subprocess.run(
                [sys.executable, "-B", "-c", code], cwd=directory, env=env,
                capture_output=True, text=True, encoding="utf-8",
            )
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_missing_source_does_not_create_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run(
                [sys.executable, "-B", str(PACKAGE / "run.py"), "analyze",
                 "--source-dir", "absent", "--output-dir", "products"],
                cwd=directory, capture_output=True, text=True, encoding="utf-8",
            )
            self.assertEqual(result.returncode, 2)
            self.assertIn("Required cohort input", result.stderr)
            self.assertFalse((Path(directory) / "products").exists())


if __name__ == "__main__":
    unittest.main()
