"""Test the 18-job command matrix with fake sbatch/python executables."""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest


class SubmissionTests(unittest.TestCase):
    def test_submits_exactly_one_job_per_experiment_and_size(self):
        repo = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bin_dir = root / "bin"
            bin_dir.mkdir()
            python = bin_dir / "python"
            python.write_text('#!/bin/bash\nexit 0\n')
            python.chmod(0o755)
            sbatch = bin_dir / "sbatch"
            sbatch.write_text('#!/bin/bash\nprintf "%s\\n" "$*" >> "$SUBMISSION_TEST_LOG"\n')
            sbatch.chmod(0o755)
            log = root / "jobs.txt"
            env = {**os.environ, "QWEN_VENV_DIR": str(root), "SUBMISSION_TEST_LOG": str(log),
                   "PATH": str(bin_dir) + os.pathsep + os.environ["PATH"]}
            subprocess.run(["bash", "scripts/submit_qwen_experiments.sh", "capella", "--resume"],
                           cwd=repo, env=env, check=True, capture_output=True, text=True)
            jobs = log.read_text().splitlines()
            self.assertEqual(len(jobs), 18)
            for experiment in ("abstracts_only", "abstracts_and_titles", "questions"):
                for n in (100, 500, 1000, 5000, 10000, 50000):
                    self.assertEqual(sum(f"full_pipeline {n} {experiment} 32 1000 --profile qwen --resume" in line
                                         for line in jobs), 1)
            result = subprocess.run(["bash", "scripts/submit_qwen_experiments.sh", "capella", "--smoke"],
                                    cwd=repo, env=env, capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(len(log.read_text().splitlines()), 18)


if __name__ == "__main__":
    unittest.main()
