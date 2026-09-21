"""Test the 18-job command matrix with fake sbatch/python executables."""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest


class SubmissionTests(unittest.TestCase):
    def test_llama_eosfix_helper_submits_isolated_18_job_grid(self):
        repo = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "data/t_ws").mkdir(parents=True)
            (root / "data/watermark_config.json").write_text("{}")
            (root / "data/t_ws/combined_t_ws.json").write_text("[]")
            submit_script = str(repo / "scripts/submit_llama_eosfix_experiments.sh")
            bin_dir = root / "bin"
            bin_dir.mkdir()
            python = bin_dir / "python"
            python.write_text('#!/bin/bash\nexit 0\n')
            python.chmod(0o755)
            sbatch = bin_dir / "sbatch"
            sbatch.write_text('#!/bin/bash\nprintf "%s\\n" "$*" >> "$SUBMISSION_TEST_LOG"\n')
            sbatch.chmod(0o755)
            log = root / "jobs.txt"
            env = {**os.environ, "LLAMA_VENV_DIR": str(root),
                   "SUBMISSION_TEST_LOG": str(log),
                   "PATH": str(bin_dir) + os.pathsep + os.environ["PATH"]}
            subprocess.run(["bash", submit_script, "capella"], cwd=root, env=env,
                           check=True, capture_output=True, text=True)
            jobs = log.read_text().splitlines()
            self.assertEqual(len(jobs), 18)
            for sample_type in ("abstracts_only", "abstracts_and_titles", "questions"):
                for n in (100, 500, 1000, 5000, 10000, 50000):
                    expected = ("full_pipeline_llama_eosfix "
                                f"{n} {sample_type} 32 1000")
                    self.assertEqual(sum(expected in line for line in jobs), 1)
            self.assertTrue(all("launch_capella_llama.sh" in line for line in jobs))

    def test_qwen_on_llama_additional_eval_helper_submits_four_jobs(self):
        repo = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "data/t_ws").mkdir(parents=True)
            (root / "data/experiment_config_qwen.json").write_text("{}")
            (root / "data/t_ws/combined_t_ws.json").write_text("[]")
            submit_script = str(repo / "scripts/submit_qwen_on_llama_additional_evals.sh")
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
            subprocess.run(["bash", submit_script, "capella"], cwd=root, env=env,
                           check=True, capture_output=True, text=True)
            jobs = log.read_text().splitlines()
            self.assertEqual(len(jobs), 4)
            self.assertEqual(sum("qwen_on_llama_open_pipeline" in line for line in jobs), 1)
            self.assertIn("qwen_on_llama_open_pipeline 1000 abstracts_only 32 1000 --resume", jobs[0])
            self.assertEqual(sum("similarity_eval --profile qwen_on_llama" in line for line in jobs), 3)
            self.assertIn("--resume", jobs[0])
            self.assertTrue(all("--skip-existing" in line for line in jobs[1:]))

    def test_submits_exactly_one_job_per_experiment_and_size(self):
        repo = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "data").mkdir()
            (root / "data/experiment_config_qwen.json").write_text("{}")
            submit_script = str(repo / "scripts/submit_qwen_experiments.sh")
            bin_dir = root / "bin"
            bin_dir.mkdir()
            python = bin_dir / "python"
            python.write_text('#!/bin/bash\nexit 0\n')
            python.chmod(0o755)
            sbatch = bin_dir / "sbatch"
            sbatch.write_text('#!/bin/bash\n[[ -d job_outputs ]] || exit 9\nprintf "%s\\n" "$*" >> "$SUBMISSION_TEST_LOG"\n')
            sbatch.chmod(0o755)
            log = root / "jobs.txt"
            env = {**os.environ, "QWEN_VENV_DIR": str(root), "SUBMISSION_TEST_LOG": str(log),
                   "PATH": str(bin_dir) + os.pathsep + os.environ["PATH"]}
            subprocess.run(["bash", submit_script, "capella", "--resume"],
                           cwd=root, env=env, check=True, capture_output=True, text=True)
            self.assertTrue((root / "job_outputs").is_dir())
            jobs = log.read_text().splitlines()
            self.assertEqual(len(jobs), 18)
            for experiment in ("abstracts_only", "abstracts_and_titles", "questions"):
                for n in (100, 500, 1000, 5000, 10000, 50000):
                    self.assertEqual(sum(f"full_pipeline {n} {experiment} 32 1000 --profile qwen --resume" in line
                                         for line in jobs), 1)
            result = subprocess.run(["bash", submit_script, "capella", "--smoke"],
                                    cwd=root, env=env, capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(len(log.read_text().splitlines()), 18)

    def test_batch64_helper_submits_isolated_matrix_with_fixed_microbatches(self):
        repo = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "data/t_ws").mkdir(parents=True)
            (root / "data/experiment_config_qwen.json").write_text("{}")
            (root / "data/t_ws/combined_t_ws.json").write_text("[]")
            submit_script = str(repo / "scripts/submit_qwen_on_llama_batch64_experiments.sh")
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
            subprocess.run(["bash", submit_script, "capella", "--resume"],
                           cwd=root, env=env, check=True, capture_output=True, text=True)
            jobs = log.read_text().splitlines()
            self.assertEqual(len(jobs), 18)
            for experiment in ("abstracts_only", "abstracts_and_titles", "questions"):
                micro = 32 if experiment == "abstracts_only" else 16
                for n in (100, 500, 1000, 5000, 10000, 50000):
                    expected = (f"qwen_on_llama_batch64_pipeline {n} {experiment} 64 1000 "
                                f"--micro-batch-size {micro} --resume")
                    self.assertEqual(sum(expected in line for line in jobs), 1)

    def test_all_launchers_use_job_outputs(self):
        repo = Path(__file__).resolve().parents[1]
        self.assertTrue((repo / "job_outputs/.gitkeep").is_file())
        for script in (repo / "scripts").glob("launch_*.sh"):
            for option, extension in (("output", "out"), ("error", "err")):
                directives = [line for line in script.read_text().splitlines()
                              if line.startswith(f"#SBATCH --{option}=")]
                self.assertEqual(len(directives), 1, script.name)
                self.assertTrue(directives[0].endswith(f"job_outputs/slurm-%x-%j.{extension}"), script.name)


if __name__ == "__main__":
    unittest.main()
