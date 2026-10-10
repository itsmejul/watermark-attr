"""Exercise Slurm dependency/environment wiring without contacting Slurm."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


class LlamaSubmissionTests(unittest.TestCase):
    def test_four_generation_jobs_then_two_dependent_training_jobs(self):
        script = Path(__file__).resolve().parents[1] / 'scripts/submit/submit_llama_prefix_augmentation.sh'
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ('data/t_ws/config_llama.json', 'scripts/launch_capella.sh'):
                p = root / name
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text('')
            # Stub environment checks and input preflight; those are tested separately.
            for name in ('.venv-watermark', '.venv-experiment'):
                p = root / name / 'bin/python'
                p.parent.mkdir(parents=True)
                p.write_text('#!/bin/sh\nexit 0\n')
                p.chmod(0o755)
            bindir = root / 'bin'
            bindir.mkdir()
            sbatch = bindir / 'sbatch'
            sbatch.write_text(f'#!{sys.executable}\n' + '''import json, os, sys
from pathlib import Path
p = Path(os.environ['SUBMIT_LOG'])
rows = json.loads(p.read_text()) if p.exists() else []
rows.append(dict(args=sys.argv[1:], venv=os.environ['RUN_VENV_DIR']))
p.write_text(json.dumps(rows))
print(7000 + len(rows))
''')
            sbatch.chmod(0o755)
            env = dict(os.environ, PATH=f'{bindir}:{os.environ["PATH"]}', SUBMIT_LOG=str(root/'jobs.json'))
            for key in ('WATERMARK_VENV_DIR', 'EXPERIMENT_VENV_DIR'):
                env.pop(key, None)
            subprocess.run(['bash', str(script), 'capella'], cwd=root, env=env,
                           check=True, capture_output=True, text=True)
            jobs = json.loads((root/'jobs.json').read_text())
            self.assertEqual(len(jobs), 6)
            for i, job in enumerate(jobs[:4], 2):
                self.assertEqual(job['venv'], str(root/'.venv-watermark'))
                self.assertIn('src.data_creation.create_llama_prefix_variants', job['args'])
                self.assertEqual(job['args'][-2:], ['--version', str(i)])
            for job in jobs[4:]:
                self.assertEqual(job['venv'], str(root/'.venv-experiment'))
                self.assertIn('src.experiments.main.llama_prefix_augmentation', job['args'])
                self.assertIn('--kill-on-invalid-dep=yes', job['args'])
                self.assertIn('--resume', job['args'])
            self.assertIn('--dependency=afterok:7001:7002', jobs[4]['args'])
            self.assertIn('--dependency=afterok:7001:7002:7003:7004', jobs[5]['args'])


if __name__ == '__main__':
    unittest.main()
