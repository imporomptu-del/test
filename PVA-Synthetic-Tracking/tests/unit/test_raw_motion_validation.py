from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('raw_motion_validation', ROOT / 'scripts/validate_raw16_motion_v3.py')
validation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(validation)


def control_report(seed):
    candidate = 'raw_robust_u16_v1'
    return dict(candidate=candidate, candidate_flow_backend='CUDA', fixture_seed=seed,
        motion_config_sha256=validation.validation.FROZEN_HASHES[validation.validation.MOTION],
        max_translation_error_px=.35, candidate_passed=True, case_count=12,
        cases=[dict(modes={candidate: dict(passed=True)}) for _ in range(12)],
        script_sha256=validation.sha(ROOT / 'scripts/check_raw16_motion_controls.py'),
        runtime_sha256={str(p.relative_to(ROOT)): validation.sha(p)
                        for p in (ROOT/'tiny_target/motion').glob('*.py')})


class RawMotionValidationTests(unittest.TestCase):
    def test_only_two_motion_options_changed_from_previous_frozen_config(self):
        validation.verify_candidate_config()
        old = json.loads((ROOT/'configs/evaluation/phase20_motion_v8.json').read_text())
        new = json.loads(validation.CANDIDATE.read_text())
        self.assertEqual(old['global_motion'], new['global_motion'])
        self.assertEqual(old['stabilization'], new['stabilization'])
        self.assertEqual(set(new['motion']) - set(old['motion']),
                         {'feature_intensity_mapping', 'optical_flow_backend'})

    def test_missing_failed_or_stale_controls_cannot_open_development_gate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaises(FileNotFoundError):
                validation.verify_generated_controls(root)
            for seed in validation.CONTROL_SEEDS:
                (root/f'final_controls_seed{seed}.json').write_text(json.dumps(control_report(seed)))
            self.assertEqual(len(validation.verify_generated_controls(root)), 3)
            seed = validation.CONTROL_SEEDS[0]
            good = control_report(seed)
            variants = []
            failed = deepcopy(good)
            failed['cases'][4]['modes']['raw_robust_u16_v1']['passed'] = False
            variants.append(failed)
            stale = deepcopy(good)
            stale['runtime_sha256']['tiny_target/motion/pva_pyrlk.py'] = 'stale'
            variants.append(stale)
            permissive = deepcopy(good)
            permissive['max_translation_error_px'] = 1.
            variants.append(permissive)
            for bad in variants:
                (root/f'final_controls_seed{seed}.json').write_text(json.dumps(bad))
                with self.assertRaises(ValueError):
                    validation.verify_generated_controls(root)


if __name__ == '__main__':
    unittest.main()
