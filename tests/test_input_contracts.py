"""Cohort field normalization, observation status, and portfolio sample size."""
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from core import clinical, harmonize, rift


class InputContracts(unittest.TestCase):
    def test_charls_continuous_measurements_keep_their_own_missing_values(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            pd.DataFrame({
                "ID": [1, 2, 3], "wave": [1, 1, 1],
                "total_cognition": [np.nan, 0., 20.], "cog_status": [1., 1., 0.],
                "frailtya": [np.nan, 0., 35.], "frailtyb": [70., 80., 90.],
            }).to_csv(folder / "charls.csv", index=False)
            panel = harmonize.build("CHARLS", folder)
            self.assertTrue(np.isnan(panel.loc[0, "cognition"]))
            self.assertTrue(np.isnan(panel.loc[0, "frailty"]))
            np.testing.assert_allclose(panel.loc[1:, "cognition"], [0, 20])
            np.testing.assert_allclose(panel.loc[1:, "frailty"], [0, 35])

    def test_repeated_elsa_field_stems_use_the_same_first_column(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            source = pd.DataFrame({
                "idauniqc (identifier)": [10]*4,
                "idauniqc (additional label)": [99]*4,
                "wave": [1, 2, 3, 4], "iwindy": [2008, 2010, 2012, 2014],
                "iwindm": [6]*4, "iwstat": [0]*4, "agey": [68, 70, 72, 74],
                "hip (interval)": [0, 0, 1, 0],
                "hip (additional label)": [1]*4, "hipe": [0, 0, 1, 1],
                "dressa": [0, 0, 1, 1], "eata": [0, 0, 0, 0],
                "beda": [0, 0, 1, 0], "toilta": [0, 0, 0, 0],
            })
            source.to_csv(folder / "elsa.csv", index=False)
            panel = harmonize.finalise(harmonize.build("ELSA", folder))
            self.assertEqual(panel.id.unique().tolist(), ["10"])
            (folder / "harmonized").mkdir()
            panel.to_csv(folder / "harmonized/panel_elsa.csv", index=False)
            landmarks, _ = clinical.build("ELSA", folder, folder)
            self.assertEqual(landmarks.id.tolist(), ["10"])
            self.assertEqual(landmarks.a.tolist(), [1])
            self.assertTrue(landmarks.obs1.all())

    def test_death_interviews_and_year_precision_preserve_observation_status(self):
        frame = pd.DataFrame({
            "id": ["person", "person"], "wave": [1, 2],
            "iyear": [2010, 2012], "imonth": [6, np.nan],
            "age": [70, 72], "birthyear": [1940]*2,
            "iwstat": [0, 1], "hip_ever": [0, 1],
            "dyear": [np.nan, 2012], "dmonth": [np.nan, np.nan], "adl": [0, 3],
        })
        panel = harmonize.finalise(frame)
        self.assertEqual(panel.living_interview.tolist(), [True, False])
        self.assertEqual(panel.death_time_precision.tolist(), ["year", "year"])
        np.testing.assert_allclose(panel.death_t_lower, [2012, 2012])
        np.testing.assert_allclose(panel.death_t_upper, [2013, 2013])

    def test_fractional_and_invalid_case_counts_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            table = pd.DataFrame({
                "cohort": np.repeat(list(clinical.SEEDS), 16),
                "state": np.tile(np.arange(16), 3),
                "fracture": np.repeat(1/16, 48), "case_count": np.ones(48),
            })
            for invalid in [1.5, -1., np.nan, 0.]:
                candidate = table.copy()
                candidate["case_count"] = invalid
                candidate.to_csv(folder / "joint_states.csv", index=False)
                with self.assertRaisesRegex(ValueError, "case_count"):
                    rift._portfolios(folder, folder)
            table.to_csv(folder / "joint_states.csv", index=False)
            portfolios = rift._portfolios(folder, folder)
            self.assertEqual([total for _, total in portfolios.values()], [16]*3)

    def test_joint_state_required_fields_have_explicit_errors(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            pd.DataFrame({"cohort": ["HRS"]}).to_csv(folder / "joint_states.csv", index=False)
            with self.assertRaisesRegex(ValueError, "requires fields"):
                rift._portfolios(folder, folder)


if __name__ == "__main__":
    unittest.main()
