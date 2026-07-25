from __future__ import annotations

import unittest

import numpy as np

from mpc.preview_nominal import nominal_command, nominal_index, nominal_window


class PreviewNominalTests(unittest.TestCase):
    def setUp(self) -> None:
        self.reference = np.arange(20, dtype=np.float32)[:, None]

    def test_zero_preview_preserves_direct_nominal_index(self) -> None:
        self.assertEqual(nominal_index(4, 0), 5)
        np.testing.assert_array_equal(nominal_command(self.reference, 4, 0), self.reference[5])
        np.testing.assert_array_equal(nominal_window(self.reference, 4, 3, 0), self.reference[5:8])

    def test_preview_advances_only_nominal_window(self) -> None:
        self.assertEqual(nominal_index(4, 7), 12)
        np.testing.assert_array_equal(nominal_command(self.reference, 4, 7), self.reference[12])
        np.testing.assert_array_equal(nominal_window(self.reference, 4, 3, 7), self.reference[12:15])
        # The caller retains reference[step + 1] as its tracking target.
        self.assertEqual(float(self.reference[5, 0]), 5.0)

    def test_invalid_or_unpadded_reference_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "non-negative"):
            nominal_index(0, -1)
        with self.assertRaisesRegex(IndexError, "outside reference"):
            nominal_command(self.reference, 19, 1)
        with self.assertRaisesRegex(IndexError, "exceeds reference"):
            nominal_window(self.reference, 15, 2, 3)


if __name__ == "__main__":
    unittest.main()
