import unittest
from pathlib import Path


FEM_ROOT = Path(__file__).resolve().parents[1]
SOLVE_SCRIPT = FEM_ROOT / "temp" / "solve_inp_sequential.ps1"


class SolveInpSequentialScriptTests(unittest.TestCase):
    def read_solve_script(self):
        return SOLVE_SCRIPT.read_text(encoding="utf-8")

    def test_default_cpu_count_uses_single_standard_license(self):
        text = self.read_solve_script()

        self.assertIn("[int]$Cpus = 1", text)
        self.assertNotIn("[int]$Cpus = 2", text)

    def test_explicit_cpu_override_is_still_forwarded_to_abaqus(self):
        text = self.read_solve_script()

        self.assertIn('"cpus=$Cpus"', text)
        self.assertIn('& abaqus "job=$jobName" "input=$($inp.FullName)" "cpus=$Cpus" interactive', text)


if __name__ == "__main__":
    unittest.main()
