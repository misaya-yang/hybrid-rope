import json
from pathlib import Path
import subprocess
import sys
import unittest

from experiments.native_enhancement_oral_20260915.run import build_command


class PlanTests(unittest.TestCase):
    def test_default_is_read_only_without_server_assets(self):
        result = subprocess.run([sys.executable, "-m", "experiments.native_enhancement_oral_20260915.run"], capture_output=True, text=True, check=True)
        value = json.loads(result.stdout)
        self.assertEqual(value["status"], "PLAN_ONLY")
        self.assertFalse(value["gpu_execution"])
        for command in value["commands_without_execute"].values():
            self.assertNotIn("--execute", command)

    def test_static_and_native_interfaces(self):
        for table in (None, Path("ncp.json")):
            command = build_command(python="python", model=Path("m"), panel=Path("p"), data=Path("d"), out=Path("o"), arm="ncp", table=table)
            self.assertIn("--skip-lm", command)
            self.assertIn("--only-extra-panels", command)
            self.assertEqual("--static-table-json" in command, table is not None)
            self.assertNotIn("--execute", command)


if __name__ == "__main__":
    unittest.main()
