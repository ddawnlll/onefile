"""Include the native execution self-test in the suite without another interpreter."""
import contextlib
import io
import unittest

import one


class ExecutionSelfTests(unittest.TestCase):
    def test_native_execution_contract(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            one.self_test()
        self.assertIn("SELF-TEST PASSED: 38 checks", output.getvalue())
