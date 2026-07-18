import os

import pytest

from tests.verify_reference_pipeline import main as verify_reference_pipeline


@pytest.mark.slow
@pytest.mark.skipif(
    os.environ.get("BENISSE_RUN_SLOW_TESTS") != "1",
    reason="set BENISSE_RUN_SLOW_TESTS=1 to run the complete Python + R oracle",
)
def test_complete_reference_pipeline():
    verify_reference_pipeline()
