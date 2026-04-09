import pytest
from unittest.mock import MagicMock
from engine import perform_gap_analysis

def test_perform_gap_analysis_mocked():
    # 1. Mock the LLM (A Level: Mock Objects)
    mock_llm = MagicMock()
    mock_llm.invoke.return_value.content = "FLAGGED: Missing Human Appeal Path."

    # 2. Call your actual function
    result = perform_gap_analysis(
        content="Artificial Intelligence policy for brain-computer interface.",
        laws=["Colorado SB 24-205"],
        org="Synchron",
        llm=mock_llm
    )

    # 3. Assertions (A Level: Unit Tests)
    assert "FLAGGED" in result
    mock_llm.invoke.assert_called_once()
