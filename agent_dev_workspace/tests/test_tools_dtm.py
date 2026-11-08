import pytest
import numpy as np
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_dtm import make_tools

def test_build_dtm_tool_known_answer_4():
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    load_res = load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })
    assert load_res["n_documents"] == 8

    tools = make_tools(session)
    build_dtm_tool = next(t for t in tools if t.name == "build_dtm_tool")

    dtm_res = build_dtm_tool.invoke({})

    assert dtm_res["n_terms"] == 5
    assert session.feature_names == ["alpha", "always", "beta", "delta", "gamma"]
    assert dtm_res["n_documents"] == 8
    assert dtm_res["non_zero"] == 21
    assert dtm_res["total_elements"] == 40
    assert pytest.approx(dtm_res["sparsity_pct"], abs=1e-3) == 47.5
    assert "count_vectorizer" in session.artifacts

def test_term_frequency_tool_known_answer_14():
    # Instantiate fresh session for test isolation
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    tools = make_tools(session)
    build_dtm_tool = next(t for t in tools if t.name == "build_dtm_tool")
    term_frequency_tool = next(t for t in tools if t.name == "term_frequency_tool")

    # First build the global DTM
    build_dtm_tool.invoke({})

    # Execute term_frequency_tool
    tf_res = term_frequency_tool.invoke({"top_n": 20})

    # Assert known answer #14 from TEST_FIXTURE.md:
    # alpha: 6, always: 7, beta: 3, delta: 4, gamma: 7
    expected_frequencies = {
        "alpha": 6,
        "always": 7,
        "beta": 3,
        "delta": 4,
        "gamma": 7
    }

    assert tf_res["total_unique_terms"] == 5
    assert tf_res["total_term_occurrences"] == 27

    for term, expected_count in expected_frequencies.items():
        assert tf_res["top_terms"][term] == expected_count

def test_dtm_heatmap_tool_known_answer_21():
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    tools = make_tools(session)
    build_dtm_tool = next(t for t in tools if t.name == "build_dtm_tool")
    dtm_heatmap_tool = next(t for t in tools if t.name == "dtm_heatmap_tool")

    build_dtm_tool.invoke({})

    # Full matrix slice check with defaults (n_terms=20, n_documents=20)
    res_full = dtm_heatmap_tool.invoke({"n_terms": 20, "n_documents": 20})

    assert res_full["n_documents"] == 8
    assert res_full["n_terms"] == 5
    assert res_full["terms"] == ["alpha", "always", "beta", "delta", "gamma"]

    # Known answer #21 full 8x5 matrix check:
    expected_full_matrix = [
        [3, 1, 1, 0, 0],  # Doc 0
        [2, 1, 1, 0, 0],  # Doc 1
        [1, 1, 1, 0, 0],  # Doc 2
        [0, 1, 0, 1, 3],  # Doc 3
        [0, 1, 0, 1, 2],  # Doc 4
        [0, 1, 0, 1, 1],  # Doc 5
        [0, 1, 0, 1, 1],  # Doc 6
        [0, 0, 0, 0, 0],  # Doc 7
    ]
    np.testing.assert_array_equal(res_full["matrix"], expected_full_matrix)
    assert session.pending_figure is not None

    # Corner slice check (n_terms=2, n_documents=3)
    res_corner = dtm_heatmap_tool.invoke({"n_terms": 2, "n_documents": 3})
    assert res_corner["n_documents"] == 3
    assert res_corner["n_terms"] == 2
    assert res_corner["terms"] == ["alpha", "always"]

    expected_corner_matrix = [
        [3, 1],
        [2, 1],
        [1, 1],
    ]
    np.testing.assert_array_equal(res_corner["matrix"], expected_corner_matrix)
