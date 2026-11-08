import pytest
import numpy as np
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_exploration import make_tools

def test_cosine_similarity_known_answer_16():
    """
    Test cosine_similarity_tool against known answer #16 in TEST_FIXTURE.md.
    
    Checks raw count cosine similarity on sample_fixture.csv:
    - 0 vs 1 (both catA, near-identical text) ~ 0.9847
    - 0 vs 3 (catA vs catB, share only 'always') ~ 0.0909
    - 3 vs 4 (both catB, near-identical text) ~ 0.9847
    - 0 vs 7 (catA vs empty text) == 0.0
    """
    session = SessionState()
    
    # 1. Load sample dataset via premade tool
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })
    
    # 2. Build DTM so session.feature_matrix and vectorizer artifact exist
    from tools_dtm import make_tools as make_dtm_tools
    dtm_tools = make_dtm_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    build_dtm_tool.invoke({})
    
    # 3. Get cosine_similarity_tool
    exploration_tools = make_tools(session)
    cosine_tool = next(t for t in exploration_tools if t.name == "cosine_similarity_tool")
    
    # Pair 0 vs 1: expected ~ 0.9847
    res_0_1 = cosine_tool.invoke({"doc_index_1": 0, "doc_index_2": 1})
    assert res_0_1["similarity"] == pytest.approx(0.9847, abs=1e-3)
    
    # Pair 0 vs 3: expected ~ 0.0909
    res_0_3 = cosine_tool.invoke({"doc_index_1": 0, "doc_index_2": 3})
    assert res_0_3["similarity"] == pytest.approx(0.0909, abs=1e-3)
    
    # Pair 3 vs 4: expected ~ 0.9847
    res_3_4 = cosine_tool.invoke({"doc_index_1": 3, "doc_index_2": 4})
    assert res_3_4["similarity"] == pytest.approx(0.9847, abs=1e-3)
    
    # Pair 0 vs 7: expected 0.0 (empty text edge case)
    res_0_7 = cosine_tool.invoke({"doc_index_1": 0, "doc_index_2": 7})
    assert res_0_7["similarity"] == pytest.approx(0.0, abs=1e-3)


def test_feature_correlation_matrix_known_answer_19():
    """
    Test feature_correlation_matrix_tool against known answer #19 in TEST_FIXTURE.md.

    Checks top 20 terms by variance on sample_fixture.csv (all 5 terms fit):
    - Order by variance descending: alpha, gamma, delta, beta, always
    - Correlation matrix values:
      - alpha vs alpha: 1.0000
      - alpha vs gamma: -0.5718
      - alpha vs delta: -0.6882
      - alpha vs beta:  0.8885
      - alpha vs always: 0.2601
      - gamma vs delta: 0.8307
      - delta vs beta: -0.7746
      - always vs always: 1.0000
    - Figure set on session.pending_figure
    """
    session = SessionState()

    # 1. Load sample dataset via premade tool
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    # 2. Build DTM
    from tools_dtm import make_tools as make_dtm_tools
    dtm_tools = make_dtm_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    build_dtm_tool.invoke({})

    # 3. Get feature_correlation_matrix_tool
    exploration_tools = make_tools(session)
    corr_tool = next(t for t in exploration_tools if t.name == "feature_correlation_matrix_tool")

    res = corr_tool.invoke({"top_n": 20})

    # Term order by descending variance
    expected_terms = ["alpha", "gamma", "delta", "beta", "always"]
    assert res["term_names"] == expected_terms

    # Expected correlation matrix from known answer #19
    expected_matrix = [
        [1.0000, -0.5718, -0.6882,  0.8885,  0.2601],
        [-0.5718,  1.0000,  0.8307, -0.6435,  0.3140],
        [-0.6882,  0.8307,  1.0000, -0.7746,  0.3780],
        [ 0.8885, -0.6435, -0.7746,  1.0000,  0.2928],
        [ 0.2601,  0.3140,  0.3780,  0.2928,  1.0000]
    ]

    actual_matrix = res["correlation_matrix"]
    assert np.allclose(actual_matrix, expected_matrix, atol=1e-3)

    # Verify pending figure was set
    assert session.pending_figure is not None
