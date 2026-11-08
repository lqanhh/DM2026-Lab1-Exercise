import numpy as np
import pytest
from scipy.spatial import procrustes
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_dtm import make_tools as make_dtm_tools
from tools_reduction import make_tools as make_reduction_tools


def _setup_session_with_dtm():
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    dtm_tools = make_dtm_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    build_dtm_tool.invoke({})

    return session


def test_reduce_dimensions_pca_known_answer_7():
    session = _setup_session_with_dtm()
    reduction_tools = make_reduction_tools(session)
    reduce_dimensions_tool = next(t for t in reduction_tools if t.name == "reduce_dimensions_tool")

    res = reduce_dimensions_tool.invoke({"method": "pca", "n_components": 2})

    assert res["status"] == "success"
    assert res["method"] == "pca"

    # Known answer #7 coordinates
    expected_coords = [
        [2.3669, 0.8871],
        [1.7039, 0.2642],
        [1.0408, -0.3587],
        [-2.0759, 1.0129],
        [-1.4554, 0.3405],
        [-0.8349, -0.3319],
        [-0.8349, -0.3319],
        [0.0894, -1.4822]
    ]

    coords = np.array(res["coordinates"])
    np.testing.assert_allclose(coords, expected_coords, atol=1e-3)

    expected_variance = [0.7532, 0.1965]
    np.testing.assert_allclose(res["explained_variance_ratio"], expected_variance, atol=1e-3)

    assert session.pending_figure is not None


def test_reduce_dimensions_tsne_known_answer_8():
    session = _setup_session_with_dtm()
    reduction_tools = make_reduction_tools(session)
    reduce_dimensions_tool = next(t for t in reduction_tools if t.name == "reduce_dimensions_tool")

    res = reduce_dimensions_tool.invoke({
        "method": "tsne",
        "n_components": 2,
        "perplexity": 2,
        "random_state": 42
    })

    assert res["status"] == "success"
    assert res["method"] == "tsne"

    expected_coords = np.array([
        [1330.9991, 32.1039],
        [951.2459, 393.5588],
        [530.8209, 682.9791],
        [-1558.6448, -314.1625],
        [-915.5153, -346.3907],
        [-280.0313, -1261.2115],
        [-280.0313, -1261.2115],
        [-701.3674, 1465.3973]
    ])

    coords = np.array(res["coordinates"])

    # Check that identical documents (row 5 and row 6) receive identical coordinates
    np.testing.assert_allclose(coords[5], coords[6], atol=1e-3)
    
    # Procrustes analysis compares shapes after optimal rotation, translation, and scaling
    mtx1, mtx2, disparity = procrustes(expected_coords, coords)
    assert disparity < 0.6, f"t-SNE disparity {disparity} exceeds expected shape similarity threshold 0.6"

    assert session.pending_figure is not None


def test_reduce_dimensions_umap_known_answer_9():
    session = _setup_session_with_dtm()
    reduction_tools = make_reduction_tools(session)
    reduce_dimensions_tool = next(t for t in reduction_tools if t.name == "reduce_dimensions_tool")

    res = reduce_dimensions_tool.invoke({
        "method": "umap",
        "n_components": 2,
        "n_neighbors": 3,
        "random_state": 42
    })

    assert res["status"] == "success"
    assert res["method"] == "umap"

    expected_coords = [
        [6.6811, -6.8218],
        [6.9907, -7.2245],
        [7.7022, -7.3712],
        [10.2228, -6.2310],
        [9.8759, -5.6537],
        [9.3831, -6.5748],
        [9.2153, -5.8297],
        [8.5860, -7.2622]
    ]

    coords = np.array(res["coordinates"])
    np.testing.assert_allclose(coords, expected_coords, atol=1e-1)

    assert session.pending_figure is not None


def test_binarize_labels_tool():
    # 1. Error case: empty session with no labels
    session_empty = SessionState()
    tools_empty = make_reduction_tools(session_empty)
    binarize_tool_empty = next(t for t in tools_empty if t.name == "binarize_labels_tool")
    res_empty = binarize_tool_empty.invoke({})
    assert res_empty["status"] == "error"

    # 2. Standard fixture case (2 categories: catA, catB)
    session = _setup_session_with_dtm()
    tools = make_reduction_tools(session)
    binarize_tool = next(t for t in tools if t.name == "binarize_labels_tool")

    res = binarize_tool.invoke({})
    assert res["status"] == "success"
    assert res["categories"] == ["catA", "catB"]
    assert res["shape"] == [8, 2]

    expected_matrix = [
        [1, 0],  # row 0: catA
        [1, 0],  # row 1: catA
        [1, 0],  # row 2: catA
        [0, 1],  # row 3: catB
        [0, 1],  # row 4: catB
        [0, 1],  # row 5: catB
        [0, 1],  # row 6: catB
        [1, 0],  # row 7: catA
    ]
    assert res["binary_matrix"] == expected_matrix
    np.testing.assert_array_equal(session.artifacts["binarized_labels"], expected_matrix)

    # 3. Custom multi-category case with 3 categories and a missing/None label
    session_custom = SessionState()
    session_custom.labels = np.array(["catA", "catC", None, "catB", "catA", "catC"], dtype=object)
    tools_custom = make_reduction_tools(session_custom)
    binarize_custom = next(t for t in tools_custom if t.name == "binarize_labels_tool")

    res_custom = binarize_custom.invoke({})
    assert res_custom["status"] == "success"
    assert res_custom["categories"] == ["catA", "catB", "catC"]
    assert res_custom["shape"] == [6, 3]

    # Column ordering: [catA, catB, catC]
    expected_custom_matrix = [
        [1, 0, 0],  # catA
        [0, 0, 1],  # catC
        [0, 0, 0],  # None (missing value -> all zeros)
        [0, 1, 0],  # catB
        [1, 0, 0],  # catA
        [0, 0, 1],  # catC
    ]
    assert res_custom["binary_matrix"] == expected_custom_matrix
