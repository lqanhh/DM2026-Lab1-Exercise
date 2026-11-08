import numpy as np
import pytest
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_data import make_tools

def test_tokenize_tool_known_answer_10():
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
    tokenize_tool = next(t for t in tools if t.name == "tokenize_tool")

    tok_res = tokenize_tool.invoke({})

    assert tok_res["status"] == "success"
    assert tok_res["n_documents"] == 8

    expected_tokens_by_doc = [
        ["always", "alpha", "alpha", "alpha", "beta"],
        ["always", "alpha", "alpha", "beta"],
        ["always", "alpha", "beta"],
        ["always", "gamma", "gamma", "gamma", "delta"],
        ["always", "gamma", "gamma", "delta"],
        ["always", "gamma", "delta"],
        ["always", "gamma", "delta"],
        []
    ]

    assert "unigrams" in session.artifacts
    unigrams = session.artifacts["unigrams"]
    assert len(unigrams) == 8

    for doc_idx, expected_tokens in enumerate(expected_tokens_by_doc):
        assert unigrams[doc_idx] == expected_tokens

def test_list_files_tool_known_answer_11():
    session = SessionState()
    tools = make_tools(session)
    list_files_tool = next(t for t in tools if t.name == "list_files_tool")

    res = list_files_tool.invoke({"subdirectory": "newdataset"})

    assert res["status"] == "success"
    assert res["directories"] == []
    assert res["files"] == ["Reddit-stock-sentiment.csv"]

def test_check_missing_tool_known_answer_2():
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    tools = make_tools(session)
    check_missing_tool = next(t for t in tools if t.name == "check_missing_tool")

    res = check_missing_tool.invoke({})

    # Known answer #2: Exactly 1 row has missing/empty text (the last row, index 7)
    assert res["status"] == "success"
    assert res["n_missing"] == 1
    assert res["missing_indices"] == [7]

def test_check_duplicates_tool_known_answer_3():
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    tools = make_tools(session)
    check_duplicates_tool = next(t for t in tools if t.name == "check_duplicates_tool")

    # Known answer #3 check: rows 5 and 6 are exact duplicates
    res_check = check_duplicates_tool.invoke({"drop": False})
    assert res_check["status"] == "success"
    assert res_check["n_duplicates"] == 2
    assert res_check["duplicate_indices"] == [5, 6]
    assert res_check["remaining_rows"] == 8

    # Known answer #3 drop: dropping with keep=False leaves exactly 6 rows (both 5 and 6 removed)
    res_drop = check_duplicates_tool.invoke({"drop": True})
    assert res_drop["status"] == "success"
    assert res_drop["dropped"] is True
    assert res_drop["remaining_rows"] == 6
    assert len(session.dataframe) == 6

def test_sample_data_tool_known_answer_13():
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    tools = make_tools(session)
    sample_data_tool = next(t for t in tools if t.name == "sample_data_tool")

    # Known answer #13: n=4, random_state=42 returns exactly rows [1, 5, 0, 7]
    res_4 = sample_data_tool.invoke({"n": 4, "random_state": 42})
    assert res_4["status"] == "success"
    assert res_4["n_sampled"] == 4
    assert res_4["sample_indices"] == [1, 5, 0, 7]

    # Boundary test: n=20 when DataFrame only has 8 rows
    res_20 = sample_data_tool.invoke({"n": 20, "random_state": 42})
    assert res_20["status"] == "success"
    assert res_20["n_requested"] == 20
    assert res_20["n_sampled"] == 8
    assert len(res_20["sample_indices"]) == 8

def test_describe_data_tool_known_answer_18():
    # Single session reused for load_dataset_tool and describe_data_tool
    session = SessionState()

    # Load premade load_dataset_tool
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    # Load test fixture
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label",
    })

    # Load student's tools_data
    student_tools = make_tools(session)
    describe_data_tool = next(t for t in student_tools if t.name == "describe_data_tool")

    # Execute describe_data_tool
    res = describe_data_tool.invoke({})

    # Check that text_length list in artifacts matches known answer #18
    text_lengths = session.artifacts.get("text_length")
    if text_lengths is not None:
        expected_lengths = [29, 23, 17, 30, 24, 18, 18, 0]
        np.testing.assert_array_equal(text_lengths, expected_lengths)

    # Inspect summary or stored result metrics
    stats = res.get("overall_stats", res.get("stats", res))

    assert stats["count"] == 8
    assert stats["mean"] == pytest.approx(19.875, abs=1e-3)
    assert stats["std"] == pytest.approx(9.4330, abs=1e-3)
    assert stats["min"] == 0
    assert stats["25%"] == pytest.approx(17.75, abs=1e-3)
    assert stats["50%"] == pytest.approx(20.5, abs=1e-3)
    assert stats["75%"] == pytest.approx(25.25, abs=1e-3)
    assert stats["max"] == 30

    # Per-category stats checks if included in returned dict
    if "by_category" in res:
        cat_a = res["by_category"]["catA"]
        assert cat_a["count"] == 4
        assert cat_a["mean"] == pytest.approx(17.25, abs=1e-3)
        assert cat_a["std"] == pytest.approx(12.5, abs=1e-3)
        assert cat_a["min"] == 0
        assert cat_a["25%"] == pytest.approx(12.75, abs=1e-3)
        assert cat_a["50%"] == pytest.approx(20.0, abs=1e-3)
        assert cat_a["75%"] == pytest.approx(24.5, abs=1e-3)
        assert cat_a["max"] == 29

        cat_b = res["by_category"]["catB"]
        assert cat_b["count"] == 4
        assert cat_b["mean"] == pytest.approx(22.5, abs=1e-3)
        assert cat_b["std"] == pytest.approx(5.7446, abs=1e-3)
        assert cat_b["min"] == 18
        assert cat_b["25%"] == pytest.approx(18.0, abs=1e-3)
        assert cat_b["50%"] == pytest.approx(21.0, abs=1e-3)
        assert cat_b["75%"] == pytest.approx(25.5, abs=1e-3)
        assert cat_b["max"] == 30

    # Check plotting requirement
    assert session.pending_figure is not None

def test_inspect_data_tool():
    # 1. Error state test: no dataframe loaded
    session_empty = SessionState()
    tools_empty = make_tools(session_empty)
    inspect_data_tool_empty = next(t for t in tools_empty if t.name == "inspect_data_tool")
    res_empty = inspect_data_tool_empty.invoke({})
    assert res_empty["status"] == "error"

    # 2. Loaded state setup
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    tools = make_tools(session)
    inspect_data_tool = next(t for t in tools if t.name == "inspect_data_tool")

    # Default n=5 test
    res_default = inspect_data_tool.invoke({})
    assert res_default["status"] == "success"
    assert res_default["dimensions"] == [8, 3]
    assert "text" in res_default["columns"]
    assert "category_name" in res_default["columns"]
    assert res_default["inspected_rows"] == 5
    assert len(res_default["sample_data"]) == 5

    # Custom n=3 test
    res_3 = inspect_data_tool.invoke({"n": 3})
    assert res_3["status"] == "success"
    assert res_3["inspected_rows"] == 3
    assert len(res_3["sample_data"]) == 3

    # Boundary n=10 test (n > len(df))
    res_10 = inspect_data_tool.invoke({"n": 10})
    assert res_10["status"] == "success"
    assert res_10["inspected_rows"] == 8
    assert len(res_10["sample_data"]) == 8
