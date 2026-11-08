import pytest
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_patterns import make_tools as make_patterns_tools

def test_mine_patterns_tool_fpgrowth_known_answer_17():
    session = SessionState()

    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label",
    })

    pattern_tools = make_patterns_tools(session)
    mine_patterns_tool = next(t for t in pattern_tools if t.name == "mine_patterns_tool")

    # 1. catA with filtering_method="tfidf", algorithm="fpgrowth", min_sup=1
    res_catA = mine_patterns_tool.invoke({
        "category_name": "catA",
        "filtering_method": "tfidf",
        "algorithm": "fpgrowth",
        "min_sup": 1,
    })

    assert res_catA["kept_terms"] == ["alpha"]
    assert len(res_catA["patterns"]) == 1
    assert "alpha" in res_catA["patterns"][0]["pattern"]
    assert res_catA["patterns"][0]["support"] == 3

    # 2. catB with filtering_method="tfidf", algorithm="fpgrowth", min_sup=1
    res_catB = mine_patterns_tool.invoke({
        "category_name": "catB",
        "filtering_method": "tfidf",
        "algorithm": "fpgrowth",
        "min_sup": 1,
    })

    assert res_catB["kept_terms"] == ["gamma"]
    assert len(res_catB["patterns"]) == 1
    assert "gamma" in res_catB["patterns"][0]["pattern"]
    assert res_catB["patterns"][0]["support"] == 4

    # 3. catA with filtering_method="variance" -> 0 terms survive
    res_var = mine_patterns_tool.invoke({
        "category_name": "catA",
        "filtering_method": "variance",
        "algorithm": "fpgrowth",
        "min_sup": 1,
    })

    assert res_var["kept_terms"] == []
    assert res_var["patterns"] == []
    assert "no terms survived filtering" in res_var.get("note", "")


def test_mine_patterns_tool_topk_and_maxfpgrowth_known_answer_20():
    session = SessionState()

    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label",
    })

    pattern_tools = make_patterns_tools(session)
    mine_patterns_tool = next(t for t in pattern_tools if t.name == "mine_patterns_tool")

    # 1. Top-K algorithm with k=1 for catA
    res_topk = mine_patterns_tool.invoke({
        "category_name": "catA",
        "filtering_method": "tfidf",
        "algorithm": "topk",
        "k": 1,
    })

    assert len(res_topk["patterns"]) == 1
    assert "alpha" in res_topk["patterns"][0]["pattern"]
    assert res_topk["patterns"][0]["support"] == 3

    # 2. MaxFPGrowth algorithm for catA -> returns 0 patterns on single-term data
    res_max = mine_patterns_tool.invoke({
        "category_name": "catA",
        "filtering_method": "tfidf",
        "algorithm": "maxfpgrowth",
        "min_sup": 1,
    })

    assert res_max["patterns"] == []
