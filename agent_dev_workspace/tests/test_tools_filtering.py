import pytest
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_dtm import make_tools as make_dtm_tools
from tools_filtering import make_tools as make_filtering_tools

def test_variance_filter_tool_known_answer_5():
    session = SessionState()

    # Load premade load_dataset_tool
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label",
    })

    # Build DTM using student's tools_dtm
    dtm_tools = make_dtm_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    build_dtm_tool.invoke({})

    # Load student's tools_filtering
    filtering_tools = make_filtering_tools(session)
    variance_filter_tool = next(t for t in filtering_tools if t.name == "variance_filter_tool")

    # Run variance_filter_tool with threshold=0.15 matching known answer #5
    res = variance_filter_tool.invoke({"threshold": 0.15})

    assert res["total_terms"] == 5
    assert res["n_kept"] == 4
    assert res["n_removed"] == 1

    # Check kept vs removed term sets
    assert set(res["kept_terms"]) == {"alpha", "beta", "delta", "gamma"}
    assert set(res["removed_terms"]) == {"always"}

    # Retrieve full report stored in results_store and verify expected variances
    result_id = res["result_id"]
    stored_entry = session.results_store.get(result_id)
    assert stored_entry is not None

    full_report = stored_entry["full_report"]
    var_dict = dict(zip(full_report["term"], full_report["variance"]))

    assert var_dict["alpha"] == pytest.approx(1.1875, abs=1e-3)
    assert var_dict["gamma"] == pytest.approx(1.109375, abs=1e-3)
    assert var_dict["delta"] == pytest.approx(0.25, abs=1e-3)
    assert var_dict["beta"] == pytest.approx(0.234375, abs=1e-3)
    assert var_dict["always"] == pytest.approx(0.109375, abs=1e-3)


def test_pearson_filter_tool_known_answer_6():
    session = SessionState()

    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label",
    })

    dtm_tools = make_dtm_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    build_dtm_tool.invoke({})

    filtering_tools = make_filtering_tools(session)
    pearson_filter_tool = next(t for t in filtering_tools if t.name == "pearson_filter_tool")

    res = pearson_filter_tool.invoke({"target_class": "catB", "threshold": 0.0})

    assert res["total_terms"] == 5
    assert res["target_class"] == "catB"

    result_id = res["result_id"]
    stored_entry = session.results_store.get(result_id)
    assert stored_entry is not None

    full_report = stored_entry["full_report"]
    r_dict = dict(zip(full_report["term"], full_report["pearson_r"]))

    # Known answer #6 values for Pearson r (target_class="catB")
    assert r_dict["alpha"] == pytest.approx(-0.6882, abs=1e-3)
    assert r_dict["always"] == pytest.approx(0.3780, abs=1e-3)
    assert r_dict["beta"] == pytest.approx(-0.7746, abs=1e-3)
    assert r_dict["delta"] == pytest.approx(1.0000, abs=1e-3)
    assert r_dict["gamma"] == pytest.approx(0.8307, abs=1e-3)


def test_spearman_filter_tool_known_answer_6():
    session = SessionState()

    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label",
    })

    dtm_tools = make_dtm_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    build_dtm_tool.invoke({})

    filtering_tools = make_filtering_tools(session)
    spearman_filter_tool = next(t for t in filtering_tools if t.name == "spearman_filter_tool")

    res = spearman_filter_tool.invoke({"target_class": "catB", "threshold": 0.0})

    assert res["total_terms"] == 5
    assert res["target_class"] == "catB"

    result_id = res["result_id"]
    stored_entry = session.results_store.get(result_id)
    assert stored_entry is not None

    full_report = stored_entry["full_report"]
    r_dict = dict(zip(full_report["term"], full_report["spearman_r"]))

    # Known answer #6 values for Spearman r (target_class="catB")
    assert r_dict["alpha"] == pytest.approx(-0.7500, abs=1e-3)
    assert r_dict["always"] == pytest.approx(0.3780, abs=1e-3)
    assert r_dict["beta"] == pytest.approx(-0.7746, abs=1e-3)
    assert r_dict["delta"] == pytest.approx(1.0000, abs=1e-3)
    assert r_dict["gamma"] == pytest.approx(0.9363, abs=1e-3)
