from typing import List, Dict, Any
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from langchain_core.tools import tool
from agent_pipeline.session_state import SessionState

def make_tools(session: SessionState) -> List[Any]:
    """Factory function returning tools for tools_filtering.py bound to the provided session."""

    @tool
    def variance_filter_tool(threshold: float = 0.0) -> Dict[str, Any]:
        """Flags near-constant terms by variance across all documents in the document-term matrix.

        Computes term variance across session.feature_matrix columns, filters terms with variance
        below the specified threshold, stores a full report DataFrame sorted descending by variance,
        and returns filter counts and top term summaries.

        Args:
            threshold: Minimum variance required to keep a term (default 0.0).

        Returns:
            Dict containing result_id, threshold, total_terms, n_kept, n_removed, kept_terms, and removed_terms.
        """
        if session.feature_matrix is None or session.feature_names is None:
            return {"error": "Document-term matrix is not built yet. Please call build_dtm_tool first."}

        dense_dtm = session.feature_matrix.toarray()
        variances = np.var(dense_dtm, axis=0)
        terms = session.feature_names

        # Build full report DataFrame with term as 1st col, variance as 2nd col
        df_report = pd.DataFrame({"term": terms, "variance": variances})
        df_report.sort_values(by="variance", ascending=False, inplace=True)
        df_report.reset_index(drop=True, inplace=True)

        kept_df = df_report[df_report["variance"] >= threshold]
        removed_df = df_report[df_report["variance"] < threshold]

        kept_terms = kept_df["term"].tolist()
        removed_terms = removed_df["term"].tolist()

        result_id = session.next_result_id("variance_filter")
        summary = {
            "result_id": result_id,
            "threshold": threshold,
            "total_terms": len(terms),
            "n_kept": len(kept_terms),
            "n_removed": len(removed_terms),
            "kept_terms": kept_terms[:20],
            "removed_terms": removed_terms[:20],
        }

        session.store_result(
            "variance_filter_tool",
            {"threshold": threshold},
            summary,
            full_report=df_report
        )

        return summary

    @tool
    def pearson_filter_tool(target_class: str, threshold: float = 0.0) -> Dict[str, Any]:
        """Computes Pearson correlation coefficient between each term count and a target class indicator.

        Calculates Pearson correlation between each term column in session.feature_matrix and a binary
        one-vs-rest indicator for the target class in session.dataframe['category_name']. Filters terms
        whose absolute correlation is at or above threshold, sorted descending by pearson_r.

        Args:
            target_class: The category name to compute correlation against (one-vs-rest).
            threshold: Minimum absolute Pearson correlation required to keep a term (default 0.0).

        Returns:
            Dict containing result_id, target_class, threshold, total_terms, n_kept, n_removed, kept_terms, and removed_terms.
        """
        if session.feature_matrix is None or session.feature_names is None:
            return {"error": "Document-term matrix is not built yet. Please call build_dtm_tool first."}
        if session.dataframe is None or "category_name" not in session.dataframe.columns:
            return {"error": "DataFrame with category_name is required. Please call load_dataset_tool first."}

        dense_dtm = session.feature_matrix.toarray()
        binary_target = (session.dataframe["category_name"] == target_class).astype(int).values

        terms = session.feature_names
        pearson_r_list = []

        for i in range(dense_dtm.shape[1]):
            col = dense_dtm[:, i]
            if np.all(col == col[0]) or np.all(binary_target == binary_target[0]):
                r = 0.0
            else:
                r, _ = pearsonr(col, binary_target)
                if np.isnan(r):
                    r = 0.0
            pearson_r_list.append(float(r))

        df_report = pd.DataFrame({"term": terms, "pearson_r": pearson_r_list})
        df_report.sort_values(by="pearson_r", ascending=False, inplace=True)
        df_report.reset_index(drop=True, inplace=True)

        kept_df = df_report[df_report["pearson_r"].abs() >= threshold]
        removed_df = df_report[df_report["pearson_r"].abs() < threshold]

        kept_terms = kept_df["term"].tolist()
        removed_terms = removed_df["term"].tolist()

        result_id = session.next_result_id("pearson_filter")
        summary = {
            "result_id": result_id,
            "target_class": target_class,
            "threshold": threshold,
            "total_terms": len(terms),
            "n_kept": len(kept_terms),
            "n_removed": len(removed_terms),
            "kept_terms": kept_terms[:20],
            "removed_terms": removed_terms[:20],
        }

        session.store_result(
            "pearson_filter_tool",
            {"target_class": target_class, "threshold": threshold},
            summary,
            full_report=df_report
        )

        return summary

    @tool
    def spearman_filter_tool(target_class: str, threshold: float = 0.0) -> Dict[str, Any]:
        """Computes Spearman rank correlation coefficient between each term count and a target class indicator.

        Calculates Spearman rank correlation between each term column in session.feature_matrix and a binary
        one-vs-rest indicator for the target class in session.dataframe['category_name']. Filters terms
        whose absolute correlation is at or above threshold, sorted descending by spearman_r.

        Args:
            target_class: The category name to compute correlation against (one-vs-rest).
            threshold: Minimum absolute Spearman correlation required to keep a term (default 0.0).

        Returns:
            Dict containing result_id, target_class, threshold, total_terms, n_kept, n_removed, kept_terms, and removed_terms.
        """
        if session.feature_matrix is None or session.feature_names is None:
            return {"error": "Document-term matrix is not built yet. Please call build_dtm_tool first."}
        if session.dataframe is None or "category_name" not in session.dataframe.columns:
            return {"error": "DataFrame with category_name is required. Please call load_dataset_tool first."}

        dense_dtm = session.feature_matrix.toarray()
        binary_target = (session.dataframe["category_name"] == target_class).astype(int).values

        terms = session.feature_names
        spearman_r_list = []

        for i in range(dense_dtm.shape[1]):
            col = dense_dtm[:, i]
            if np.all(col == col[0]) or np.all(binary_target == binary_target[0]):
                r = 0.0
            else:
                r, _ = spearmanr(col, binary_target)
                if np.isnan(r):
                    r = 0.0
            spearman_r_list.append(float(r))

        df_report = pd.DataFrame({"term": terms, "spearman_r": spearman_r_list})
        df_report.sort_values(by="spearman_r", ascending=False, inplace=True)
        df_report.reset_index(drop=True, inplace=True)

        kept_df = df_report[df_report["spearman_r"].abs() >= threshold]
        removed_df = df_report[df_report["spearman_r"].abs() < threshold]

        kept_terms = kept_df["term"].tolist()
        removed_terms = removed_df["term"].tolist()

        result_id = session.next_result_id("spearman_filter")
        summary = {
            "result_id": result_id,
            "target_class": target_class,
            "threshold": threshold,
            "total_terms": len(terms),
            "n_kept": len(kept_terms),
            "n_removed": len(removed_terms),
            "kept_terms": kept_terms[:20],
            "removed_terms": removed_terms[:20],
        }

        session.store_result(
            "spearman_filter_tool",
            {"target_class": target_class, "threshold": threshold},
            summary,
            full_report=df_report
        )

        return summary

    return [variance_filter_tool, pearson_filter_tool, spearman_filter_tool]
