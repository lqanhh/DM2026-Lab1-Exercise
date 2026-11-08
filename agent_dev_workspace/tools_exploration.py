from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from langchain_core.tools import tool
from agent_pipeline.session_state import SessionState

def make_tools(session: SessionState) -> List[Any]:
    """Factory function returning exploration tools bound to the provided session."""

    @tool
    def cosine_similarity_tool(
        doc_index_1: Optional[int] = None,
        doc_index_2: Optional[int] = None,
        text1: Optional[str] = None,
        text2: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Computes cosine similarity between two documents.

        Can compare two documents in session.feature_matrix by index (doc_index_1 and doc_index_2),
        or compare two raw text inputs (text1 and text2) using session.artifacts['count_vectorizer'].

        Args:
            doc_index_1: 0-based index of the first document in session.feature_matrix.
            doc_index_2: 0-based index of the second document in session.feature_matrix.
            text1: Raw text string for first document.
            text2: Raw text string for second document.

        Returns:
            Dict containing result_id, similarity, and metadata details.
        """
        if doc_index_1 is not None and doc_index_2 is not None:
            if session.feature_matrix is None:
                return {"error": "Feature matrix (DTM) is not built yet. Please run build_dtm_tool first."}
            
            n_docs = session.feature_matrix.shape[0]
            if doc_index_1 < 0 or doc_index_1 >= n_docs or doc_index_2 < 0 or doc_index_2 >= n_docs:
                return {"error": f"Document indices must be between 0 and {n_docs - 1}."}

            v1 = session.feature_matrix[doc_index_1]
            v2 = session.feature_matrix[doc_index_2]
            score = float(cosine_similarity(v1, v2)[0, 0])

            result_id = session.next_result_id("cosine_similarity")
            summary = {
                "result_id": result_id,
                "doc_index_1": doc_index_1,
                "doc_index_2": doc_index_2,
                "similarity": round(score, 4),
            }
            session.store_result("cosine_similarity_tool", {"doc_index_1": doc_index_1, "doc_index_2": doc_index_2}, summary)
            return summary

        elif text1 is not None and text2 is not None:
            vectorizer = session.artifacts.get("count_vectorizer")
            if vectorizer is None:
                return {"error": "Count vectorizer not found in session.artifacts. Please run build_dtm_tool first."}

            v1 = vectorizer.transform([text1])
            v2 = vectorizer.transform([text2])
            score = float(cosine_similarity(v1, v2)[0, 0])

            result_id = session.next_result_id("cosine_similarity")
            summary = {
                "result_id": result_id,
                "text1": text1,
                "text2": text2,
                "similarity": round(score, 4),
            }
            session.store_result("cosine_similarity_tool", {"text1": text1, "text2": text2}, summary)
            return summary

        else:
            return {"error": "Must provide either both (doc_index_1, doc_index_2) or both (text1, text2)."}

    @tool
    def feature_correlation_matrix_tool(top_n: int = 20) -> Dict[str, Any]:
        """Computes feature-vs-feature Pearson correlation matrix for top variance terms and plots a heatmap.

        Operates on the global document-term matrix (session.feature_matrix). Selects up to top_n
        terms with highest variance, computes Pearson correlation (np.corrcoef), plots an annotated
        heatmap using seaborn, sets session.pending_figure, and stores the result.

        Args:
            top_n: Number of highest-variance terms to include (default 20).

        Returns:
            Dict containing result_id, term_names, and correlation_matrix (as nested list).
        """
        if session.feature_matrix is None or session.feature_names is None:
            return {"error": "Feature matrix (DTM) is not built yet. Please run build_dtm_tool first."}

        # Convert sparse matrix to dense array (n_docs, n_terms)
        X = session.feature_matrix.toarray() if hasattr(session.feature_matrix, "toarray") else np.asarray(session.feature_matrix)
        feature_names = np.array(session.feature_names)

        if X.shape[1] == 0:
            return {"error": "Feature matrix has zero columns."}

        # Compute term variances across all documents
        variances = np.var(X, axis=0)

        # Take top_n by variance (descending order)
        n_terms_to_take = min(top_n, X.shape[1])
        top_indices = np.argsort(variances)[::-1][:n_terms_to_take]

        selected_X = X[:, top_indices]
        selected_terms = feature_names[top_indices].tolist()

        # Compute Pearson correlation matrix across selected terms: shape (n_terms, n_terms)
        # np.corrcoef expects variables as rows, so transpose selected_X (shape: n_terms x n_docs)
        corr_matrix = np.corrcoef(selected_X.T)
        if np.ndim(corr_matrix) == 0:
            corr_matrix = np.array([[1.0]])

        # Handle NaNs from zero-variance features if any
        corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)

        # Build annotated heatmap
        fig, ax = plt.subplots(figsize=(max(6, n_terms_to_take * 0.8), max(5, n_terms_to_take * 0.7)))
        sns.heatmap(
            corr_matrix,
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            xticklabels=selected_terms,
            yticklabels=selected_terms,
            ax=ax,
            vmin=-1.0,
            vmax=1.0,
        )
        ax.set_title("Feature Correlation Matrix (Top Variance Terms)")
        plt.tight_layout()

        session.pending_figure = fig

        result_id = session.next_result_id("feature_correlation")
        rounded_corr = np.round(corr_matrix, 4).tolist()

        summary = {
            "result_id": result_id,
            "top_n": n_terms_to_take,
            "term_names": selected_terms,
            "correlation_matrix": rounded_corr,
        }

        # Full report as DataFrame for store_result
        full_df = pd.DataFrame(rounded_corr, index=selected_terms, columns=selected_terms)
        session.store_result("feature_correlation_matrix_tool", {"top_n": top_n}, summary, full_report=full_df)

        return summary

    return [cosine_similarity_tool, feature_correlation_matrix_tool]
