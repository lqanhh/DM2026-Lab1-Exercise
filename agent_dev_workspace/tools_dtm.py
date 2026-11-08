from typing import List, Optional, Dict, Any, Union
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
from langchain_core.tools import tool

def make_tools(session):
    @tool
    def build_dtm_tool(
        ngram_range: List[int] = [1, 1],
        min_df: Union[int, float] = 1,
        max_df: Union[int, float] = 1.0
    ) -> Dict[str, Any]:
        """Builds a Document-Term Matrix (DTM) from document text in session.dataframe.

        Transforms text into a matrix of token counts using CountVectorizer,
        populates session feature state, computes matrix sparsity, and stores
        the vectorizer artifact.

        Args:
            ngram_range: Lower and upper boundary of range of n-values for n-grams, e.g. [1, 1].
            min_df: Ignore terms that have a document frequency strictly lower than threshold.
            max_df: Ignore terms that have a document frequency strictly higher than threshold.

        Returns:
            Dict containing result_id, document count, term count, sparsity details,
            and vocabulary sample.
        """
        if session.dataframe is None or session.dataframe.empty:
            return {"error": "session.dataframe is missing or empty. Please load a dataset first."}

        text_series = session.dataframe["text"].fillna("")

        # Convert min_df / max_df to int only if > 1.0 or if passed as a float integer like 2.0
        # Preserve float 1.0 as proportion (100%), while int 1 means absolute count of 1
        if isinstance(min_df, float) and min_df > 1.0 and min_df.is_integer():
            min_df_val = int(min_df)
        else:
            min_df_val = min_df

        if isinstance(max_df, float) and max_df > 1.0 and max_df.is_integer():
            max_df_val = int(max_df)
        else:
            max_df_val = max_df

        vectorizer = CountVectorizer(
            ngram_range=tuple(ngram_range),
            min_df=min_df_val,
            max_df=max_df_val
        )

        dtm_matrix = vectorizer.fit_transform(text_series)
        feature_names = vectorizer.get_feature_names_out().tolist()

        # Update session attributes
        session.feature_matrix = dtm_matrix
        session.feature_names = feature_names

        # Store fitted vectorizer artifact for downstream tools like cosine_similarity_tool
        session.artifacts["count_vectorizer"] = vectorizer

        # Set labels and categories if label or category info is present
        if "category_name" in session.dataframe.columns:
            session.set_labels(session.dataframe["category_name"].values)
        elif "category" in session.dataframe.columns:
            session.set_labels(session.dataframe["category"].values)

        # Calculate sparsity exactly as specified by Master notebook / spec
        n_rows, n_cols = dtm_matrix.shape
        non_zero = int(dtm_matrix.nnz)
        total_elements = int(n_rows * n_cols)
        sparsity_pct = 100.0 * (1.0 - non_zero / total_elements) if total_elements > 0 else 0.0

        result_id = session.next_result_id("dtm")
        summary = {
            "result_id": result_id,
            "n_documents": int(n_rows),
            "n_terms": int(n_cols),
            "non_zero": non_zero,
            "total_elements": total_elements,
            "sparsity_pct": round(sparsity_pct, 4),
            "vocabulary_sample": feature_names[:10]
        }

        session.store_result("build_dtm_tool", {"ngram_range": ngram_range, "min_df": min_df, "max_df": max_df}, summary)
        return summary

    @tool
    def term_frequency_tool(top_n: int = 20) -> Dict[str, Any]:
        """Aggregates term frequencies across all documents in the document-term matrix.

        Sums occurrences for each term, ranks them, stores a full report DataFrame,
        and returns top term frequencies in the summary.

        Args:
            top_n: Number of top most frequent terms to include in the returned summary.

        Returns:
            Dict containing result_id, total unique terms, total term occurrences,
            and top term frequencies.
        """
        if session.feature_matrix is None or session.feature_names is None:
            return {"error": "Document-term matrix is not built yet. Please call build_dtm_tool first."}

        # Sum term counts across all documents (axis=0)
        counts = np.asarray(session.feature_matrix.sum(axis=0)).flatten().tolist()
        terms = session.feature_names

        # Build full report DataFrame and sort descending by frequency
        df_tf = pd.DataFrame({"term": terms, "frequency": counts})
        df_tf.sort_values(by="frequency", ascending=False, inplace=True)
        df_tf.reset_index(drop=True, inplace=True)

        result_id = session.next_result_id("term_freq")

        top_terms_dict = dict(zip(df_tf["term"].head(top_n), df_tf["frequency"].head(top_n)))

        summary = {
            "result_id": result_id,
            "total_unique_terms": len(terms),
            "total_term_occurrences": int(sum(counts)),
            "top_terms": top_terms_dict
        }

        session.store_result("term_frequency_tool", {"top_n": top_n}, summary, full_report=df_tf)
        return summary

    @tool
    def dtm_heatmap_tool(n_terms: int = 20, n_documents: int = 20) -> Dict[str, Any]:
        """Generates a heatmap of a raw positional slice of the document-term matrix.

        Slices the document-term matrix for the specified number of terms and documents,
        renders an annotated heatmap onto session.pending_figure, and returns the slice data.

        Args:
            n_terms: Number of terms (columns) to slice from the beginning (default 20).
            n_documents: Number of documents (rows) to slice from the beginning (default 20).

        Returns:
            Dict containing result_id, n_documents, n_terms, terms list, and matrix slice as nested lists.
        """
        if session.feature_matrix is None or session.feature_names is None:
            return {"error": "Document-term matrix is not built yet. Please call build_dtm_tool first."}

        total_docs, total_terms = session.feature_matrix.shape

        actual_docs = min(n_documents, total_docs)
        actual_terms = min(n_terms, total_terms)

        sliced_matrix = session.feature_matrix[:actual_docs, :actual_terms].toarray()
        sliced_terms = session.feature_names[:actual_terms]
        doc_labels = [f"Doc {i}" for i in range(actual_docs)]

        # Render heatmap figure
        fig, ax = plt.subplots(figsize=(max(6, actual_terms * 0.8), max(4, actual_docs * 0.5)))
        sns.heatmap(
            sliced_matrix,
            annot=True,
            fmt="d",
            cmap="YlGnBu",
            xticklabels=sliced_terms,
            yticklabels=doc_labels,
            ax=ax
        )
        ax.set_title(f"DTM Slice ({actual_docs} Docs x {actual_terms} Terms)")
        ax.set_xlabel("Terms")
        ax.set_ylabel("Documents")
        plt.tight_layout()

        session.pending_figure = fig

        result_id = session.next_result_id("dtm_heatmap")
        summary = {
            "result_id": result_id,
            "n_documents": actual_docs,
            "n_terms": actual_terms,
            "terms": sliced_terms,
            "matrix": sliced_matrix.tolist()
        }

        session.store_result("dtm_heatmap_tool", {"n_terms": n_terms, "n_documents": n_documents}, summary)
        return summary

    return [build_dtm_tool, term_frequency_tool, dtm_heatmap_tool]
