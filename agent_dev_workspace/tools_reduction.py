import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Literal
from langchain_core.tools import tool
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
from agent_pipeline.session_state import SessionState


def make_tools(session: SessionState) -> List[Any]:
    """Factory function returning tools for tools_reduction.py bound to the provided session."""

    @tool
    def reduce_dimensions_tool(
        method: Literal["pca", "tsne", "umap"] = "pca",
        n_components: int = 2,
        perplexity: float = 30.0,
        n_neighbors: int = 15,
        random_state: Optional[int] = 42,
    ) -> Dict[str, Any]:
        """Reduces the dimensionality of session.feature_matrix using PCA, t-SNE, or UMAP, and plots a scatter plot.

        Args:
            method: Method to use for reduction ("pca", "tsne", or "umap"). Defaults to "pca".
            n_components: Target number of dimensions (default 2).
            perplexity: Perplexity parameter for t-SNE (default 30.0).
            n_neighbors: Number of neighbors parameter for UMAP (default 15).
            random_state: Random state seed for reproducible reduction (default 42).

        Returns:
            Dict containing result_id, status, method, coordinates, and explained_variance_ratio (for PCA).
        """
        if session.feature_matrix is None:
            return {"error": "No feature_matrix found in session. Please build a document-term matrix first using build_dtm_tool."}

        # Convert sparse matrix to dense array for dimensional reduction
        X = session.feature_matrix.toarray() if hasattr(session.feature_matrix, "toarray") else np.asarray(session.feature_matrix)

        explained_variance_ratio: Optional[List[float]] = None

        if method == "pca":
            reducer = PCA(n_components=n_components, random_state=random_state)
            coords = reducer.fit_transform(X)
            explained_variance_ratio = reducer.explained_variance_ratio_.tolist()
        elif method == "tsne":
            reducer = TSNE(n_components=n_components, perplexity=perplexity, random_state=random_state)
            coords = reducer.fit_transform(X)
        elif method == "umap":
            reducer = umap.UMAP(n_components=n_components, n_neighbors=n_neighbors, random_state=random_state)
            coords = reducer.fit_transform(X)
        else:
            return {"error": f"Unsupported method '{method}'. Choose from 'pca', 'tsne', or 'umap'."}

        # Save reduced coordinates to artifacts
        session.artifacts[f"reduced_coords_{method}"] = coords

        # Generate scatter plot
        fig, ax = plt.subplots(figsize=(7, 5))
        
        # Color points by category label if available
        labels = None
        if session.labels is not None:
            labels = session.labels
        elif session.dataframe is not None:
            if "category_name" in session.dataframe.columns:
                labels = session.dataframe["category_name"].values
            elif "category" in session.dataframe.columns:
                labels = session.dataframe["category"].values

        if labels is not None:
            sns.scatterplot(x=coords[:, 0], y=coords[:, 1], hue=labels, palette="tab10", ax=ax, s=60)
            ax.legend(title="Category", bbox_to_anchor=(1.05, 1), loc="upper left")
        else:
            ax.scatter(coords[:, 0], coords[:, 1], color="blue", alpha=0.7)

        ax.set_title(f"2D Projection using {method.upper()}")
        ax.set_xlabel("Component 1")
        ax.set_ylabel("Component 2")
        plt.tight_layout()

        # Set figure on session for auto-rendering
        session.pending_figure = fig

        result_id = session.next_result_id("reduce_dimensions")
        summary: Dict[str, Any] = {
            "result_id": result_id,
            "status": "success",
            "method": method,
            "coordinates": coords.tolist(),
        }

        if explained_variance_ratio is not None:
            summary["explained_variance_ratio"] = explained_variance_ratio

        session.store_result("reduce_dimensions_tool", {
            "method": method,
            "n_components": n_components,
            "perplexity": perplexity,
            "n_neighbors": n_neighbors,
            "random_state": random_state,
        }, summary)

        return summary

    @tool
    def binarize_labels_tool() -> Dict[str, Any]:
        """Converts category labels into a binary (one-hot) encoded 2D array representation.

        Reads labels from session.labels or session.dataframe["category_name"], encodes them into
        a binary matrix of shape (n_documents, n_categories), stores it in session.artifacts["binarized_labels"],
        and returns the encoded matrix and metadata. Handles missing/null category values as all-zero rows.

        Returns:
            Dict containing result_id, status, categories, shape, and binary_matrix.
        """
        raw_labels = None
        if session.labels is not None:
            raw_labels = session.labels
        elif session.dataframe is not None:
            if "category_name" in session.dataframe.columns:
                raw_labels = session.dataframe["category_name"].values
            elif "category" in session.dataframe.columns:
                raw_labels = session.dataframe["category"].values

        if raw_labels is None or len(raw_labels) == 0:
            result_id = session.next_result_id("binarize_labels")
            summary = {
                "result_id": result_id,
                "status": "error",
                "message": "No category labels found in session. Please load a dataset first.",
            }
            session.store_result("binarize_labels_tool", {}, summary)
            return summary

        # Determine categories (sorted unique non-null categories)
        if session.categories:
            categories = sorted([str(c) for c in session.categories if pd.notna(c) and str(c).strip() != ""])
        else:
            categories = sorted(list({str(lbl) for lbl in raw_labels if pd.notna(lbl) and str(lbl).strip() != ""}))

        n_docs = len(raw_labels)
        n_cats = len(categories)

        # Build binary matrix (n_docs x n_cats)
        binary_matrix = np.zeros((n_docs, n_cats), dtype=int)
        cat_to_idx = {cat: idx for idx, cat in enumerate(categories)}

        for i, lbl in enumerate(raw_labels):
            if pd.notna(lbl):
                lbl_str = str(lbl)
                if lbl_str in cat_to_idx:
                    binary_matrix[i, cat_to_idx[lbl_str]] = 1

        # Store in session artifacts
        session.artifacts["binarized_labels"] = binary_matrix

        result_id = session.next_result_id("binarize_labels")
        summary = {
            "result_id": result_id,
            "status": "success",
            "categories": categories,
            "shape": [n_docs, n_cats],
            "binary_matrix": binary_matrix.tolist(),
        }

        session.store_result("binarize_labels_tool", {}, summary)
        return summary

    return [reduce_dimensions_tool, binarize_labels_tool]
