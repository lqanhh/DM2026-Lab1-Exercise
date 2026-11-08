import os
import nltk
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List, Optional
from langchain_core.tools import tool
from agent_pipeline.session_state import SessionState

# Ensure punkt is available for word_tokenize
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab', quiet=True)


def make_tools(session: SessionState) -> List[Any]:
    """Factory function returning tools for tools_data.py bound to the provided session."""

    @tool
    def tokenize_tool() -> Dict[str, Any]:
        """Tokenizes text in session.dataframe into unigrams (lowercased) using NLTK word_tokenize.

        Reads session.dataframe["text"], tokenizes each non-empty text string into a list of lowercase word
        tokens using nltk.word_tokenize, and stores the resulting token lists in session.dataframe["unigrams"]
        and session.artifacts["unigrams"].

        Returns:
            Dict containing result_id, n_documents, and sample_tokens.
        """
        if session.dataframe is None:
            return {"error": "No dataframe loaded in session. Please load a dataset first using load_dataset_tool."}

        if "text" not in session.dataframe.columns:
            return {"error": "'text' column not found in session.dataframe."}

        unigrams_list: List[List[str]] = []
        for text_val in session.dataframe["text"]:
            if isinstance(text_val, str) and text_val.strip():
                tokens = nltk.word_tokenize(text_val.lower())
            else:
                tokens = []
            unigrams_list.append(tokens)

        session.dataframe["unigrams"] = unigrams_list
        session.artifacts["unigrams"] = unigrams_list

        result_id = session.next_result_id("tokenize")

        total_tokens = sum(len(toks) for toks in unigrams_list)
        sample_preview = {
            f"doc_{idx}": toks[:5] for idx, toks in enumerate(unigrams_list[:3])
        }

        summary = {
            "result_id": result_id,
            "status": "success",
            "n_documents": len(session.dataframe),
            "total_tokens": total_tokens,
            "sample_tokens": sample_preview,
        }

        session.store_result("tokenize_tool", {}, summary)
        return summary

    @tool
    def list_files_tool(subdirectory: str = ".") -> Dict[str, Any]:
        """Lists files and directories in a given subdirectory to discover dataset paths.

        Args:
            subdirectory: Path relative to repository root to inspect (default ".").

        Returns:
            Dict containing result_id, subdirectory, directories list, and files list.
        """
        target_dir = subdirectory if subdirectory else "."

        if not os.path.exists(target_dir):
            return {"error": f"Directory '{subdirectory}' does not exist."}

        if not os.path.isdir(target_dir):
            return {"error": f"Path '{subdirectory}' is not a directory."}

        directories: List[str] = []
        files: List[str] = []

        for entry in os.scandir(target_dir):
            if entry.is_dir():
                directories.append(entry.name)
            elif entry.is_file():
                files.append(entry.name)

        directories.sort()
        files.sort()

        result_id = session.next_result_id("list_files")
        summary = {
            "result_id": result_id,
            "status": "success",
            "subdirectory": subdirectory,
            "directories": directories,
            "files": files,
        }

        session.store_result("list_files_tool", {"subdirectory": subdirectory}, summary)
        return summary

    @tool
    def inspect_data_tool(n: int = 5) -> Dict[str, Any]:
        """Inspects the structure and first n rows of the loaded DataFrame.

        Args:
            n: Number of rows to inspect (default 5). If n exceeds available rows, all rows are returned.

        Returns:
            Dict containing result_id, dimensions, columns, dtypes, inspected_rows, and sample_data.
        """
        if session.dataframe is None or session.dataframe.empty:
            result_id = session.next_result_id("inspect")
            summary = {
                "result_id": result_id,
                "status": "error",
                "message": "No dataframe loaded in session. Please load a dataset first using load_dataset_tool.",
            }
            session.store_result("inspect_data_tool", {"n": n}, summary)
            return summary

        df = session.dataframe
        n_rows, n_cols = df.shape
        preview_df = df.head(n)

        result_id = session.next_result_id("inspect")
        summary = {
            "result_id": result_id,
            "status": "success",
            "dimensions": [n_rows, n_cols],
            "columns": list(df.columns),
            "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
            "inspected_rows": len(preview_df),
            "sample_data": preview_df.to_dict(orient="records"),
        }

        session.store_result("inspect_data_tool", {"n": n}, summary)
        return summary

    @tool
    def check_missing_tool() -> Dict[str, Any]:
        """Checks for missing or empty text values in session.dataframe["text"].

        Returns:
            Dict containing result_id, total_rows, n_missing, and missing_indices.
        """
        if session.dataframe is None:
            return {"error": "No dataframe loaded in session. Please load a dataset first using load_dataset_tool."}

        if "text" not in session.dataframe.columns:
            return {"error": "'text' column not found in session.dataframe."}

        df = session.dataframe
        is_missing = df["text"].isna() | (df["text"].astype(str).str.strip() == "")
        missing_indices = df.index[is_missing].tolist()

        result_id = session.next_result_id("check_missing")
        summary = {
            "result_id": result_id,
            "status": "success",
            "total_rows": len(df),
            "n_missing": len(missing_indices),
            "missing_indices": missing_indices,
        }

        session.store_result("check_missing_tool", {}, summary)
        return summary

    @tool
    def check_duplicates_tool(drop: bool = False) -> Dict[str, Any]:
        """Checks for duplicate documents in session.dataframe and optionally drops them using keep=False.

        Args:
            drop: If True, drops all copies of duplicated rows (keep=False) in session.dataframe in place.

        Returns:
            Dict containing result_id, n_duplicates, duplicate_indices, dropped, and remaining_rows.
        """
        if session.dataframe is None:
            return {"error": "No dataframe loaded in session. Please load a dataset first using load_dataset_tool."}

        df = session.dataframe

        # Duplicate check on text column if available, else full dataframe
        subset = ["text"] if "text" in df.columns else None

        # pandas duplicated with keep=False flags every copy of a duplicate row
        dup_mask = df.duplicated(subset=subset, keep=False)
        duplicate_indices = df.index[dup_mask].tolist()
        n_duplicates = len(duplicate_indices)

        if drop and n_duplicates > 0:
            df_cleaned = df.drop_duplicates(subset=subset, keep=False).reset_index(drop=True)
            session.dataframe = df_cleaned
            dropped = True
        else:
            dropped = False

        result_id = session.next_result_id("check_duplicates")
        summary = {
            "result_id": result_id,
            "status": "success",
            "n_duplicates": n_duplicates,
            "duplicate_indices": duplicate_indices,
            "dropped": dropped,
            "remaining_rows": len(session.dataframe),
        }

        session.store_result("check_duplicates_tool", {"drop": drop}, summary)
        return summary

    @tool
    def sample_data_tool(n: int = 20, random_state: Optional[int] = 42) -> Dict[str, Any]:
        """Subsamples rows from session.dataframe using pandas sample with a random seed.

        Args:
            n: Number of rows to sample (default 20).
            random_state: Seed for random number generator (default 42).

        Returns:
            Dict containing result_id, n_requested, n_sampled, sample_indices, and sampled_rows.
        """
        if session.dataframe is None:
            return {"error": "No dataframe loaded in session. Please load a dataset first using load_dataset_tool."}

        df = session.dataframe
        actual_n = min(n, len(df))

        sampled_df = df.sample(n=actual_n, random_state=random_state)
        sample_indices = sampled_df.index.tolist()

        result_id = session.next_result_id("sample_data")
        summary = {
            "result_id": result_id,
            "status": "success",
            "n_requested": n,
            "n_sampled": actual_n,
            "sample_indices": sample_indices,
            "sampled_rows": sampled_df.to_dict(orient="records"),
        }

        session.store_result("sample_data_tool", {"n": n, "random_state": random_state}, summary)
        return summary

    @tool
    def describe_data_tool() -> Dict[str, Any]:
        """Computes summary statistics on text length and builds a box plot grouped by category.

        Computes text character length for session.dataframe, calculates overall and per-category
        describe() statistics, builds a box plot set on session.pending_figure, and returns the stats.

        Returns:
            Dict containing result_id, overall_stats, and per_category_stats.
        """
        if session.dataframe is None:
            return {"error": "No dataframe loaded in session. Please load a dataset first using load_dataset_tool."}

        if "text" not in session.dataframe.columns:
            return {"error": "'text' column not found in session.dataframe."}

        df = session.dataframe.copy()
        df["text_length"] = df["text"].apply(lambda x: len(x) if isinstance(x, str) else 0)

        overall_stats = df["text_length"].describe().to_dict()

        cat_col = "category_name" if "category_name" in df.columns else ("category" if "category" in df.columns else None)

        per_category_stats: Dict[str, Dict[str, float]] = {}
        if cat_col:
            for cat_val, group in df.groupby(cat_col):
                per_category_stats[str(cat_val)] = group["text_length"].describe().to_dict()

        # Build box plot
        fig, ax = plt.subplots(figsize=(6, 4))
        if cat_col:
            sns.boxplot(data=df, x=cat_col, y="text_length", ax=ax)
            ax.set_title("Text Length Distribution by Category")
            ax.set_xlabel("Category")
        else:
            sns.boxplot(data=df, y="text_length", ax=ax)
            ax.set_title("Text Length Distribution")

        ax.set_ylabel("Text Length (characters)")
        plt.tight_layout()

        session.pending_figure = fig

        result_id = session.next_result_id("describe_data")
        summary = {
            "result_id": result_id,
            "status": "success",
            "overall_stats": overall_stats,
            "per_category_stats": per_category_stats,
        }

        session.store_result("describe_data_tool", {}, summary)
        return summary

    return [tokenize_tool, list_files_tool, inspect_data_tool, check_missing_tool, check_duplicates_tool, sample_data_tool, describe_data_tool]
