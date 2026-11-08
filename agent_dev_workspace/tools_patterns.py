import os
import tempfile
from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd
from scipy.sparse import issparse
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from langchain_core.tools import tool
from agent_pipeline.session_state import SessionState

# Import PAMI algorithms
from PAMI.frequentPattern.basic import FPGrowth
from PAMI.frequentPattern.topk import FAE
from PAMI.frequentPattern.maximal import MaxFPGrowth

def make_tools(session: SessionState) -> List[Any]:
    """Factory function returning tools for tools_patterns.py bound to the provided session."""

    @tool
    def mine_patterns_tool(
        category_name: str,
        filtering_method: str = "tfidf",
        algorithm: str = "fpgrowth",
        min_sup: Optional[int] = None,
        k: Optional[int] = None,
        top_n: int = 10,
    ) -> Dict[str, Any]:
        """Mines frequent term patterns for a specific category using vocabulary filtering and PAMI.

        Filters session.dataframe for documents in category_name, builds a category-local DTM using
        CountVectorizer(stop_words='english'), filters terms using the chosen filtering_method
        ('variance', 'tfidf', or 'term_frequency'), converts to a transactional database, and mines
        frequent itemsets with PAMI using algorithm ('fpgrowth', 'topk', or 'maxfpgrowth').

        Args:
            category_name: Target category string in session.dataframe['category_name'].
            filtering_method: Method to filter vocabulary ('variance', 'tfidf', or 'term_frequency'). Default 'tfidf'.
            algorithm: Mining algorithm ('fpgrowth', 'topk', or 'maxfpgrowth'). Default 'fpgrowth'.
            min_sup: Minimum support threshold (required for 'fpgrowth' and 'maxfpgrowth').
            k: Top-K pattern count (required for 'topk').
            top_n: Maximum number of top patterns to include in the tool summary (default 10).

        Returns:
            Dict containing result_id, category_name, filtering_method, algorithm, kept_terms, patterns (top N), total_patterns_found, and optional note.
        """
        print(f"[mine_patterns_tool] Step 0: Initializing pattern mining for category='{category_name}', filtering='{filtering_method}', algo='{algorithm}'...")
        if session.dataframe is None or "category_name" not in session.dataframe.columns or "text" not in session.dataframe.columns:
            return {"error": "DataFrame with 'category_name' and 'text' columns is required. Please load dataset first."}

        # Filter dataframe for category
        cat_df = session.dataframe[session.dataframe["category_name"] == category_name]
        if cat_df.empty:
            return {"error": f"No documents found for category '{category_name}'."}

        # Non-empty text rows only
        texts = cat_df["text"].dropna().tolist()
        texts = [t for t in texts if str(t).strip()]
        if not texts:
            return {"error": f"No valid non-empty text found for category '{category_name}'."}

        # Step 1: Category-local CountVectorizer(stop_words="english")
        print(f"[mine_patterns_tool] Step 1: Building category-local DTM over {len(texts)} documents...")
        vectorizer = CountVectorizer(stop_words="english")
        try:
            dtm = vectorizer.fit_transform(texts)
            feature_names = np.array(vectorizer.get_feature_names_out())
        except ValueError:
            # Vectorizer raises ValueError if all words are stopwords or empty
            feature_names = np.array([])
            dtm = None

        if len(feature_names) == 0 or dtm is None or dtm.shape[1] == 0:
            print("[mine_patterns_tool] Step 1: No terms survived stopword removal or vectorization.")
            result_id = session.next_result_id("mine_patterns")
            summary = {
                "result_id": result_id,
                "category_name": category_name,
                "filtering_method": filtering_method,
                "algorithm": algorithm,
                "kept_terms": [],
                "patterns": [],
                "total_patterns_found": 0,
                "note": "no terms survived stopword removal or vectorization",
            }
            session.store_result("mine_patterns_tool", {"category_name": category_name, "filtering_method": filtering_method, "algorithm": algorithm}, summary)
            return summary

        # Step 2: Apply vocabulary filtering method using memory-efficient sparse matrix calculations
        print(f"[mine_patterns_tool] Step 2: Filtering {len(feature_names)} terms using method='{filtering_method}'...")
        if filtering_method == "variance":
            # Variance on sparse matrix: Var(X) = E[X^2] - (E[X])^2
            mean = np.array(dtm.mean(axis=0)).ravel()
            squared_mean = np.array(dtm.power(2).mean(axis=0)).ravel()
            scores = squared_mean - (mean ** 2)
            p5, p95 = np.percentile(scores, 5), np.percentile(scores, 95)
            keep_mask = (scores > p5) & (scores < p95)
        elif filtering_method == "term_frequency":
            scores = np.array(dtm.sum(axis=0)).ravel()
            p5, p95 = np.percentile(scores, 5), np.percentile(scores, 95)
            keep_mask = (scores > p5) & (scores < p95)
        elif filtering_method == "tfidf":
            tfidf = TfidfTransformer().fit_transform(dtm)
            scores = np.array(tfidf.mean(axis=0)).ravel()
            p20 = np.percentile(scores, 20)
            keep_mask = scores >= p20
        else:
            return {"error": f"Unknown filtering_method '{filtering_method}'. Expected 'variance', 'tfidf', or 'term_frequency'."}

        kept_terms = feature_names[keep_mask].tolist()
        filtered_dtm = dtm[:, keep_mask]

        if len(kept_terms) == 0:
            print("[mine_patterns_tool] Step 2: Zero terms survived filtering.")
            result_id = session.next_result_id("mine_patterns")
            summary = {
                "result_id": result_id,
                "category_name": category_name,
                "filtering_method": filtering_method,
                "algorithm": algorithm,
                "kept_terms": [],
                "patterns": [],
                "total_patterns_found": 0,
                "note": "no terms survived filtering",
            }
            session.store_result("mine_patterns_tool", {"category_name": category_name, "filtering_method": filtering_method, "algorithm": algorithm}, summary)
            return summary

        # Step 3: Write transactional DB directly from sparse matrix (avoiding dense matrix conversion)
        print(f"[mine_patterns_tool] Step 3: Writing transactional DB from sparse DTM ({len(kept_terms)} kept terms)...")
        csr_dtm = filtered_dtm.tocsr()

        with tempfile.TemporaryDirectory() as tmp_dir:
            db_file = os.path.join(tmp_dir, "transactional_db.txt")
            out_file = os.path.join(tmp_dir, "patterns_output.txt")

            with open(db_file, "w", encoding="utf-8") as f:
                for row_idx in range(csr_dtm.shape[0]):
                    start = csr_dtm.indptr[row_idx]
                    end = csr_dtm.indptr[row_idx + 1]
                    col_indices = csr_dtm.indices[start:end]
                    if len(col_indices) > 0:
                        row_terms = [kept_terms[c] for c in col_indices]
                        f.write("\t".join(row_terms) + "\n")

            # Step 4: Mine transactional DB using algorithm
            if algorithm == "fpgrowth":
                if min_sup is None:
                    return {"error": "min_sup parameter is required for algorithm='fpgrowth'."}
                miner = FPGrowth.FPGrowth(iFile=db_file, minSup=min_sup, sep="\t")
            elif algorithm == "topk":
                if k is None:
                    return {"error": "k parameter is required for algorithm='topk'."}
                miner = FAE.FAE(iFile=db_file, k=k, sep="\t")
            elif algorithm == "maxfpgrowth":
                if min_sup is None:
                    return {"error": "min_sup parameter is required for algorithm='maxfpgrowth'."}
                miner = MaxFPGrowth.MaxFPGrowth(iFile=db_file, minSup=min_sup, sep="\t")
            else:
                return {"error": f"Unknown algorithm '{algorithm}'. Expected 'fpgrowth', 'topk', or 'maxfpgrowth'."}

            print(f"[mine_patterns_tool] Step 4a: Mining patterns with algorithm='{algorithm}' (min_sup={min_sup}, k={k})...")
            miner.mine()
            print(f"[mine_patterns_tool] Step 4b: Saving patterns to output file...")
            miner.save(out_file)

            # Step 5: Read mined patterns from file
            print(f"[mine_patterns_tool] Step 5: Reading mined patterns output from file...")
            patterns = []
            if os.path.exists(out_file):
                with open(out_file, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        parts = line.split(":")
                        if len(parts) == 2:
                            pattern_str = parts[0].strip()
                            try:
                                sup = int(float(parts[1].strip()))
                            except ValueError:
                                sup = parts[1].strip()
                            patterns.append({"pattern": pattern_str, "support": sup})

        print(f"[mine_patterns_tool] Step 6: Storing result ({len(patterns)} patterns found)...")
        result_id = session.next_result_id("mine_patterns")
        
        # Save full patterns list to session artifacts
        artifact_key = f"patterns_{category_name}_{filtering_method}_{algorithm}"
        session.artifacts[artifact_key] = patterns

        # Summary returns only top N patterns to avoid large message payloads
        summary = {
            "result_id": result_id,
            "category_name": category_name,
            "filtering_method": filtering_method,
            "algorithm": algorithm,
            "kept_terms": kept_terms,
            "patterns": patterns[:top_n],
            "total_patterns_found": len(patterns),
        }

        full_report = {
            "patterns": patterns,
            "kept_terms": kept_terms,
        }

        session.store_result(
            "mine_patterns_tool",
            {"category_name": category_name, "filtering_method": filtering_method, "algorithm": algorithm, "min_sup": min_sup, "k": k, "top_n": top_n},
            summary,
            full_report=full_report,
        )

        print(f"[mine_patterns_tool] Completed successfully. Result ID: {result_id}")
        return summary

    return [mine_patterns_tool]
