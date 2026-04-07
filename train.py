import os
import ast
import pickle
import numpy as np
import pandas as pd
import mlflow
from scipy.sparse import save_npz
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


DATA_DIR = "data"
MOVIES_CSV_PATH = os.path.join(DATA_DIR, "tmdb_5000_movies.csv")
CREDITS_CSV_PATH = os.path.join(DATA_DIR, "tmdb_5000_credits.csv")

ARTIFACTS_DIR = "artifacts"
MOVIES_OUTPUT_PATH = os.path.join(ARTIFACTS_DIR, "movies.pkl")

TFIDF_SIMILARITY_OUTPUT_PATH = os.path.join(ARTIFACTS_DIR, "similarity_tfidf.npy")
COUNT_SIMILARITY_OUTPUT_PATH = os.path.join(ARTIFACTS_DIR, "similarity_count.npy")

TFIDF_VECTORIZER_OUTPUT_PATH = os.path.join(ARTIFACTS_DIR, "vectorizer_tfidf.pkl")
COUNT_VECTORIZER_OUTPUT_PATH = os.path.join(ARTIFACTS_DIR, "vectorizer_count.pkl")

TFIDF_MATRIX_OUTPUT_PATH = os.path.join(ARTIFACTS_DIR, "tfidf_matrix.npz")
COUNT_MATRIX_OUTPUT_PATH = os.path.join(ARTIFACTS_DIR, "count_matrix.npz")


os.makedirs("mlruns", exist_ok=True)
mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("movie-recommender")


def convert(obj):
    try:
        return " ".join([i["name"] for i in ast.literal_eval(obj)][:3])
    except Exception:
        return ""


def get_director(obj):
    try:
        for i in ast.literal_eval(obj):
            if i["job"] == "Director":
                return i["name"]
        return ""
    except Exception:
        return ""


def build_dataset():
    movies = pd.read_csv(MOVIES_CSV_PATH)
    credits = pd.read_csv(CREDITS_CSV_PATH)

    movies = movies.merge(credits, on="title")

    movies["genres"] = movies["genres"].apply(convert)
    movies["cast"] = movies["cast"].apply(convert)
    movies["crew"] = movies["crew"].apply(get_director)

    movies["overview"] = movies["overview"].fillna("")
    movies["genres"] = movies["genres"].fillna("")
    movies["cast"] = movies["cast"].fillna("")
    movies["crew"] = movies["crew"].fillna("")

    movies["combined"] = (
        movies["genres"] + " " + movies["genres"] + " " + movies["genres"] + " " +
        movies["cast"] + " " + movies["cast"] + " " + movies["cast"] + " " +
        movies["crew"] + " " + movies["crew"] + " " +
        movies["overview"]
    )

    return movies


def train_tfidf_model(movies):
    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 3),
        min_df=1,
        max_features=5000
    )

    matrix = vectorizer.fit_transform(movies["combined"])
    similarity = cosine_similarity(matrix)

    return vectorizer, matrix, similarity


def train_count_model(movies):
    vectorizer = CountVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        max_features=10000
    )

    matrix = vectorizer.fit_transform(movies["combined"])
    similarity = cosine_similarity(matrix)

    return vectorizer, matrix, similarity


def save_artifacts(
    movies,
    tfidf_vectorizer,
    tfidf_matrix,
    tfidf_similarity,
    count_vectorizer,
    count_matrix,
    count_similarity
):
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)

    with open(MOVIES_OUTPUT_PATH, "wb") as f:
        pickle.dump(movies, f)

    with open(TFIDF_VECTORIZER_OUTPUT_PATH, "wb") as f:
        pickle.dump(tfidf_vectorizer, f)

    with open(COUNT_VECTORIZER_OUTPUT_PATH, "wb") as f:
        pickle.dump(count_vectorizer, f)

    save_npz(TFIDF_MATRIX_OUTPUT_PATH, tfidf_matrix)
    save_npz(COUNT_MATRIX_OUTPUT_PATH, count_matrix)

    np.save(TFIDF_SIMILARITY_OUTPUT_PATH, tfidf_similarity)
    np.save(COUNT_SIMILARITY_OUTPUT_PATH, count_similarity)

    print("Artifacts saved successfully:")
    print(f"- {MOVIES_OUTPUT_PATH}")
    print(f"- {TFIDF_VECTORIZER_OUTPUT_PATH}")
    print(f"- {COUNT_VECTORIZER_OUTPUT_PATH}")
    print(f"- {TFIDF_MATRIX_OUTPUT_PATH}")
    print(f"- {COUNT_MATRIX_OUTPUT_PATH}")
    print(f"- {TFIDF_SIMILARITY_OUTPUT_PATH}")
    print(f"- {COUNT_SIMILARITY_OUTPUT_PATH}")


if __name__ == "__main__":
    from evaluate import evaluate_model, evaluate_hybrid_model, TEST_MOVIES

    with mlflow.start_run(run_name="hybrid-training"):
        movies_df = build_dataset()

        mlflow.log_param("dataset_size", len(movies_df))
        mlflow.log_param("tfidf_ngram_range", "(1,3)")
        mlflow.log_param("tfidf_min_df", 1)
        mlflow.log_param("tfidf_max_features", 10000)
        mlflow.log_param("count_ngram_range", "(1,2)")
        mlflow.log_param("count_max_features", 10000)

        tfidf_vectorizer, tfidf_matrix, tfidf_similarity = train_tfidf_model(movies_df)
        count_vectorizer, count_matrix, count_similarity = train_count_model(movies_df)

        save_artifacts(
            movies_df,
            tfidf_vectorizer,
            tfidf_matrix,
            tfidf_similarity,
            count_vectorizer,
            count_matrix,
            count_similarity
        )

        tfidf_overall, _ = evaluate_model(
            movies_df,
            tfidf_similarity,
            TEST_MOVIES,
            "TF-IDF",
            top_n=5
        )

        count_overall, _ = evaluate_model(
            movies_df,
            count_similarity,
            TEST_MOVIES,
            "CountVectorizer",
            top_n=5
        )

        hybrid_overall, _ = evaluate_hybrid_model(
            movies_df,
            tfidf_similarity,
            count_similarity,
            TEST_MOVIES,
            model_name="Hybrid Dynamic",
            top_n=5
        )

        mlflow.log_metric("tfidf_overall_score", tfidf_overall)
        mlflow.log_metric("count_overall_score", count_overall)
        mlflow.log_metric("hybrid_overall_score", hybrid_overall)

        mlflow.log_artifacts(ARTIFACTS_DIR)

        print("MLflow logging completed successfully.")