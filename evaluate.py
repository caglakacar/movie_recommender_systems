import os
import pickle
import numpy as np


ARTIFACTS_DIR = "artifacts"
MOVIES_PATH = os.path.join(ARTIFACTS_DIR, "movies.pkl")
TFIDF_SIMILARITY_PATH = os.path.join(ARTIFACTS_DIR, "similarity_tfidf.npy")
COUNT_SIMILARITY_PATH = os.path.join(ARTIFACTS_DIR, "similarity_count.npy")


TEST_MOVIES = [
    "Inception",
    "The Dark Knight",
    "Avatar",
    "Titanic",
    "Interstellar",
    "The Matrix",
    "Gladiator",
    "Shutter Island"
]


def load_artifacts():
    with open(MOVIES_PATH, "rb") as f:
        movies = pickle.load(f)

    tfidf_similarity = np.load(TFIDF_SIMILARITY_PATH)
    count_similarity = np.load(COUNT_SIMILARITY_PATH)

    return movies, tfidf_similarity, count_similarity


def find_best_match(title, movies):
    title = title.strip().lower()
    all_titles = movies["title"].str.lower()

    exact_matches = movies[all_titles == title]
    if not exact_matches.empty:
        return exact_matches.index[0]

    partial_matches = movies[all_titles.str.contains(title, na=False)]
    if not partial_matches.empty:
        return partial_matches.index[0]

    return None


def get_dynamic_weights(movie_row):
    overview = str(movie_row.get("overview", "")).strip()
    genres = str(movie_row.get("genres", "")).strip()
    cast = str(movie_row.get("cast", "")).strip()
    crew = str(movie_row.get("crew", "")).strip()

    overview_word_count = len(overview.split())

    metadata_score = 0
    if genres:
        metadata_score += 1
    if cast:
        metadata_score += 1
    if crew:
        metadata_score += 1

    if overview_word_count >= 80 and metadata_score == 3:
        return 0.75, 0.25
    elif overview_word_count >= 40 and metadata_score >= 2:
        return 0.60, 0.40
    elif overview_word_count >= 20:
        return 0.50, 0.50
    else:
        return 0.30, 0.70


def get_top_recommendation_indices(idx, similarity_matrix, top_n=5):
    scores = list(enumerate(similarity_matrix[idx]))
    sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)
    sorted_scores = [s for s in sorted_scores if s[0] != idx][:top_n]
    return [i[0] for i in sorted_scores]


def get_top_hybrid_recommendation_indices(
    idx,
    similarity_tfidf,
    similarity_count,
    tfidf_weight,
    count_weight,
    top_n=5
):
    hybrid_scores = []

    for i in range(len(similarity_tfidf[idx])):
        final_score = (
            tfidf_weight * similarity_tfidf[idx][i] +
            count_weight * similarity_count[idx][i]
        )
        hybrid_scores.append((i, final_score))

    sorted_scores = sorted(hybrid_scores, key=lambda x: x[1], reverse=True)
    sorted_scores = [s for s in sorted_scores if s[0] != idx][:top_n]

    return [i[0] for i in sorted_scores]


def safe_split(text):
    if not isinstance(text, str) or not text.strip():
        return set()
    return set(text.lower().split())


def score_recommendation(query_movie, recommended_movie):
    """
    Weighted content-based evaluation score:
    - Genre overlap: 50%
    - Director match: 30%
    - Cast overlap: 20%
    """
    query_genres = safe_split(query_movie.get("genres", ""))
    rec_genres = safe_split(recommended_movie.get("genres", ""))

    query_cast = safe_split(query_movie.get("cast", ""))
    rec_cast = safe_split(recommended_movie.get("cast", ""))

    query_director = str(query_movie.get("crew", "")).strip().lower()
    rec_director = str(recommended_movie.get("crew", "")).strip().lower()

    if query_genres or rec_genres:
        genre_score = len(query_genres & rec_genres) / len(query_genres | rec_genres)
    else:
        genre_score = 0.0

    director_score = 1.0 if query_director and query_director == rec_director else 0.0

    if query_cast or rec_cast:
        cast_score = len(query_cast & rec_cast) / len(query_cast | rec_cast)
    else:
        cast_score = 0.0

    final_score = (
        0.5 * genre_score +
        0.3 * director_score +
        0.2 * cast_score
    )

    return round(final_score, 4)


def evaluate_model(movies, similarity_matrix, test_titles, model_name, top_n=5):
    movie_scores = []
    detailed_results = []

    for title in test_titles:
        idx = find_best_match(title, movies)
        if idx is None:
            print(f"[{model_name}] Movie not found: {title}")
            continue

        query_movie = movies.iloc[idx]
        rec_indices = get_top_recommendation_indices(idx, similarity_matrix, top_n=top_n)

        rec_scores = []
        rec_titles = []

        for rec_idx in rec_indices:
            recommended_movie = movies.iloc[rec_idx]
            score = score_recommendation(query_movie, recommended_movie)
            rec_scores.append(score)
            rec_titles.append(recommended_movie["title"])

        avg_score = round(sum(rec_scores) / len(rec_scores), 4) if rec_scores else 0.0
        movie_scores.append(avg_score)

        detailed_results.append({
            "query": title,
            "recommended_titles": rec_titles,
            "recommendation_scores": rec_scores,
            "average_score": avg_score
        })

    overall_score = round(sum(movie_scores) / len(movie_scores), 4) if movie_scores else 0.0

    return overall_score, detailed_results


def evaluate_hybrid_model(
    movies,
    similarity_tfidf,
    similarity_count,
    test_titles,
    model_name="Hybrid Dynamic",
    top_n=5
):
    movie_scores = []
    detailed_results = []

    for title in test_titles:
        idx = find_best_match(title, movies)
        if idx is None:
            print(f"[{model_name}] Movie not found: {title}")
            continue

        query_movie = movies.iloc[idx]
        tfidf_weight, count_weight = get_dynamic_weights(query_movie)

        rec_indices = get_top_hybrid_recommendation_indices(
            idx,
            similarity_tfidf,
            similarity_count,
            tfidf_weight=tfidf_weight,
            count_weight=count_weight,
            top_n=top_n
        )

        rec_scores = []
        rec_titles = []

        for rec_idx in rec_indices:
            recommended_movie = movies.iloc[rec_idx]
            score = score_recommendation(query_movie, recommended_movie)
            rec_scores.append(score)
            rec_titles.append(recommended_movie["title"])

        avg_score = round(sum(rec_scores) / len(rec_scores), 4) if rec_scores else 0.0
        movie_scores.append(avg_score)

        detailed_results.append({
            "query": title,
            "tfidf_weight": tfidf_weight,
            "count_weight": count_weight,
            "recommended_titles": rec_titles,
            "recommendation_scores": rec_scores,
            "average_score": avg_score
        })

    overall_score = round(sum(movie_scores) / len(movie_scores), 4) if movie_scores else 0.0

    return overall_score, detailed_results


def print_results(model_name, overall_score, detailed_results):
    print("=" * 80)
    print(f"{model_name} RESULTS")
    print("=" * 80)
    print(f"Overall Average Score: {overall_score}")
    print()

    for result in detailed_results:
        print(f"Query Movie: {result['query']}")

        if "tfidf_weight" in result and "count_weight" in result:
            print(f"  Dynamic Weights -> TF-IDF: {result['tfidf_weight']}, Count: {result['count_weight']}")

        for title, score in zip(result["recommended_titles"], result["recommendation_scores"]):
            print(f"  - {title:<35} Score: {score}")
        print(f"  Average for this movie: {result['average_score']}")
        print("-" * 80)


if __name__ == "__main__":
    movies, tfidf_similarity, count_similarity = load_artifacts()

    tfidf_overall, tfidf_details = evaluate_model(
        movies, tfidf_similarity, TEST_MOVIES, "TF-IDF", top_n=5
    )

    count_overall, count_details = evaluate_model(
        movies, count_similarity, TEST_MOVIES, "CountVectorizer", top_n=5
    )

    hybrid_overall, hybrid_details = evaluate_hybrid_model(
        movies,
        tfidf_similarity,
        count_similarity,
        TEST_MOVIES,
        model_name="Hybrid Dynamic",
        top_n=5
    )

    print_results("TF-IDF", tfidf_overall, tfidf_details)
    print_results("CountVectorizer", count_overall, count_details)
    print_results("Hybrid Dynamic", hybrid_overall, hybrid_details)

    print("\n" + "=" * 80)
    print("FINAL COMPARISON")
    print("=" * 80)
    print(f"TF-IDF Overall Score         : {tfidf_overall}")
    print(f"CountVectorizer Overall Score: {count_overall}")
    print(f"Hybrid Dynamic Overall Score : {hybrid_overall}")

    best_score = max(tfidf_overall, count_overall, hybrid_overall)

    if best_score == hybrid_overall:
        print("Winner: Hybrid Dynamic")
    elif best_score == count_overall:
        print("Winner: CountVectorizer")
    else:
        print("Winner: TF-IDF")