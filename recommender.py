import requests
from sklearn.metrics.pairwise import cosine_similarity


def get_poster_url(movie_title, api_key):
    if not api_key or api_key == "YOUR_TMDB_API_KEY_HERE":
        return "https://via.placeholder.com/150x220?text=No+Image"

    try:
        url = f"https://api.themoviedb.org/3/search/movie?api_key={api_key}&query={movie_title}"
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        data = response.json()

        if data.get("results"):
            poster_path = data["results"][0].get("poster_path")
            if poster_path:
                return f"https://image.tmdb.org/t/p/w500{poster_path}"

    except Exception:
        pass

    return "https://via.placeholder.com/150x220?text=No+Image"


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


def get_hybrid_scores(idx, similarity_tfidf, similarity_count, tfidf_weight, count_weight):
    hybrid_scores = []

    for i in range(len(similarity_tfidf[idx])):
        score = (
            tfidf_weight * similarity_tfidf[idx][i] +
            count_weight * similarity_count[idx][i]
        )
        hybrid_scores.append((i, score))

    return hybrid_scores


def build_reason_tags(query_movie, recommended_movie):
    reasons = []

    query_genres = set(str(query_movie.get("genres", "")).lower().split())
    rec_genres = set(str(recommended_movie.get("genres", "")).lower().split())

    query_director = str(query_movie.get("crew", "")).strip().lower()
    rec_director = str(recommended_movie.get("crew", "")).strip().lower()

    query_title = str(query_movie.get("title", "")).strip().lower()
    rec_title = str(recommended_movie.get("title", "")).strip().lower()

    if query_genres and rec_genres and len(query_genres & rec_genres) > 0:
        reasons.append("Similar Genre")

    if query_director and query_director == rec_director:
        reasons.append("Same Director")

    query_words = query_title.split()
    rec_words = rec_title.split()

    if query_words and rec_words:
        shared_title_words = set(query_words) & set(rec_words)
        if len(shared_title_words) >= 2:
            reasons.append("Same Franchise")

    if not reasons:
        reasons.append("Similar to your search")

    return reasons[:2]


def get_recommendations(title, movies, similarity_tfidf, similarity_count, favorites, api_key, top_n=8):
    if not title or not title.strip():
        return []

    idx = find_best_match(title, movies)
    if idx is None:
        return []

    query_movie = movies.iloc[idx]
    tfidf_weight, count_weight = get_dynamic_weights(query_movie)

    hybrid_scores = get_hybrid_scores(
        idx,
        similarity_tfidf,
        similarity_count,
        tfidf_weight=tfidf_weight,
        count_weight=count_weight
    )

    sorted_scores = sorted(hybrid_scores, key=lambda x: x[1], reverse=True)
    sorted_scores = [s for s in sorted_scores if s[0] != idx][:top_n]

    favorite_titles_lower = [f.lower() for f in favorites]
    results = []

    for i in sorted_scores:
        movie = movies.iloc[i[0]]
        reason_tags = build_reason_tags(query_movie, movie)

        results.append({
            "title": movie["title"],
            "genre": movie["genres"],
            "description": movie["overview"],
            "director": movie["crew"],
            "poster": get_poster_url(movie["title"], api_key),
            "similarity": round(min(i[1] * 100, 100), 1),
            "trailer": f"https://www.youtube.com/results?search_query={movie['title'].replace(' ', '+')}+trailer",
            "is_favorite": movie["title"].lower() in favorite_titles_lower,
            "tfidf_weight": tfidf_weight,
            "count_weight": count_weight,
            "why": reason_tags
        })

    return results


def get_user_based_recommendations(
    favorites,
    movies,
    tfidf_matrix,
    count_matrix,
    api_key,
    top_n=8
):
    if not favorites:
        return []

    favorite_indices = []

    for title in favorites:
        idx = find_best_match(title, movies)
        if idx is not None:
            favorite_indices.append(idx)

    if not favorite_indices:
        return []

    tfidf_vectors = tfidf_matrix[favorite_indices]
    count_vectors = count_matrix[favorite_indices]

    user_tfidf_vector = tfidf_vectors.mean(axis=0)
    user_count_vector = count_vectors.mean(axis=0)

    user_tfidf_vector = user_tfidf_vector.A
    user_count_vector = user_count_vector.A

    tfidf_sim = cosine_similarity(user_tfidf_vector, tfidf_matrix).flatten()
    count_sim = cosine_similarity(user_count_vector, count_matrix).flatten()

    tfidf_weight = 0.6
    count_weight = 0.4

    hybrid_scores = (
        tfidf_weight * tfidf_sim +
        count_weight * count_sim
    )

    scored_movies = list(enumerate(hybrid_scores))
    scored_movies = sorted(scored_movies, key=lambda x: x[1], reverse=True)

    favorite_titles_lower = [f.lower() for f in favorites]
    favorite_index_set = set(favorite_indices)

    results = []

    for i, score in scored_movies:
        if i in favorite_index_set:
            continue

        movie = movies.iloc[i]

        similarity_score = float(round(min(float(score) * 100, 100), 1))

        results.append({
            "title": str(movie["title"]),
            "genre": str(movie["genres"]),
            "description": str(movie["overview"]),
            "director": str(movie["crew"]),
            "poster": str(get_poster_url(movie["title"], api_key)),
            "similarity": similarity_score,
            "trailer": f"https://www.youtube.com/results?search_query={str(movie['title']).replace(' ', '+')}+trailer",
            "is_favorite": str(movie["title"]).lower() in favorite_titles_lower,
            "why": ["Based on Favorites"]
        })

        if len(results) >= top_n:
            break

    return results