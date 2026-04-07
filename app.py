import os
import pickle
import numpy as np
import redis
from dotenv import load_dotenv
from flask_caching import Cache
from flask import Flask, render_template, request, jsonify, session, redirect, url_for
from scipy.sparse import load_npz
from recommender import (
    get_recommendations,
    get_user_based_recommendations,
    get_poster_url
)

load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "movie_recommender")


def configure_cache(flask_app: Flask) -> Cache:
    redis_host = os.getenv("CACHE_REDIS_HOST", "redis")
    redis_port = int(os.getenv("CACHE_REDIS_PORT", 6379))
    cache_timeout = 3600

    try:
        test_client = redis.Redis(
            host=redis_host,
            port=redis_port,
            socket_connect_timeout=1,
            socket_timeout=1
        )
        test_client.ping()

        flask_app.config["CACHE_TYPE"] = "RedisCache"
        flask_app.config["CACHE_REDIS_HOST"] = redis_host
        flask_app.config["CACHE_REDIS_PORT"] = redis_port
        flask_app.config["CACHE_DEFAULT_TIMEOUT"] = cache_timeout

        print(f"[Cache] Using RedisCache at {redis_host}:{redis_port}")
    except Exception:
        flask_app.config["CACHE_TYPE"] = "SimpleCache"
        flask_app.config["CACHE_DEFAULT_TIMEOUT"] = cache_timeout

        print("[Cache] Redis unavailable. Falling back to SimpleCache")

    return Cache(flask_app)


cache = configure_cache(app)

ARTIFACTS_DIR = "artifacts"
MOVIES_PATH = os.path.join(ARTIFACTS_DIR, "movies.pkl")
SIMILARITY_TFIDF_PATH = os.path.join(ARTIFACTS_DIR, "similarity_tfidf.npy")
SIMILARITY_COUNT_PATH = os.path.join(ARTIFACTS_DIR, "similarity_count.npy")
TFIDF_MATRIX_PATH = os.path.join(ARTIFACTS_DIR, "tfidf_matrix.npz")
COUNT_MATRIX_PATH = os.path.join(ARTIFACTS_DIR, "count_matrix.npz")

TMDB_API_KEY = os.getenv("TMDB_API_KEY")

DEFAULT_TOP_N = 8
TOP_N_STEP = 8


def load_artifacts():
    required_paths = [
        MOVIES_PATH,
        SIMILARITY_TFIDF_PATH,
        SIMILARITY_COUNT_PATH,
        TFIDF_MATRIX_PATH,
        COUNT_MATRIX_PATH
    ]

    for path in required_paths:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Artifact not found: {path}. Please run 'python train.py' first."
            )

    with open(MOVIES_PATH, "rb") as f:
        movies = pickle.load(f)

    similarity_tfidf = np.load(SIMILARITY_TFIDF_PATH)
    similarity_count = np.load(SIMILARITY_COUNT_PATH)

    tfidf_matrix = load_npz(TFIDF_MATRIX_PATH)
    count_matrix = load_npz(COUNT_MATRIX_PATH)

    return movies, similarity_tfidf, similarity_count, tfidf_matrix, count_matrix


movies, similarity_tfidf, similarity_count, tfidf_matrix, count_matrix = load_artifacts()


def parse_top_n(value, default=DEFAULT_TOP_N):
    try:
        parsed = int(value)
        return parsed if parsed > 0 else default
    except (TypeError, ValueError):
        return default


def build_favorites_key(favorites):
    return "|".join(sorted([f.lower() for f in favorites]))


@cache.memoize(timeout=1800)
def cached_recommendations(title, favorites_key, top_n):
    favorites = favorites_key.split("|") if favorites_key else []

    return get_recommendations(
        title=title,
        movies=movies,
        similarity_tfidf=similarity_tfidf,
        similarity_count=similarity_count,
        favorites=favorites,
        api_key=TMDB_API_KEY,
        top_n=top_n
    )


@cache.memoize(timeout=1800)
def cached_user_based_recommendations(favorites_key, top_n):
    favorites = favorites_key.split("|") if favorites_key else []

    return get_user_based_recommendations(
        favorites=favorites,
        movies=movies,
        tfidf_matrix=tfidf_matrix,
        count_matrix=count_matrix,
        api_key=TMDB_API_KEY,
        top_n=top_n
    )


@cache.memoize(timeout=3600)
def cached_autocomplete(query):
    if not query:
        return []

    matched_titles = movies[
        movies["title"].str.lower().str.contains(query, na=False)
    ]["title"].tolist()

    matched_titles = sorted(matched_titles, key=lambda x: x.lower() != query)
    return matched_titles[:8]


@cache.memoize(timeout=86400)
def cached_poster_url(movie_title):
    return get_poster_url(movie_title, TMDB_API_KEY)


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "cache": app.config["CACHE_TYPE"]
    })


@app.route("/", methods=["GET", "POST"])
def index():
    recommendations = []
    query = ""
    results_title = None
    error_message = None
    has_more_results = False
    show_more_url = None
    current_top_n = DEFAULT_TOP_N

    favorites = session.get("favorites", [])
    favorites_key = build_favorites_key(favorites)

    if request.method == "POST":
        query = request.form.get("movie", "").strip()
        current_top_n = DEFAULT_TOP_N

        if query:
            recommendations = cached_recommendations(query, favorites_key, current_top_n)

            if recommendations:
                results_title = f'Recommended for "{query}"'
                has_more_results = len(recommendations) == current_top_n
                if has_more_results:
                    show_more_url = url_for(
                        "index",
                        source="search",
                        query=query,
                        top_n=current_top_n + TOP_N_STEP
                    )
            else:
                error_message = "No matching movie found. Try another title."

    else:
        source = request.args.get("source", "").strip()
        query = request.args.get("query", "").strip()

        if source == "search" and query:
            current_top_n = parse_top_n(request.args.get("top_n"), DEFAULT_TOP_N)
            recommendations = cached_recommendations(query, favorites_key, current_top_n)

            if recommendations:
                results_title = f'Recommended for "{query}"'
                has_more_results = len(recommendations) == current_top_n
                if has_more_results:
                    show_more_url = url_for(
                        "index",
                        source="search",
                        query=query,
                        top_n=current_top_n + TOP_N_STEP
                    )
            else:
                error_message = "No matching movie found. Try another title."

    all_titles = sorted(movies["title"].dropna().unique().tolist())

    return render_template(
        "index.html",
        recommendations=recommendations,
        query=query,
        all_titles=all_titles,
        results_title=results_title,
        error_message=error_message,
        has_more_results=has_more_results,
        show_more_url=show_more_url,
        current_top_n=current_top_n
    )


@app.route("/autocomplete", methods=["GET"])
def autocomplete():
    query = request.args.get("q", "").lower().strip()
    suggestions = cached_autocomplete(query)
    return jsonify(suggestions)


@app.route("/recommend", methods=["GET"])
def recommend_api():
    title = request.args.get("movie", "").strip()
    top_n = parse_top_n(request.args.get("top_n"), DEFAULT_TOP_N)

    favorites = session.get("favorites", [])
    favorites_key = build_favorites_key(favorites)

    recommendations = cached_recommendations(title, favorites_key, top_n)
    return jsonify(recommendations)


@app.route("/favorites_recommendations", methods=["GET"])
def favorites_recommendations():
    favorites = session.get("favorites", [])

    if not favorites:
        return redirect(url_for("favorites", empty_recommendation="1"))

    top_n = parse_top_n(request.args.get("top_n"), DEFAULT_TOP_N)
    favorites_key = build_favorites_key(favorites)
    recommendations = cached_user_based_recommendations(favorites_key, top_n)

    all_titles = sorted(movies["title"].dropna().unique().tolist())
    has_more_results = len(recommendations) == top_n
    show_more_url = None

    if has_more_results:
        show_more_url = url_for(
            "favorites_recommendations",
            top_n=top_n + TOP_N_STEP
        )

    return render_template(
        "index.html",
        recommendations=recommendations,
        query="",
        all_titles=all_titles,
        results_title="Recommended Based on Your Favorites",
        error_message=None,
        has_more_results=has_more_results,
        show_more_url=show_more_url,
        current_top_n=top_n
    )


@app.route("/add_favorite", methods=["POST"])
def add_favorite():
    data = request.get_json(silent=True) or {}
    title = data.get("title")

    if not title:
        return jsonify({"status": "error", "message": "Title is required"}), 400

    favorites = session.get("favorites", [])

    if title.lower() not in [f.lower() for f in favorites]:
        favorites.append(title)
        session["favorites"] = favorites

    return jsonify({"status": "success"})


@app.route("/favorites")
def favorites():
    favorite_titles = session.get("favorites", [])
    favorite_movies = []
    empty_recommendation = request.args.get("empty_recommendation") == "1"

    for title in favorite_titles:
        movie_data = movies[movies["title"].str.lower() == title.lower()]

        if not movie_data.empty:
            movie = movie_data.iloc[0]
            favorite_movies.append({
                "title": movie["title"],
                "genre": movie["genres"],
                "description": movie["overview"],
                "director": movie["crew"],
                "poster": cached_poster_url(movie["title"]),
                "trailer": f"https://www.youtube.com/results?search_query={movie['title'].replace(' ', '+')}+trailer"
            })

    return render_template(
        "favorites.html",
        favorites=favorite_movies,
        empty_recommendation=empty_recommendation
    )


@app.route("/remove_favorite", methods=["POST"])
def remove_favorite():
    data = request.get_json(silent=True) or {}
    title = data.get("title")

    if not title:
        return jsonify({"status": "error", "message": "Title is required"}), 400

    favorites = session.get("favorites", [])

    for f in favorites:
        if f.lower() == title.lower():
            favorites.remove(f)
            session["favorites"] = favorites
            break

    return jsonify({"status": "success"})


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)