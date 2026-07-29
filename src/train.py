import pickle

from data_loader import load_data
from preprocessing import preprocess
from feature_engineering import create_features
from vectorizer import build_similarity


def main():

    movies = load_data(
        "data/tmdb_5000_movies.csv",
        "data/tmdb_5000_credits.csv",
    )

    movies = preprocess(movies)

    new_df = create_features(movies)

    new_df, similarity = build_similarity(new_df)

    pickle.dump(
        new_df,
        open("models/movies.pkl", "wb"),
    )

    pickle.dump(
        similarity,
        open("models/similarity.pkl", "wb"),
    )

    # Optional
    pickle.dump(
        new_df.to_dict(),
        open("models/movie_dict.pkl", "wb"),
    )


if __name__ == "__main__":
    main()