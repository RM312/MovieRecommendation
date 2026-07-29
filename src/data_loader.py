import pandas as pd


def load_data(movie_path: str, credit_path: str):
    movies = pd.read_csv(movie_path)
    credits = pd.read_csv(credit_path)

    movies = movies.merge(credits, on="title")

    movies = movies[
        [
            "movie_id",
            "title",
            "overview",
            "genres",
            "keywords",
            "cast",
            "crew",
        ]
    ]

    movies.dropna(inplace=True)
    movies.drop_duplicates(inplace=True)

    return movies