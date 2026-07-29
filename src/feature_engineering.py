def create_features(movies):

    movies["tags"] = (
        movies["overview"]
        + movies["genres"]
        + movies["keywords"]
        + movies["cast"]
        + movies["crew"]
    )

    new_df = movies[["movie_id", "title", "tags"]]

    new_df["tags"] = new_df["tags"].apply(lambda x: " ".join(x))

    new_df["tags"] = new_df["tags"].apply(lambda x: x.lower())

    return new_df