def recommend(movie, movies, similarity):

    movie_index = movies[movies["title"] == movie].index[0]

    distances = similarity[movie_index]

    movies_list = sorted(
        list(enumerate(distances)),
        reverse=True,
        key=lambda x: x[1],
    )[1:6]

    recommendations = []

    for i in movies_list:
        recommendations.append(movies.iloc[i[0]].title)

    return recommendations