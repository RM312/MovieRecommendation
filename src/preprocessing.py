import ast


def convert(obj):
    lst = []

    for i in ast.literal_eval(obj):
        lst.append(i["name"])

    return lst


def convert3(obj):
    lst = []

    counter = 0

    for i in ast.literal_eval(obj):
        if counter != 3:
            lst.append(i["name"])
            counter += 1
        else:
            break

    return lst


def fetch_director(obj):
    lst = []

    for i in ast.literal_eval(obj):
        if i["job"] == "Director":
            lst.append(i["name"])
            break

    return lst


def preprocess(movies):

    movies["genres"] = movies["genres"].apply(convert)
    movies["keywords"] = movies["keywords"].apply(convert)
    movies["cast"] = movies["cast"].apply(convert3)
    movies["crew"] = movies["crew"].apply(fetch_director)

    movies["overview"] = movies["overview"].apply(lambda x: x.split())

    for col in ["genres", "keywords", "cast", "crew"]:
        movies[col] = movies[col].apply(
            lambda x: [i.replace(" ", "") for i in x]
        )

    return movies