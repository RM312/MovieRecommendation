from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

ps = PorterStemmer()


def stem(text):

    words = []

    for word in text.split():
        words.append(ps.stem(word))

    return " ".join(words)


def build_similarity(new_df):

    new_df["tags"] = new_df["tags"].apply(stem)

    cv = CountVectorizer(
        max_features=5000,
        stop_words="english",
    )

    vectors = cv.fit_transform(new_df["tags"]).toarray()

    similarity = cosine_similarity(vectors)

    return new_df, similarity