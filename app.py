import streamlit as st
import pickle
import pandas as pd
import requests

def fetch_poster(movie_id):
    response = requests.get(f'https://api.themoviedb.org/3/movie/{movie_id}?api_key=cafa370c077ac222520388da3fc6bd3d')
    data = response.json()
    return "https://image.tmdb.org/t/p/w500/" + data['poster_path'] if 'poster_path' in data else None

def recommend(movie):
    movie_index = movies[movies['title'] == movie].index[0]
    distances = similarity[movie_index]
    movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]

    recommend_movies=[]
    recommend_movies_poster=[]
    for i in movies_list:
        movie_id= movies.iloc[i[0]].movie_id
        recommend_movies.append(movies.iloc[i[0]].title)
        recommend_movies_poster.append(fetch_poster(movie_id))
    return recommend_movies, recommend_movies_poster

similarity = pickle.load(open('similarity.pkl','rb'))
movies_dict = pickle.load(open('movie_dict.pkl','rb'))
movies = pd.DataFrame(movies_dict)

st.title("Movie Recommender System")

selected_movie_name = st.selectbox(
    'Type or Select a Movie from the Drop Down',
    movies['title'].values
)
if st.button("Recommend"):
    names, poster = recommend(selected_movie_name)

    col = st.columns(5)

    for i in range(5):
        with col[i]:
            st.text(names[i])
            st.image(poster[i])