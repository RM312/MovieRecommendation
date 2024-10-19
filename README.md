Here is your updated README file with the link to access the project included:

---

# Movie Recommender System

## Overview

This project is a **Movie Recommender System** that processes data in a Jupyter Notebook and uses the processed data in a Python-based UI built with **Streamlit**. The recommendation system is based on **Cosine Similarity**, a technique commonly used to measure the similarity between two vectors (in this case, movie features). The workflow is split into two parts:

1. **Data Processing in Jupyter Notebook**: Data cleaning, processing, and model training are performed in the `MovieRecommender.ipynb`. Cosine similarity is used to calculate the similarity between movies based on their features. The results, including the recommendation model, are serialized into a binary file that the main program can load.
  
2. **User Interface using Streamlit**: The UI is developed using Streamlit (`app.py`), which provides an interactive interface where users can input their preferences and get movie recommendations.

You can access the live version of the project here: [Movie Recommender System](https://movierecommendation-wgbzf6btctioktz9zj9rjv.streamlit.app/)

---

## Dataset

The dataset used for building this movie recommender system is taken from Kaggle, available at the following link:

[Kaggle TMDB Movie Metadata](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata)

Download the dataset and place it in the appropriate location as specified in the Jupyter Notebook for data processing.

---

## Project Structure

- **MovieRecommender.ipynb**: 
    - Handles the data loading, cleaning, feature extraction, and training of the movie recommendation model.
    - Utilizes **Cosine Similarity** to measure the similarity between movies based on their features (e.g., genres, keywords, cast).
    - After processing, the model and necessary data are saved as a binary file (e.g., using `pickle` or `joblib`).

- **app.py**: 
    - Contains the Streamlit-based user interface.
    - Loads the binary model and data generated from the Jupyter Notebook.
    - Provides an interactive interface for users to select movie preferences and get recommendations based on **Cosine Similarity**.

---

## How to Run

### 1. Data Processing

1. Open the `MovieRecommender.ipynb` in Jupyter Notebook.
2. Run all cells to process the data, compute the cosine similarity matrix, and train the recommendation model. Save the model as a binary file (e.g., `model.pkl`).
   - This binary file will be used by the main Streamlit app.

### 2. Running the Streamlit App

1. Install the required dependencies using `pip`:
   ```bash
   pip install streamlit
   pip install pandas
   pip install numpy
   pip install scikit-learn
   pip install matplotlib
   pip install seaborn
   pip install pickle-mixin
   ```

2. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

3. Access the app on your browser at `http://localhost:8501`.

### 3. Model File

Ensure the binary file generated from the notebook (e.g., `model.pkl`) is in the same directory as `app.py` or provide the correct file path in the app code.

---

## Cosine Similarity for Recommendations

The recommendation engine is based on **Cosine Similarity**, which measures the cosine of the angle between two vectors in a multi-dimensional space. In this project, movie features such as genre, cast, and keywords are vectorized, and the similarity between two movies is calculated using this technique. Movies with higher cosine similarity scores are considered more similar, and these are recommended to the user.

---

## Dependencies

Install the required dependencies using the following commands:
```bash
pip install streamlit
pip install pandas
pip install numpy
pip install scikit-learn
pip install matplotlib
pip install seaborn
pip install pickle-mixin
```

These commands ensure all necessary packages are installed for both data processing and running the UI.

---


### Explanation:

1. **Data Processing and Cosine Similarity**: The Jupyter Notebook processes the Kaggle dataset, cleans it, and transforms the movie data into feature vectors. **Cosine Similarity** is then calculated between these vectors, and movies with higher similarity scores are recommended. The model is saved as a binary file for use in the Streamlit app.

2. **Streamlit for the User Interface**: The Streamlit app (`app.py`) provides an intuitive web-based UI where users can input their movie preferences. It uses the cosine similarity matrix and the pre-trained model to provide movie recommendations.

3. **Dataset from Kaggle**: The dataset for this recommender system is sourced from Kaggle’s TMDB Movie Metadata dataset, and the user is expected to download it before running the Notebook.

