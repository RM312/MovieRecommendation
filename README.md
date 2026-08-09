# 🎬 Movie Recommender System

## Overview

-This project is a Movie Recommender System that processes data in a Jupyter Notebook and uses the processed data in a Python-based UI built with Streamlit. The recommendation system is based on Cosine Similarity, a technique commonly used to measure the similarity between two vectors (in this case, movie features). The workflow is split into two parts:

-Data Processing in Jupyter Notebook: Data cleaning, processing, and model training are performed in the MovieRecommender.ipynb. Cosine similarity is used to calculate the similarity between movies based on their features. The results, including the recommendation model, are serialized into a binary file that the main program can load.
User Interface using Streamlit: The UI is developed using Streamlit (app.py), which provides an interactive interface where users can input their preferences and get movie recommendations.

---


👉 You can access the live version of the project here: [Movie Recommnedation System](https://movie-deploy-za23lmsb9v5udkypcrpmop.streamlit.app/)

---

## 🚀 Features

- Content-based recommendation system  
- Cosine similarity-based recommendations  
- Interactive UI using **Streamlit**  
- Modular project structure  
- Fast recommendations using precomputed similarity matrix  

---

## 📁 Project Structure

```text
MovieRecommendation/

├── app/                    # Streamlit application
│   └── app.py
│
├── data/                   # Dataset and processed files
│   ├── movies.pkl
│   └── similarity.pkl
│
├── notebooks/              # Jupyter notebooks for preprocessing
│   └── MovieRecommender.ipynb
│
├── src/                    # Core recommendation logic
│   └── recommender.py (or similar logic files)
│
├── requirements.txt        # Dependencies
└── README.md
```

---

## 🧠 Methodology

The system follows a **content-based filtering approach**:

1. Data cleaning and preprocessing  
2. Feature extraction (genres, keywords, cast, overview)  
3. Text vectorization using CountVectorizer  
4. Cosine similarity computation  
5. Saving processed data as `.pkl` files  
6. Recommendation through Streamlit UI  

---

## 📊 Dataset

The dataset used is:

🔗 https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata  

### Dataset Includes:

- Movie titles  
- Genres  
- Cast & crew  
- Keywords  
- Overview  

---

## 🔧 Installation

Clone the repository:

```bash
git clone https://github.com/RM312/MovieRecommendation.git
cd MovieRecommendation
```

---

## 📦 Requirements

All dependencies are listed in:

```
requirements.txt
```

Install them using:

```bash
pip install -r requirements.txt
```

---

## 🚀 Quick Start

### Step 1: Run Notebook (Data Processing)

1. Open:
   ```
   notebooks/MovieRecommender.ipynb
   ```
2. Run all cells
3. This will generate:

```
data/movies.pkl
data/similarity.pkl
```

---

### Step 2: Run Streamlit App

```bash
streamlit run app/app.py
```

Then open:

```
http://localhost:8501
```

---

## ⚙️ Model Files

The following files are generated after preprocessing:

- `data/movies.pkl` → Processed movie dataset  
- `data/similarity.pkl` → Cosine similarity matrix  

Ensure these files exist before running the application.

---

## ▶️ How It Works

- User selects a movie  
- System computes similarity scores  
- Top similar movies are recommended  

---

## 📈 Output

- List of recommended movies  
- Ranked by similarity score  

---

## 🛠️ Technologies Used

- Python  
- Pandas  
- NumPy  
- Scikit-learn  
- Streamlit  
- Pickle  

---

## 📊 Results

- Fast recommendations using precomputed similarity matrix  
- Accurate similarity-based suggestions  
- Interactive web-based UI  

---

## 🔮 Future Improvements

- Hybrid recommendation system  
- User-based personalization  
- Improved UI/UX  
- Deployment with authentication  

---

## 📜 License

This project is for **educational purposes only**.

---

## 🙏 Acknowledgements

- Kaggle TMDB Dataset  
- Scikit-learn  
- Streamlit  
