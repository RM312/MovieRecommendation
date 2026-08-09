# 🎬 Movie Recommender System

## 📌 Overview

This project is a Content-Based Movie Recommendation System that suggests movies based on similarity between their features such as genres, cast, keywords, and overview. The system is built using a modular pipeline and deployed with an interactive Streamlit web application.
The workflow is divided into two main stages:
- Data Processing & Model Building  
Handles data cleaning, feature engineering, vectorization, and similarity computation.
- User Interface (Streamlit App)  
Provides an interactive platform for users to select a movie and receive recommendations in real time.


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
├── data/                   # Processed dataset and model files
│   ├── movies.pkl
│   └── similarity.pkl
│
├── notebooks/              # Data preprocessing & model building
│   └── MovieRecommender.ipynb
│
├── src/                    # Core recommendation logic (modular pipeline)
│   ├── data_loader.py
|   ├── feature_engineering.py
|   ├── preprocessing.py
|   ├── train.py
|   ├── vectorizer.py
|   └── recommender.py
│
├── requirements.txt        # Project dependencies
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

## 🙏 Acknowledgements

- Kaggle TMDB Dataset  
- Scikit-learn  
- Streamlit  
