# 🎬 Movie Recommender System

A **Content-Based Movie Recommendation System** built using **Python, Scikit-learn, and Streamlit**.  
The system recommends movies based on similarity between features like **genres, cast, keywords, and overview** using **Cosine Similarity**.

---

## 🌐 Live Demo

👉 [Click Here to Try the App](https://movie-deploy-za23lmsb9v5udkypcrpmop.streamlit.app/)

---

## 🚀 Features

- Content-based recommendation system  
- Cosine similarity-based recommendations  
- Interactive UI using **Streamlit**  
- Preprocessing and model building using Jupyter Notebook  
- Fast and scalable recommendation pipeline  

---

## 📁 Project Structure

```text
MovieRecommendation/

├── MovieRecommender.ipynb   # Data preprocessing & model building
├── app.py                   # Streamlit UI
├── model.pkl                # Serialized model (generated)
├── requirements.txt
├── README.md
```

---

## 🧠 Methodology

The system follows a **content-based filtering approach**:

1. Data cleaning and preprocessing  
2. Feature extraction (genres, keywords, cast, overview)  
3. Text vectorization (CountVectorizer / TF-IDF)  
4. Cosine similarity computation  
5. Model serialization (pickle/joblib)  
6. Recommendation via Streamlit UI  

---

## 📊 Dataset

The dataset used is:

🔗 [Kaggle TMDB Movie Metadata](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata)

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

### Step 1: Run Notebook (Model Creation)

1. Open `MovieRecommender.ipynb`
2. Run all cells
3. This will generate:
   ```
   model.pkl
   ```

---

### Step 2: Run Streamlit App

```bash
streamlit run app.py
```

Then open:

```
http://localhost:8501
```

---

## ⚙️ Model File

- The trained model is saved as `model.pkl`
- Ensure it is in the same directory as `app.py`
- This file contains:
  - Processed movie data  
  - Cosine similarity matrix  

---

## 📈 How It Works

- User selects a movie  
- System computes similarity scores  
- Top similar movies are returned  

---

## 🛠️ Technologies Used

- Python  
- Pandas  
- NumPy  
- Scikit-learn  
- Streamlit  
- Pickle / Joblib  

---

## 📊 Results

- Accurate similarity-based recommendations  
- Fast retrieval using precomputed similarity matrix  
- Interactive user experience via Streamlit  

---

## 🔮 Future Improvements

- Hybrid recommendation system  
- User-based personalization  
- Deploy with user login system  
- Improve UI/UX  

---

## 🙏 Acknowledgements

- Kaggle TMDB Dataset  
- Scikit-learn Documentation  
- Streamlit  
