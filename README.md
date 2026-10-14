# 🎬 Movie Recommendation System

A movie recommendation web app built with **Python**, **Machine Learning** and **Streamlit**. Select a movie you like and the app suggests similar movies you might enjoy. It also includes user authentication backed by a database.

---

## ✨ Features

- 🎯 Recommends similar movies based on a movie you select
- 🧠 Machine Learning model using a precomputed similarity matrix
- 🔐 User authentication (login / signup)
- 🗄️ Database integration for storing user data
- 🎨 Custom-styled Streamlit interface (`style.css`)
- 📓 Jupyter notebooks showing data analysis and model building

---

## 🛠️ Tech Stack

| Area | Technology |
|------|------------|
| Language | Python |
| Web framework | Streamlit |
| ML / Data | Pandas, NumPy, Scikit-learn |
| Model storage | Pickle (`.pkl`) |
| Styling | CSS |

---

## 📁 Project Structure

```
movie-recommandation-system/
├── data/              # Dataset used to build the model
├── notebooks/         # Jupyter notebooks (EDA + model building)
├── app.py             # Main Streamlit application
├── auth.py            # Authentication logic (login / signup)
├── db.py              # Database connection and queries
├── movies.pkl         # Processed movies data
├── similarity.pkl     # Precomputed similarity matrix
├── style.css          # Custom styling for the app
├── .gitattributes
└── .gitignore
```

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/suhel-ali112/movie-recommandation-system.git
cd movie-recommandation-system
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install streamlit pandas numpy scikit-learn
```

> 💡 Add any other libraries your project uses (for example, requests or a database driver) to this command, or create a `requirements.txt` file.

### 4. Run the app

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`.

---

## 🧩 How It Works

1. **Data preparation** – The movie dataset in `data/` is cleaned and processed in the notebooks.
2. **Feature extraction** – Movie information is converted into numerical vectors.
3. **Similarity matrix** – Similarity between all movies is computed and saved as `similarity.pkl`, along with the processed movie list in `movies.pkl`.
4. **Recommendation** – When a user selects a movie, the app finds the most similar movies from the similarity matrix and displays them.
5. **User management** – `auth.py` and `db.py` handle user accounts and database operations.

---

## 📸 Screenshots

_Add screenshots of your app here._

```
![Home Page](screenshots/home.png)
![Recommendations](screenshots/recommendations.png)
```

---

## 🔮 Future Improvements

- Show movie posters and details using a movie API
- Add filters by genre, year and rating
- Save favourite movies per user
- Deploy the app online (Streamlit Community Cloud)

---

## 🤝 Contributing

Contributions are welcome! Fork the repository, create a new branch, make your changes and open a pull request.

---

## 👤 Author
Last Commit


---

⭐ If you like this project, please give it a star!
