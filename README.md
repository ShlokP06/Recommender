
# 🎮 Movie Recommender System

A comprehensive and intuitive Movie Recommender System that harnesses the power of **Machine Learning**, **Python**, and **Flask** on the backend, paired with a responsive **HTML/CSS/JavaScript** frontend to deliver a seamless user experience. This project demonstrates the implementation of multiple recommendation strategies, including content-based filtering, collaborative filtering, and hybrid approaches.

---

## 🔍 Overview

This application allows users to receive personalized movie recommendations by selecting a movie title. The system supports:

- **Content-Based Filtering**: Calculates the similarity between movies based on metadata such as genre, cast, crew, and plot descriptions using natural language processing and cosine similarity.
- **Collaborative Filtering**: Utilizes user ratings to recommend movies that similar users have enjoyed, implemented using the `Surprise` library with matrix factorization techniques like SVD.
- **Hybrid Filtering**: Integrates both content and collaborative filtering to enhance the accuracy and relevance of recommendations by combining their output. The final recommendation list is generated by merging scores from both techniques and prioritizing overlapping results.

---

## 🤨 Machine Learning Pipeline

- **Data Preprocessing**: Leveraged `pandas` to clean and merge multiple movie-related datasets.
- **Feature Extraction**: Created a composite textual feature using genre, keywords, cast, and director.
- **Text Vectorization**: Applied `CountVectorizer` from `scikit-learn` to transform text features into token counts.
- **Similarity Computation**: Utilized **Cosine Similarity** for content-based similarity scoring.
- **Collaborative Model**: Built a user-item matrix and implemented collaborative filtering using `Surprise` with SVD.
- **Model Storage**: Serialized models, similarity matrices, and title mappings using `joblib` for efficient retrieval.

---

## 🌐 Technologies Used

**Frontend:**

- HTML5
- CSS3 (Custom styles for responsiveness and layout)
- JavaScript (Client-side interactivity)

**Backend:**

- Python 3
- Flask (Lightweight WSGI framework for routing and API endpoints)

**ML & Data Handling:**

- pandas
- scikit-learn
- joblib
- numpy
- surprise
- nltk
- ast

---

## 🚀 Key Features

- 🔎 **Real-time Movie Recommendations**: Get top similar movies based on your selection.
- 📊 **Collaborative Filtering**: Discover movies based on similar users' ratings.
- 🧪 **Hybrid Recommendations**: Combine the power of content and collaborative approaches.
- 🎨 **Interactive & Responsive UI**: Optimized for desktops and mobile devices.
- 🌐 **Web-based Interface**: Easily accessible through any modern browser.
- 🧠 **ML-Driven Insights**: Powered by robust algorithms for intelligent movie suggestions.

---

## 🧰 Getting Started

Follow these instructions to set up and run the application locally.

### 1. Clone the Repository *(or download only what you need)*

```bash
git clone https://github.com/ShlokP06/Movie-Recommender-System.git
cd Movie-Recommender-System
```

> 💡 **Tip**: If you want to download just a folder or specific file without cloning the full repo, use [GitHub1s](https://github1s.com/) or download the folder as a ZIP from the GitHub web interface.

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Launch the Flask App

```bash
python app.py
```

### 4. Access the Application

Open your web browser and navigate to:[http://127.0.0.1:5000](http://127.0.0.1:5000)

---

## 📁 Directory Structure

```
Movie-Recommender-System/
├── app.py                    # Flask backend script
├── data/                     # Data files
│   ├── movies.csv            # Movie metadata
│   ├── ratings.csv           # User ratings
│   └── ...                   # Other relevant CSVs or datasets
├── models/                   # Serialized model files
│   ├── cosine_sim.npy        # Content similarity matrix
│   └── model_compressed.pkl  # Trained hybrid or collaborative model
├── static/                   # Static assets
│   ├── css/                  # CSS styles
│   └── js/                   # JavaScript files
├── templates/                # HTML templates
│   ├── index.html            # Homepage template
├── requirements.txt          # Python dependencies
└── README.md                 # Project documentation
```

---

## 📚 Future Enhancements

- Improve hybrid filtering logic by adding weight tunability
- Using Real-time datasets to incorporate newly released movies too.
- Displaying data such as movie poster, director, cast and a brief plot summary with the recommendations on the frontend.
- Allowing registration of new users and storing their data in the database.
- Making it possible for users to rate certain movies and then saving the data dynamically for future recommendations.

---

## 🙌 Contributions

Contributions are welcome! If you have suggestions for improvements or wish to add features, feel free to open a pull request. For major changes, please start by opening an issue to discuss what you would like to propose.

---

## 📸 Visual Previews

### Screenshot
![User Interface] (static/UI.png)


### Screen Recording
[![Movie Recommender] (static/UI.png)](static/demo.mp4)


---

## 📬 Connect With Me

**Shlok P** – [GitHub](https://github.com/ShlokP06)

**LinkedIn** – [https://www.linkedin.com/in/shlok-parikh-370773335/](https://www.linkedin.com/in/shlok-parikh-370773335/)

**Email** – [shlokparikhchoco@gmail.com](mailto\:shlokparikhchoco@gmail.com)

For questions, suggestions, or collaborations, feel free to reach out!

---

## ⚠️ Limitations

While the system performs well in many cases, there are several known limitations:

- **Cold Start Problem**: Struggles with recommending movies for new users or newly added movies without sufficient data.
- **Data Sparsity**: User-item rating matrix is sparse, affecting the performance of collaborative filtering.
- **Scalability**: Currently tested on a moderate dataset; larger datasets may require optimization or use of more scalable infrastructure.
- **Real-Time Updates**: Model retraining is offline; new ratings or user behavior aren’t reflected until models are retrained.
- **Metadata Dependency**: Content-based filtering heavily relies on accurate and comprehensive metadata (genres, cast, etc.).
