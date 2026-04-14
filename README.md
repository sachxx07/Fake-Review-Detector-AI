# ReviewGuard AI — Fake Product Review Detector

An AI-powered web application that detects fake product reviews using Machine Learning and Natural Language Processing.

---

## Features

- **Real-time Review Analysis** — Paste any product review and get an instant Fake / Genuine classification
- **Confidence Score** — Every prediction includes a percentage confidence level
- **Exaggeration Detection** — Catches overly promotional language, excessive punctuation, and ALL CAPS patterns
- **Short Input Handling** — Reviews with fewer than 5 words return a neutral "insufficient data" response
- **Professional UI** — Modern glassmorphic dark-mode interface with smooth animations

---

## Tech Stack

| Layer      | Technology                                       |
| ---------- | ------------------------------------------------ |
| Backend    | Python, Flask                                    |
| ML Model   | Logistic Regression / Linear SVM (scikit-learn)  |
| Features   | TF-IDF (unigrams + bigrams + trigrams) + custom behavioral features |
| Frontend   | HTML5, CSS3, Vanilla JavaScript                  |
| Deployment | Render (gunicorn)                                |

---

## Project Structure

```
├── app.py                 # Flask backend + prediction API
├── model.pkl              # Trained ML model
├── vectorizer.pkl         # TF-IDF vectorizer
├── scaler.pkl             # Feature scaler for meta-features
├── requirements.txt       # Python dependencies
├── Procfile               # Render deployment config
├── templates/
│   └── index.html         # Frontend UI
├── static/
│   └── style.css          # Stylesheet
└── README.md              # This file
```

---

## How to Run Locally

```bash
# 1. Clone or navigate to the project folder
cd CSE-275

# 2. Install dependencies
pip install -r requirements.txt

# 3. Start the Flask server
python app.py

# 4. Open in browser
# http://127.0.0.1:5000
```

---

## Deployment (Render)

The project is deployed on Render using gunicorn.

**Live URL:** _(add your Render link here)_

**Render settings:**
- **Build Command:** `pip install -r requirements.txt`
- **Start Command:** `gunicorn app:app`

---

## How It Works

1. User enters a product review in the text box
2. The backend cleans the text and extracts TF-IDF features (up to 8,000 n-grams)
3. Additional behavioral features are computed: review length, exclamation count, ALL CAPS words, and promotional keyword density
4. TF-IDF + behavioral features are combined and passed to the trained ML model
5. The model returns a Fake / Genuine prediction with a confidence score

---

## Screenshots

_(Add screenshots of the app here)_
