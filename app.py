from flask import Flask, render_template, request, jsonify
import pickle
import os
import re
import numpy as np
from scipy.sparse import hstack, csr_matrix

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model = pickle.load(open(os.path.join(BASE_DIR, "model.pkl"), "rb"))
vectorizer = pickle.load(open(os.path.join(BASE_DIR, "vectorizer.pkl"), "rb"))
scaler = pickle.load(open(os.path.join(BASE_DIR, "scaler.pkl"), "rb"))

# Massively expanded promotional triggers for exaggerated fake classification
promo_words = [
    "best", "amazing", "perfect", "must buy", "100%", "guaranteed", 
    "excellent", "fantastic", "incredible", "unbelievable", "awesome", 
    "happier", "love", "highly recommend", "flawless", "superb",
    "wonderful", "outstanding", "brilliant", "phenomenal"
]

def extract_meta(text):
    text_str = str(text)
    review_len = len(text_str)
    excl_count = text_str.count('!')
    u_count = sum(1 for c in text_str.split() if c.isupper() and len(c) > 1)
    text_lower = text_str.lower()
    promo_count = sum(1 for w in promo_words if w in text_lower)
    return np.array([[review_len, excl_count, u_count, promo_count]])

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    review = data.get("review", "")
    
    if not review.strip():
        return jsonify({"error": "Please enter a review text."}), 400
        
    meta_features = extract_meta(review)
    meta_scaled = scaler.transform(meta_features)
    
    cleaned = clean_text(review)
    transformed_tfidf = vectorizer.transform([cleaned])
    
    X_input = hstack([transformed_tfidf, csr_matrix(meta_scaled)])
    
    if hasattr(model, "predict_proba"):
        confidence = model.predict_proba(X_input)[0]
        # Use ML base but boost fake probability on exaggerated inputs
        fake_prob = confidence[1]
    else:
        decision = model.decision_function(X_input)[0]
        
        # --- EXAGGERATION DETECTION OVERRIDE ---
        review_len = meta_features[0][0]
        excl_count = meta_features[0][1]
        upper_count = meta_features[0][2]
        promo_count = meta_features[0][3]
        
        # Penalize generic excessive positivity and short fake loops
        promo_score = promo_count * 0.9
        excl_score = 0.5 if excl_count > 0 else 0
        upper_score = 0.5 if upper_count > 0 else 0
        len_penalty = 0.8 if review_len < 120 and promo_count >= 1 else 0
        
        boost = promo_score + excl_score + upper_score + len_penalty
        final_decision = decision + boost
        
        fake_prob = 1 / (1 + np.exp(-final_decision))

    prediction = 1 if fake_prob >= 0.50 else 0
    
    if prediction == 1:
        result = "Fake"
        conf_score = round(float(fake_prob) * 100, 1)
    else:
        result = "Genuine"
        conf_score = round(float(1 - fake_prob) * 100, 1)
    
    return jsonify({"result": result, "confidence": conf_score})

if __name__ == "__main__":
    app.run(debug=True)
