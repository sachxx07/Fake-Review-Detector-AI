import pandas as pd
import numpy as np
import pickle
import re
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, f1_score

df = pd.read_csv("fake reviews dataset.csv", encoding="latin1")

text_col = None
label_col = None
for col in df.columns:
    if 'text' in col.lower() or 'review' in col.lower():
        text_col = col
    if 'label' in col.lower() or 'class' in col.lower() or 'target' in col.lower():
        label_col = col

df = df[[text_col, label_col]].dropna()
df['label_encoded'] = df[label_col].apply(lambda x: 1 if str(x).strip().upper() == 'CG' else 0)

promo_words = ["best", "amazing", "perfect", "must buy", "100%", "guaranteed"]

def extract_meta(text):
    text_str = str(text)
    review_len = len(text_str)
    excl_count = text_str.count('!')
    u_count = sum(1 for c in text_str.split() if c.isupper() and len(c) > 1)
    
    text_lower = text_str.lower()
    promo_count = sum(1 for w in promo_words if w in text_lower)
    
    return [review_len, excl_count, u_count, promo_count]

meta_list = df[text_col].apply(extract_meta).tolist()
meta_df = pd.DataFrame(meta_list, columns=['review_len', 'exclamation_count', 'upper_count', 'promo_count'])

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

df['clean'] = df[text_col].apply(clean_text)

vectorizer = TfidfVectorizer(
    max_features=8000,
    ngram_range=(1, 3),
    min_df=2,
    stop_words='english'
)

X_tfidf = vectorizer.fit_transform(df['clean'])

scaler = StandardScaler()
meta_scaled = scaler.fit_transform(meta_df.values)

X = hstack([X_tfidf, csr_matrix(meta_scaled)])
y = df['label_encoded'].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

models = {
    "Logistic Regression": LogisticRegression(max_iter=2000, C=5.0, class_weight='balanced', random_state=42),
    "Naive Bayes": MultinomialNB(alpha=0.5),
    "SVM": LinearSVC(C=1.0, class_weight='balanced', dual=False, random_state=42)
}

best_f1 = 0
best_model = None

print("Training and evaluating models...\n")
for name, model in models.items():
    if name == "Naive Bayes":
        from sklearn.preprocessing import MinMaxScaler
        nb_scaler = MinMaxScaler()
        X_train_nb = hstack([X_train[:,:-4], csr_matrix(nb_scaler.fit_transform(X_train[:,-4:].toarray()))])
        X_test_nb = hstack([X_test[:,:-4], csr_matrix(nb_scaler.transform(X_test[:,-4:].toarray()))])
        model.fit(X_train_nb, y_train)
        pred = model.predict(X_test_nb)
    else:
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        
    f1 = f1_score(y_test, pred, average='weighted')
    print(f"--- {name} ---")
    print(confusion_matrix(y_test, pred))
    print(classification_report(y_test, pred))
    
    if f1 > best_f1:
        best_f1 = f1
        best_model = model

print(f"Saving best model with F1: {best_f1:.4f}")
pickle.dump(best_model, open("model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))
pickle.dump(scaler, open("scaler.pkl", "wb"))