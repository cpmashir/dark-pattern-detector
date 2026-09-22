import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score

# 1. Load dataset (adjust column names if yours differ)
df = pd.read_csv(r'C:\Users\C P M ASHIR\Desktop\S7_CS_Minor_MiniProject\dataset.csv')
text_col = 'text' if 'text' in df.columns else df.columns[0]
label_col = 'label' if 'label' in df.columns else df.columns[1]
df.dropna(subset=[text_col, label_col], inplace=True)

# 2. Split dataset (80% Train, 20% Test)
X_train, X_test, y_train, y_test = train_test_split(
    df[text_col], df[label_col], test_size=0.2, random_state=42, stratify=df[label_col]
)

# 3. TF-IDF Vectorization (Unigrams to Trigrams)
vectorizer = TfidfVectorizer(ngram_range=(1, 3), max_features=5000, stop_words='english')
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# 4. Benchmark the 4 models
models = {
    "Multinomial Naive Bayes": MultinomialNB(),
    "Logistic Regression": LogisticRegression(class_weight='balanced', max_iter=1000),
    "LinearSVC": LinearSVC(class_weight='balanced', random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
}

best_model = None
best_f1 = -1
best_model_name = ""

print("=== Model Benchmarking ===")
for name, model in models.items():
    model.fit(X_train_vec, y_train)
    preds = model.predict(X_test_vec)
    macro_f1 = f1_score(y_test, preds, average='macro')
    print(f"\n{name} -> Macro F1: {macro_f1:.4f}")
    print(classification_report(y_test, preds, zero_division=0))
    
    if macro_f1 > best_f1:
        best_f1 = macro_f1
        best_model = model
        best_model_name = name

print(f"\nSelected Model for Deployment: {best_model_name} (F1: {best_f1:.4f})")

# 5. Export serialized model and vectorizer
joblib.dump(best_model, 'best_model.pkl')
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')
print("Exported 'best_model.pkl' and 'tfidf_vectorizer.pkl' successfully.")