import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer

DATASET_PATH = r"C:\Users\C P M ASHIR\Desktop\S7_CS_Minor_MiniProject\dataset.csv"

# 1. LOAD DATASET
df = pd.read_csv(DATASET_PATH)

print("=" * 50)
print("1. DATASET LOADED SUCCESSFULLY")
print(f"Total Rows: {len(df)}")
print("Columns in dataset:", list(df.columns))
print("=" * 50)

# Automatically identify the text column
text_candidates = [
    'text',
    'Pattern_String',
    'pattern_string',
    'content',
    'Sentence',
    'sentence',
    'tokens',
]
text_col = None

for col in text_candidates:
    if col in df.columns:
        text_col = col
        break

if not text_col:
    for col in df.columns:
        if df[col].dtype == 'object':
            text_col = col
            break

if not text_col:
    text_col = df.columns[0]

print(f"\nUsing column: '{text_col}' for NLP feature extraction.\n")


# 2. TEXT PREPROCESSING FUNCTION
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()                                # Lowercasing
    text = re.sub(r'<.*?>', '', text)                  # Strip HTML tags
    text = re.sub(r'https?://\S+|www\.\S+', '', text)  # Strip URLs
    text = re.sub(r'[^\w\s]', '', text)                # Remove punctuation
    return text.strip()


# Apply cleaning
df['cleaned_text'] = df[text_col].apply(clean_text)

# 3. TF-IDF VECTORIZATION (1 to 3 Grams)
vectorizer = TfidfVectorizer(
    ngram_range=(1, 3),  # Unigrams, bigrams, and trigrams
    max_features=2000,
    sublinear_tf=True,
)

tfidf_matrix = vectorizer.fit_transform(df['cleaned_text'])

# 4. VERIFICATION OUTPUT
print("=" * 50)
print("2. PREPROCESSING & TF-IDF (1-3 GRAM) PIPELINE READY")
print("=" * 50)
print(f"Input Samples Processed : {tfidf_matrix.shape[0]}")
print(f"TF-IDF Vocabulary Size  : {tfidf_matrix.shape[1]} n-gram features")

features = vectorizer.get_feature_names_out()
multi_grams = [f for f in features if ' ' in f]

print(f"\nDiscovered {len(multi_grams)} multi-word trigger phrases (bigrams/trigrams).")
print("\nSample 1-3 Gram Features Extracted:")
for i, phrase in enumerate(multi_grams[:15], 1):
    print(f"  {i}. {phrase}")
print("=" * 50)