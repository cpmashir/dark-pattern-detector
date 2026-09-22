import os
import re
import cv2
import numpy as np
import pytesseract
from sklearn.feature_extraction.text import TfidfVectorizer

# 1. LINK INSTALLED TESSERACT ENGINE
pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)

# 2. SETUP PATHS
PROJECT_DIR = r"C:\Users\C P M ASHIR\Desktop\S7_CS_Minor_MiniProject"
IMAGE_PATH = os.path.join(PROJECT_DIR, "test_banner.png")

print("=" * 60)
print("1. PYTESSERACT OCR IMAGE EXTRACTION SETUP")
print("=" * 60)

# Create a sample dark pattern banner image if not present
if not os.path.exists(IMAGE_PATH):
  canvas = np.full((120, 620, 3), 245, dtype=np.uint8)
  cv2.putText(
      canvas,
      "HURRY! Only 2 items left at this price!",
      (20, 48),
      cv2.FONT_HERSHEY_SIMPLEX,
      0.65,
      (20, 20, 180),
      2,
  )
  cv2.putText(
      canvas,
      "Offer expires in 04:59 mins. Claim now.",
      (20, 85),
      cv2.FONT_HERSHEY_SIMPLEX,
      0.55,
      (80, 80, 80),
      1,
  )
  cv2.imwrite(IMAGE_PATH, canvas)
  print(f"[+] Created test UI artifact: {IMAGE_PATH}")
else:
  print(f"[+] Loaded existing UI artifact: {IMAGE_PATH}")

# 3. OPENCV IMAGE PREPROCESSING
img = cv2.imread(IMAGE_PATH)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
print(
    "[+] Image Preprocessing: Grayscale conversion + Otsu Thresholding applied."
)

# 4. RUN PYTESSERACT OCR
custom_config = r"--oem 3 --psm 6"
extracted_text = pytesseract.image_to_string(
    thresh, config=custom_config
).strip()

print("\n" + "=" * 60)
print("2. EXTRACTED OCR TEXT FROM UI ARTIFACT")
print("=" * 60)
print(f'Raw Output:\n"{extracted_text}"')

# 5. BRIDGE TO NLP PREPROCESSING & TF-IDF (1-3 GRAM)
print("\n" + "=" * 60)
print("3. PIPELINE BRIDGE: OCR TEXT -> TF-IDF (1-3 GRAM)")
print("=" * 60)


def clean_text(text):
  text = text.lower()
  text = re.sub(r"[^\w\s]", "", text)
  return text.strip()


cleaned_text = clean_text(extracted_text)
print(f"Cleaned String : '{cleaned_text}'")

vec = TfidfVectorizer(ngram_range=(1, 3))
tfidf_matrix = vec.fit_transform([cleaned_text])

print(f"\nExtracted Multi-Word Trigger Phrases (1-3 Grams):")
for idx, feature in enumerate(vec.get_feature_names_out(), 1):
  if " " in feature:
    print(f"  {idx}. [Trigger Phrase] -> '{feature}'")
print("=" * 60)