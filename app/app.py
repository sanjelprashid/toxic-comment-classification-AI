import pickle
import re

# -------------------------------
# Text Preprocessing Function
# -------------------------------

def clean_text(s):
    s = s.lower()
    s = re.sub(r"http\S+|www\.\S+", " ", s)
    s = re.sub(r"[^a-z\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

# ---------------------------------------------------------------
# Load Trained Artifacts TF-IDF + Logistic Regression pipeline
# ---------------------------------------------------------------

with open("../models/toxicity_pipeline.pkl", "rb") as f:
    model = pickle.load(f)

with open("../models/thresholds.pkl", "rb") as f:
    thresholds = pickle.load(f)

with open("../models/label_cols.pkl", "rb") as f:
    labels = pickle.load(f)

# -------------------------------
# Command-Line Interface
# -------------------------------

print("Toxic Comment Classifier")
print("------------------------")

while True:
    text = input("\nEnter a comment (or type 'exit'): ")
    if text.lower() == "exit":
        break

    text_clean = clean_text(text)
    probs = model.predict_proba([text_clean])[0]

    print("\nPrediction:")
    for i, label in enumerate(labels):
        result = int(probs[i] >= thresholds[label])
        print(f"{label:15s}: {result}")
