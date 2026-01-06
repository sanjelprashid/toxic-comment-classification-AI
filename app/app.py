import pickle

# Load trained model
with open("models/best_model.pkl", "rb") as f:
    model = pickle.load(f)

# Load labels
with open("models/label_cols.pkl", "rb") as f:
    LABEL_COLS = pickle.load(f)

print("Toxic Comment Classification App")
print("Type 'exit' to quit\n")

while True:
    text = input("Enter a comment: ")

    if text.lower() == "exit":
        print("Exiting app.")
        break

    preds = model.predict([text])[0]

    print("\nPrediction:")
    for label, value in zip(LABEL_COLS, preds):
        print(f"{label:15}: {value}")
    print("-" * 30)
