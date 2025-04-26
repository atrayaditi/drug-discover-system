from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np

from utils import smiles_to_fingerprint, encode_sequence

app = Flask(__name__)
CORS(app, resources={r"/predict": {"origins": "*"}})

# Load model and vectorizer
model = joblib.load("rf_model.pkl")
vectorizer = joblib.load("kmer_vectorizer.pkl")
expected_features = model.n_features_in_

@app.route("/")
def home():
    return "✅ Flask backend is running!"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    smiles = data.get("smiles")
    sequence = data.get("sequence")

    if not smiles or not sequence:
        return jsonify({"error": "Both SMILES and sequence are required."}), 400

    fp = smiles_to_fingerprint(smiles)
    if fp is None:
        return jsonify({"error": "Invalid SMILES"}), 400

    try:
        kmers = encode_sequence(sequence)
        seq_vector = vectorizer.transform([kmers]).toarray()
        combined = np.hstack([fp.reshape(1, -1), seq_vector])

        # Adjust shape to match model input (8811)
        current_features = combined.shape[1]

        if current_features > expected_features:
            combined = combined[:, :expected_features]  # Truncate
        elif current_features < expected_features:
            padding = np.zeros((1, expected_features - current_features))
            combined = np.hstack([combined, padding])  # Pad

        # Predict KIBA score
        score = model.predict(combined)[0]
        return jsonify({"kiba_score": float(score)})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
