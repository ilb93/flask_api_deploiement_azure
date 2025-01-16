import os
from flask import Flask, request, jsonify
import tensorflow as tf
import joblib
import numpy as np

# Initialiser l'application Flask
app = Flask(__name__)

# Charger le modèle LSTM
try:
    model = tf.keras.models.load_model("lstm_model.keras")
    print("Modèle LSTM chargé avec succès.")
except Exception as e:
    print(f"Erreur lors du chargement du modèle LSTM : {e}")

# Charger le tokenizer
try:
    tokenizer = joblib.load("tokenizer.pkl")  # Assurez-vous que le fichier tokenizer.pkl existe dans le même dossier
    print("Tokenizer chargé avec succès.")
except Exception as e:
    print(f"Erreur lors du chargement du tokenizer : {e}")

@app.route("/")
def home():
    return "Bienvenue sur mon API Flask déployée sur Heroku !"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        tweets = data.get('tweets', [])
        if not tweets:
            return jsonify({"error": "Aucun tweet fourni."}), 400

        # Transformer les tweets en séquences
        sequences = tokenizer.texts_to_sequences(tweets)
        max_length = 100  # Longueur maximale des séquences, ajustez si nécessaire
        padded_sequences = tf.keras.preprocessing.sequence.pad_sequences(sequences, maxlen=max_length)

        # Faire les prédictions
        predictions = model.predict(padded_sequences)

        # Formater les résultats
        results = [
            {
                "tweet": tweet,
                "sentiment": "positif" if prediction > 0.5 else "négatif"
            }
            for tweet, prediction in zip(tweets, predictions.flatten())
        ]

        return jsonify({"predictions": results})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    # Utiliser le port attribué par Heroku ou 5000 par défaut
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)



