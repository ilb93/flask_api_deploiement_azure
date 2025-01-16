from flask import Flask, request, jsonify
import tensorflow as tf
import joblib
import numpy as np

app = Flask(__name__)

# Charger le modèle LSTM
model = tf.keras.models.load_model("lstm_model.keras")
print("Modèle LSTM chargé avec succès.")

# Charger le tokenizer
tokenizer = joblib.load("tokenizer.pkl")  # Assurez-vous que le fichier tokenizer.pkl existe dans le même dossier
print("Tokenizer chargé avec succès.")

@app.route('/predict', methods=['POST'])
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

if __name__ == '__main__':
    app.run(port=5002)


