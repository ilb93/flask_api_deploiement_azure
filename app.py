import os
from flask import Flask, request, jsonify
import tensorflow as tf
import joblib
import numpy as np

# Importer les bibliothèques pour Azure Application Insights
from opencensus.ext.azure.log_exporter import AzureLogHandler
from opencensus.ext.azure.trace_exporter import AzureExporter
from opencensus.trace import config_integration
from opencensus.trace.samplers import ProbabilitySampler
from opencensus.trace.tracer import Tracer
import logging

# Initialiser l'application Flask
app = Flask(__name__)

# Configurer Application Insights
INSTRUMENTATION_KEY = "47019b65-b8ca-40be-95c8-a0552c3b62b3"  # Remplace par ta clé d'instrumentation
logger = logging.getLogger(__name__)
logger.addHandler(AzureLogHandler(connection_string=f"InstrumentationKey={INSTRUMENTATION_KEY}"))
config_integration.trace_integrations(['logging', 'requests'])

# Initialiser le traceur pour enregistrer les traces
tracer = Tracer(exporter=AzureExporter(connection_string=f"InstrumentationKey={INSTRUMENTATION_KEY}"),
                sampler=ProbabilitySampler(1.0))  # 1.0 pour enregistrer 100% des requêtes

# Charger le modèle LSTM
try:
    model = tf.keras.models.load_model("lstm_model.keras")
    logger.info("Modèle LSTM chargé avec succès.")
except Exception as e:
    logger.error(f"Erreur lors du chargement du modèle LSTM : {e}")
    model = None

# Charger le tokenizer
try:
    tokenizer = joblib.load("tokenizer.pkl")  # Assurez-vous que le fichier tokenizer.pkl existe dans le même dossier
    logger.info("Tokenizer chargé avec succès.")
except Exception as e:
    logger.error(f"Erreur lors du chargement du tokenizer : {e}")
    tokenizer = None

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

        # Logger les tweets mal prédits (ajouter votre logique pour détecter les erreurs)
        for result in results:
            if result["sentiment"] == "négatif":  # Exemple : loguer les sentiments négatifs
                logger.warning(f"Tweet mal prédit : {result}")

        return jsonify({"predictions": results})
    except Exception as e:
        logger.error(f"Erreur lors de la prédiction : {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    # Utiliser le port attribué par Heroku ou 5000 par défaut
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
