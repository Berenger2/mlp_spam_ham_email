import streamlit as st
from tensorflow.keras.models import load_model
from sklearn.feature_extraction.text import TfidfVectorizer
import os
import pickle

model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'model', 'spam_ham_classifier_model.h5')
vectorizer_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'model', 'vectorizer.pkl')

if os.path.exists(model_path):
    model = load_model(model_path)
else:
    print("Model not found.")

if os.path.exists(vectorizer_path):
    with open(vectorizer_path, "rb") as file:
        vectorizer = pickle.load(file)
else:
    print("Vectorizer file not found.")

st.title("📧 Vérificateur de mail")
st.write("Cette application permet de prédire si un email est un **Spam** ou un **Ham** (non-Spam).")

user_input = st.text_area("Saisissez un email ci-dessous :(en anglais uniquement)")

if st.button("Verifier"):
    if user_input.strip() == "":
        st.warning("Veuillez saisir un texte avant de lancer la classification.")
    else:
        # Vectorisation du user_input
        email_vectorized = vectorizer.transform([user_input]).toarray()

        # Prédiction
        prediction = model.predict(email_vectorized)
        prediction_label = "Spam" if prediction[0][0] > 0.5 else "Ham"

        # Résultat
        st.subheader("Résultat de la classification :")
        st.write(f"L'email est classifié comme : **{prediction_label}**")
        st.write(f"Probabilité de Spam : **{prediction[0][0]:.2f}**")

        # Mon result display
        if prediction_label == "Spam":
            st.error("⚠️ Attention, cet email semble être un **Spam**.")
        else:
            st.success("✅ Cet email est un **Ham** (non-Spam).")

# Footer
st.markdown("---")
st.markdown("Bérenger AKODO | IPSSI - Mastère Big Data - 2025")