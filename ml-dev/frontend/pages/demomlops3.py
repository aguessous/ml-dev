import streamlit as st
import requests
import pandas as pd
import io
import json
import config

def main():
    # Configuration de la page
    st.set_page_config(
        page_title="DemoMLOps3 - Assurance Cross-Sell",
        layout="wide",
        page_icon="🚀"
    )
    
    st.title("🚀 Projet DemoMLOps3")
    st.markdown("**Pipeline AutoML pour prédiction de cross-sell**")
    
    # --- Section Entraînement ---
    with st.expander("🔧 Entraînement du modèle", expanded=True):
        col_train1, col_train2 = st.columns([3, 1])
        
        with col_train1:
            train_file = st.file_uploader(
                "Dataset d'entraînement (CSV)",
                type=["csv"],
                key="train_upload"
            )
            
        with col_train2:
            st.markdown("### Paramètres")
            target_col = st.text_input("Colonne cible", "Response")
            max_models = st.slider("Nombre de modèles", 1, 20, 5)
            
            if st.button("Lancer l'entraînement AutoML", type="primary"):
                if train_file:
                    files = {"file": (train_file.name, train_file.getvalue())}
                    params = {
                        "target": target_col,
                        "max_models": max_models
                    }
                    
                    with st.spinner("Entraînement en cours (peut prendre plusieurs minutes)..."):
                        response = requests.post(
                            f"{config.BACKEND_URL}/train",
                            files=files,
                            data=params,
                            timeout=3600
                        )
                        
                        if response.status_code == 200:
                            st.success("Entraînement réussi !")
                            st.json(response.json())
                        else:
                            st.error(f"Erreur : {response.text}")
                else:
                    st.warning("Veuillez uploader un fichier CSV")

    # --- Section Prédiction ---
    with st.expander("🔮 Prédiction en temps réel", expanded=True):
        test_file = st.file_uploader(
            "Dataset de test (CSV)",
            type=["csv"],
            key="test_upload"
        )
        
        if test_file:
            test_df = pd.read_csv(test_file)
            st.write("Aperçu des données :")
            st.dataframe(test_df.head())
            
            if st.button("Générer les prédictions"):
                test_bytes = io.BytesIO()
                test_df.to_csv(test_bytes, index=False)
                test_bytes.seek(0)
                
                with st.spinner("Analyse en cours..."):
                    try:
                        response = requests.post(
                            f"{config.BACKEND_URL}/predict",
                            files={"file": ("test.csv", test_bytes)},
                            timeout=30
                        )
                        
                        if response.status_code == 200:
                            predictions = response.json()
                            st.success(f"Probabilité moyenne : {predictions['average_probability']:.2%}")
                            
                            col_pred1, col_pred2 = st.columns(2)
                            with col_pred1:
                                st.download_button(
                                    "Télécharger les prédictions",
                                    data=json.dumps(predictions),
                                    file_name="predictions.json"
                                )
                            with col_pred2:
                                st.metric("Clients à cibler", 
                                         f"{predictions['targeted_customers']} / {len(test_df)}")
                        else:
                            st.error(f"Erreur du serveur : {response.text}")
                            
                    except Exception as e:
                        st.error(f"Erreur de connexion : {str(e)}")

if __name__ == "__main__":
    main()