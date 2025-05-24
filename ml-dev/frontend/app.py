import streamlit as st
import requests
import config

# 1) Paramètre d’URL
params = st.experimental_get_query_params()
page = params.get("page", ["home"])[0]
if page == "demomlops3":
    import pages.demomlops3 as demo; demo.main(); st.stop()

# 2) Page d’accueil
st.set_page_config(page_title="AutoML Dashboard AMG", layout="wide")
st.title("AutoML Dashboard AMG")

# --- Barre latérale config -----------------------------------------------
with st.sidebar:
    st.title("Navigation")
    if st.button("🏠 Accueil"):
        st.experimental_set_query_params(page="home"); st.experimental_rerun()
    if st.button("🚀 Projet DemoMLOps3"):
        st.experimental_set_query_params(page="demomlops3"); st.experimental_rerun()
    st.markdown("---")
    st.title("Configuration")
    st.write(f"**Backend URL:** {config.BACKEND_URL}")
    st.write(f"**MLflow URL:** {config.MLFLOW_TRACKING_URL}")

# --- Vérification de santé backend ---------------------------------------
with st.spinner("Vérification du backend…"):
    try:
        r = requests.get(f"{config.BACKEND_URL}/health", timeout=5)
        if r.status_code == 200:
            st.success("Backend service is healthy ✔")
        else:
            st.error(f"Backend responded with status {r.status_code}")
    except Exception as exc:
        st.error(f"Connexion impossible au backend : {exc}")

st.markdown('---')

# --- Section principale améliorée ----------------------------------------
st.header("Bienvenue dans votre environnement AutoML")

# Colonnes redimensionnées
col1, col2, col3 = st.columns([1,1,1])

with col1:
    st.info("**Backend API**")
    st.markdown(f"[Ouvrir l'API]({config.BACKEND_URL})", unsafe_allow_html=True)

with col2:
    st.info("**MLflow Tracking UI**")
    st.markdown(f"[Ouvrir MLflow]({config.MLFLOW_TRACKING_URL})", unsafe_allow_html=True)

with col3:
    st.info("**Jupyter Lab**")
    st.markdown("[http://localhost:8888](http://localhost:8888)")



# --- Section documentation ----------------------------------------------
st.markdown("---")
with st.expander("📚 Documentation des projets", expanded=True):
    st.write("""
    **Projet DemoMLOps3**  
    Pipeline complet AutoML pour la prédiction de cross-sell en assurance :
    - Entraînement avec H2O AutoML
    - Tracking MLflow
    - Déploiement FastAPI
    - Interface Streamlit
    """)
    st.markdown("[Voir le dépôt GitHub](https://github.com/haythem-rehouma/demomlops3)")
