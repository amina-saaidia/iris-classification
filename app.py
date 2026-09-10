import os
import joblib
import pandas as pd
import streamlit as st

# Configure Streamlit page layout
st.set_page_config(page_title="Iris Classifier", page_icon="🌸", layout="centered")

# Robust relative path resolution targeting models/best_model.pkl
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "best_model.pkl")

# Load model pipeline safely
@st.cache_resource
def load_saved_model(path: str):
    if not os.path.exists(path):
        return None
    return joblib.load(path)

model = load_saved_model(MODEL_PATH)
species_names = ["setosa", "versicolor", "virginica"]

# SVG Icons representing Iris species
FLOWER_ICONS = {
    "setosa": """
        <svg viewBox="0 0 100 115" width="72" height="83">
            <line x1="50" y1="78" x2="50" y2="108" stroke="#5A7D4E" stroke-width="2.5"/>
            <ellipse cx="50" cy="105" rx="7" ry="3" fill="#5A7D4E"/>
            <path d="M50 58 C38 60 26 66 22 76 C20 84 26 90 34 88 C42 86 47 76 49 65 C50 62 50 60 50 58 Z" fill="#D9A7C7"/>
            <path d="M50 58 C62 60 74 66 78 76 C80 84 74 90 66 88 C58 86 53 76 51 65 C50 62 50 60 50 58 Z" fill="#D9A7C7"/>
            <path d="M50 58 C47 65 45 76 47 86 C48 90 52 90 53 86 C55 76 53 65 50 58 Z" fill="#E8BEDA"/>
            <path d="M50 58 C45 48 44 36 50 26 C56 36 55 48 50 58 Z" fill="#FBF3F8"/>
            <path d="M50 58 C42 53 32 50 26 55 C31 61 42 61 50 58 Z" fill="#F6E8F2"/>
            <path d="M50 58 C58 53 68 50 74 55 C69 61 58 61 50 58 Z" fill="#F6E8F2"/>
            <circle cx="50" cy="58" r="3.5" fill="#F4D35E"/>
        </svg>
    """,
    "versicolor": """
        <svg viewBox="0 0 100 115" width="72" height="83">
            <line x1="50" y1="83" x2="50" y2="108" stroke="#5A7D4E" stroke-width="2.5"/>
            <ellipse cx="50" cy="105" rx="7" ry="3" fill="#5A7D4E"/>
            <path d="M50 52 C34 56 16 66 13 82 C12 92 20 98 30 94 C40 90 46 76 49 61 C50 58 50 55 50 52 Z" fill="#5D6FC7"/>
            <path d="M50 52 C66 56 84 66 87 82 C88 92 80 98 70 94 C60 90 54 76 51 61 C50 58 50 55 50 52 Z" fill="#5D6FC7"/>
            <path d="M50 52 C46 62 43 78 46 92 C48 97 52 97 54 92 C57 78 54 62 50 52 Z" fill="#7182D2"/>
            <path d="M26 80 L40 68" stroke="#33418F" stroke-width="1" opacity="0.45" fill="none"/>
            <path d="M74 80 L60 68" stroke="#33418F" stroke-width="1" opacity="0.45" fill="none"/>
            <path d="M50 52 C42 40 40 24 50 12 C60 24 58 40 50 52 Z" fill="#AAB6EA"/>
            <path d="M50 52 C40 46 26 41 18 48 C26 56 40 56 50 52 Z" fill="#96A4E4"/>
            <path d="M50 52 C60 46 74 41 82 48 C74 56 60 56 50 52 Z" fill="#96A4E4"/>
            <circle cx="50" cy="52" r="4" fill="#F4D35E"/>
        </svg>
    """,
    "virginica": """
        <svg viewBox="0 0 100 115" width="72" height="83">
            <line x1="50" y1="87" x2="50" y2="110" stroke="#5A7D4E" stroke-width="2.5"/>
            <ellipse cx="50" cy="108" rx="7" ry="3" fill="#5A7D4E"/>
            <path d="M50 46 C30 51 6 64 4 84 C3 96 14 103 26 98 C38 93 45 76 49 58 C50 54 50 50 50 46 Z" fill="#4B2E7D"/>
            <path d="M50 46 C70 51 94 64 96 84 C97 96 86 103 74 98 C62 93 55 76 51 58 C50 54 50 50 50 46 Z" fill="#4B2E7D"/>
            <path d="M50 46 C45 58 41 76 44 93 C46 99 54 99 56 93 C59 76 55 58 50 46 Z" fill="#5E3D93"/>
            <path d="M22 88 L40 74" stroke="#2A1854" stroke-width="1" opacity="0.45" fill="none"/>
            <path d="M78 88 L60 74" stroke="#2A1854" stroke-width="1" opacity="0.45" fill="none"/>
            <path d="M50 46 C39 32 37 12 50 -2 C63 12 61 32 50 46 Z" fill="#7C5CB3"/>
            <path d="M50 46 C38 39 20 33 10 42 C20 52 38 52 50 46 Z" fill="#6B4AA3"/>
            <path d="M50 46 C62 39 80 33 90 42 C80 52 62 52 50 46 Z" fill="#6B4AA3"/>
            <circle cx="50" cy="46" r="4.5" fill="#EAC348"/>
        </svg>
    """,
}

st.title("Iris Flower Classifier")
st.write(
    "Adjust the measurements below to match your flower, then click **Predict** "
    "to see which species the model predicts."
)

if model is None:
    st.error(f"Model file not found at `{MODEL_PATH}`. Please run `python src/train.py` first to generate the best model artifact.")
else:
    col1, col2 = st.columns(2)
    with col1:
        sepal_length = st.slider("Sepal length (cm)", 4.0, 8.0, 5.1)
        petal_length = st.slider("Petal length (cm)", 1.0, 7.0, 1.4)
    with col2:
        sepal_width = st.slider("Sepal width (cm)", 2.0, 4.5, 3.5)
        petal_width = st.slider("Petal width (cm)", 0.1, 2.5, 0.2)

    if st.button("Predict", type="primary", use_container_width=True):
        sample = pd.DataFrame([{
            "sepal length (cm)": sepal_length,
            "sepal width (cm)": sepal_width,
            "petal length (cm)": petal_length,
            "petal width (cm)": petal_width,
        }])

        pred = model.predict(sample)[0]
        predicted_name = species_names[pred]

        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(sample)[0]
        else:
            proba = [1.0 if i == pred else 0.0 for i in range(len(species_names))]

        st.write("")  # spacing
        cards = st.columns(3)
        for i, name in enumerate(species_names):
            pct = round(proba[i] * 100)
            is_winner = (name == predicted_name)
            border_color = "#7F77DD" if is_winner else "#E5DFD0"
            bg_color = "#EEEDFE" if is_winner else "#FFFFFF"

            card_html = (
                f'<div style="text-align:center; padding:12px 6px; border-radius:10px; '
                f'background:{bg_color}; border:1.5px solid {border_color};">'
                f'{FLOWER_ICONS[name]}'
                f'<div style="font-size:13px; font-weight:500; color:#2C2C2A; margin-top:4px;">{name}</div>'
                f'<div style="font-size:15px; font-weight:600; color:#534AB7;">{pct}%</div>'
                f'</div>'
            )
            with cards[i]:
                st.markdown(card_html, unsafe_allow_html=True)

        st.markdown(
            f"<p style='text-align:center; margin-top:14px; color:#26215C;'>"
            f"Predicted species: <b>{predicted_name.capitalize()}</b></p>",
            unsafe_allow_html=True,
        )