import streamlit as st
import onnxruntime as ort
import numpy as np
from PIL import Image
import json
import time
import io
import requests
from pathlib import Path
import plotly.express as px
import pandas as pd
from huggingface_hub import hf_hub_download

st.set_page_config(
    page_title="Klasifikasi 101 Makanan",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #ff6b6b;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #28a745;
        margin: 1rem 0;
    }
    .confidence-bar {
        background-color: #e9ecef;
        border-radius: 10px;
        overflow: hidden;
        margin: 0.5rem 0;
    }
    .sidebar-info {
        background-color: #e3f2fd;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)
CLASS_NAMES = [
            "apple_pie","baby_back_ribs","baklava","beef_carpaccio","beef_tartare",
            "beet_salad","beignets","bibimbap","bread_pudding","breakfast_burrito",
            "bruschetta","caesar_salad","cannoli","caprese_salad","carrot_cake",
            "ceviche","cheesecake","cheese_plate","chicken_curry","chicken_quesadilla",
            "chicken_wings","chocolate_cake","chocolate_mousse","churros","clam_chowder",
            "club_sandwich","crab_cakes","creme_brulee","croque_madame","cup_cakes",
            "deviled_eggs","donuts","dumplings","edamame","eggs_benedict",
            "escargots","falafel","filet_mignon","fish_and_chips","foie_gras",
            "french_fries","french_onion_soup","french_toast","fried_calamari","fried_rice",
            "frozen_yogurt","garlic_bread","gnocchi","greek_salad","grilled_cheese_sandwich",
            "grilled_salmon","guacamole","gyoza","hamburger","hot_and_sour_soup",
            "hot_dog","huevos_rancheros","hummus","ice_cream","lasagna",
            "lobster_bisque","lobster_roll_sandwich","macaroni_and_cheese","macarons","miso_soup",
            "mussels","nachos","omelette","onion_rings","oysters",
            "pad_thai","paella","pancakes","panna_cotta","peking_duck",
            "pho","pizza","pork_chop","poutine","prime_rib",
            "pulled_pork_sandwich","ramen","ravioli","red_velvet_cake","risotto",
            "samosa","sashimi","scallops","seaweed_salad","shrimp_and_grits",
            "spaghetti_bolognese","spaghetti_carbonara","spring_rolls","steak","strawberry_shortcake",
            "sushi","tacos","takoyaki","tiramisu","tuna_tartare",
            "waffles"
        ]
class Food101Classifier:
    def __init__(self, repo_id="Wijdanadam/Food101_Classification", filename="food101_resnet50.onnx"):
        self.repo_id = repo_id
        self.filename = filename
        self.session = None
        self.class_names = CLASS_NAMES
        self.input_type = None
        self.load_model()

    def load_model(self):
        model_path = hf_hub_download(
            repo_id=self.repo_id,
            filename=self.filename
        )
        providers = ['CPUExecutionProvider']
        if 'CUDAExecutionProvider' in ort.get_available_providers():
            providers.insert(0, 'CUDAExecutionProvider')

        self.session = ort.InferenceSession(model_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape
        self.input_type = self.session.get_inputs()[0].type

    def preprocess_image(self, image):
        image = image.convert('RGB').resize((256, 256))
        left = (256 - 224) // 2
        top = (256 - 224) // 2
        image = image.crop((left, top, left + 224, top + 224))
        img = np.array(image, dtype=np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img = (img - mean) / std
        img = img.transpose(2, 0, 1)
        return np.expand_dims(img, axis=0).astype(np.float32)

    def predict(self, image, top_k=5):
        tensor = self.preprocess_image(image)
        start = time.time()
        outputs = self.session.run([self.output_name], {self.input_name: tensor})
        inference_time = time.time() - start
        logits = outputs[0][0]
        probs = self.softmax(logits)
        top_idx = np.argsort(probs)[::-1][:top_k]
        results = [
            {"class": self.class_names[i], "confidence": float(probs[i]), "class_id": int(i)}
            for i in top_idx
        ]
        return results, inference_time

    @staticmethod
    def softmax(x):
        x = x.astype(np.float32)
        e = np.exp(x - np.max(x))
        return e / np.sum(e)


@st.cache_resource
def load_classifier():
    return Food101Classifier()


def display_prediction_results(results, inference_time):
    top_pred = results[0]
    st.markdown(f"""
    <div class="prediction-box">
        <h2>Prediksi Teratas</h2>
        <h1 style="color: #28a745; margin: 0;">{top_pred['class'].replace('_', ' ').title()}</h1>
        <h3 style="color: #6c757d; margin: 0;">Confidence: {top_pred['confidence']:.1%}</h3>
        <p style="margin-top: 1rem;">Inference time: {inference_time*1000:.1f} ms</p>
    </div>
    """, unsafe_allow_html=True)

    classes = [r['class'].replace('_', ' ').title() for r in results]
    confidences = [r['confidence'] * 100 for r in results]
    fig = px.bar(
        x=confidences,
        y=classes,
        orientation='h',
        title="Confidence Scores (%)",
        color=confidences,
        color_continuous_scale='Viridis'
    )
    fig.update_layout(showlegend=False, height=300, yaxis={'categoryorder': 'total ascending'})
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Detail"):
        df = pd.DataFrame([
            {
                'Rank': i + 1,
                'Food Class': r['class'].replace('_', ' ').title(),
                'Confidence': f"{r['confidence']:.1%}",
                'Raw Score': f"{r['confidence']:.4f}"
            }
            for i, r in enumerate(results)
        ])
        st.dataframe(df, use_container_width=True, hide_index=True)

st.markdown('<h1 class="main-header">Klasifikasi 101 Makanan</h1>', unsafe_allow_html=True)
st.markdown('<h5 class="main-header">dengan model ResNet50</h5>', unsafe_allow_html=True)
st.markdown('<h5 class="main-header">(masih dalam tahap pengembangan)</h5>', unsafe_allow_html=True)

st.image(
        "https://storage.googleapis.com/kaggle-datasets-images/2918922/5029790/277a9147ec4854e4762767c8bd107bec/dataset-cover.png?t=2023-02-20-08-37-14",
        caption="Food-101 Dataset (Bossard et al., 2014)",
        use_container_width=True
    )

with st.sidebar:
    classifier = load_classifier()
    st.markdown(f"""
    <div class="sidebar-info">
        <b>Model:</b> ResNet50<br>
        <b>Kelas:</b> 101<br>
        <b>Ukuran input:</b> 224×224<br>
        <b>Runtime:</b> ONNX Runtime<br>
    </div>
    """, unsafe_allow_html=True)
    providers = classifier.session.get_providers()
    st.success(f"Model loaded ({'GPU' if 'CUDAExecutionProvider' in providers else 'CPU'})")
    st.info(f"Input type: {classifier.input_type}")
    top_k = st.slider("Top-K Predictions", 1, 10, 5)
    show_original = st.checkbox("Tampilkan gambar asli", True)

    st.markdown("## List Makanan")

    cols = st.columns(13)

    available_letters = sorted(set(name[0].upper() for name in CLASS_NAMES))

    selected = st.session_state.get("selected_letter", None)

    for i, letter in enumerate(available_letters):
        if cols[i % 13].button(letter):
            st.session_state["selected_letter"] = letter
            selected = letter

    if selected:
        filtered = [name for name in CLASS_NAMES if name[0].upper() == selected]
    else:
        filtered = CLASS_NAMES

    st.markdown(
        "<div style='height:400px; overflow-y:auto;'>"
        + "<br>".join(f"- {name.replace('_',' ').title()}" for name in filtered)
        + "</div>",
        unsafe_allow_html=True
    )


st.markdown("## Upload gambar makanan")
uploaded_file = st.file_uploader("Pilih gambar...", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown("### Input Image")
        if show_original:
            st.image(image, caption="Ukuran Asli", use_container_width=True)
        else:
            st.image(image.convert('RGB').resize((224, 224)), caption="Ukuran untuk di-input", use_container_width=True)
    with col2:
        if st.button("Mulai klasifikasi", type="primary"):
            results, inference_time = classifier.predict(image, top_k=top_k)
            display_prediction_results(results, inference_time)
    with st.expander("Informasi Gambar"):
        st.write(f"**Format:** {image.format}")
        st.write(f"**Ukuran:** {image.size}")
        st.write(f"**Mode:** {image.mode}")
else:
    with st.expander("Asal-Usul Dataset"):
        st.write("""
        Dataset Food-101 adalah tolak ukur yang signifikan dalam bidang pengenalan gambar makanan. Dataset ini diperkenalkan oleh Lukas Bossard, Matthieu Guillaumin, dan Luc Van Gool pada tahun 2014 untuk mengatasi kurangnya basis data publik yang menantang dan realistis untuk tugas pengenalan makanan. Total dataset terdiri dari 101.000 gambar yang terbagi dalam 101 kategori makanan berbeda.
        """)

    with st.expander("Format yang bisa di-upload"):
        st.write("""
        JPG/JPEG atau PNG
        """)
            
    main()



