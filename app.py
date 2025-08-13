# app.py

import streamlit as st
from PIL import Image
import os
from src.query_outfits import search_outfits, generate_explanations_with_llm


st.set_page_config(page_title="Outfit Recommender", layout="wide")

st.title("🧅 Outfit Search Recommender")
st.write("Enter a fashion-related prompt to get outfit recommendations with visual explanations.")

query = st.text_input("Enter your fashion query", "romantic spring date night outfit")

if st.button("Search"):
    with st.spinner("Searching and generating explanations..."):
        results = search_outfits(query)
        results = generate_explanations_with_llm(query, results)

    st.subheader("Results")
    cols = st.columns(len(results))

    for col, result in zip(cols, results):
        if os.path.exists(result["image_path"]):
            image = Image.open(result["image_path"])
            col.image(image, caption=f"\n**{result['explanation']}**\n\n(Similarity: {result['similarity_score']:.3f})", use_column_width=True)
        else:
            col.error(f"Image not found: {result['image_path']}")
