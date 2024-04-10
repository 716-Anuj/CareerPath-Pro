import numpy as np
import re
import streamlit as st
from pypdf import PdfReader
import pickle
import pandas as pd
import time
from streamlit_extras.let_it_rain import rain



def example():
    rain(
        emoji="✨",
        font_size=15,
        falling_speed=5.9,
        animation_length="infinite",
    )
example()
st.title("Career Path Predictor")


def cleanResume(text):
    cleanText = re.sub("http\S+\s"," ", text)
    cleanText = re.sub("@\S+"," ", cleanText)
    cleanText = re.sub("#\S+\s","", cleanText)
    cleanText = re.sub("RT|cc","", cleanText)
    cleanText = re.sub("[%s]"%re.escape("""!"#$%&'()*+-,/:;<=>?@[\]^_`{|}~""")," ", cleanText)
    cleanText = re.sub(r'[^\x00-x\7f]'," ", cleanText)
    cleanText = re.sub("\s+"," ", cleanText)
    return cleanText

# Load the trained classifier
with open("best_model_1.pkl", "rb") as fp:
    model = pickle.load(fp)
    clf = model["clf"]
    tfidf = model["tfidf"]
    categories = model["le"]

uploaded_file = st.file_uploader("Choose a Resume", type=['pdf'])
if uploaded_file is not None:
    # To read file as bytes:
    # bytes_data = uploaded_file.read()
    reader = PdfReader(uploaded_file)
    page = reader.pages[0]
    myresume = page.extract_text()
    cleaned_resume = cleanResume(myresume)

    progress_bar = st.progress(0)
    for perc_completed in range(100):
        time.sleep(0.02)
        progress_bar.progress(perc_completed+1)

    st.success("Pdf uploaded Successfully")

    # Transform the cleaned resume using the trained TfidVectorizer
    input_features = tfidf.transform([cleaned_resume])
    preds = clf.predict_proba(input_features)[0]
    pred_idx = [(p, i) for i, p in enumerate(preds)]
    sort_pred_idx = sorted(pred_idx, key=lambda x: x[0], reverse=True)

    st.write("### Prediction according to our Dataset")

    # st.bar_chart(data=[(p, categories[i]) for p, i in sort_pred_idx])
    st.bar_chart(pd.DataFrame({
        "x": list(map(lambda x: categories[x[1]], sort_pred_idx)),
        "y": list(map(lambda x: x[0], sort_pred_idx))
    }), x="x", y="y")

    st.write("### Top 5 Predicted Career Path ")
    for prob, i in sort_pred_idx[:5]:
        st.write(f"{categories[i]}: {prob * 100:.2f}%")