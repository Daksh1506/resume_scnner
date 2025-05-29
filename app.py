import streamlit as st
import fitz  # PyMuPDF
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nltk.download('punkt')

def extract_text_from_pdf(uploaded_file):
    pdf_doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    text = ""
    for page in pdf_doc:
        text += page.get_text()
    return text

def calculate_similarity(resume_text, jd_text):
    texts = [resume_text, jd_text]
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(texts)
    score = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]
    return round(score * 100, 2)

st.title("🧠 AI Resume Scanner Bot")
st.write("Upload your resume and paste the job description below to see how well it matches.")

uploaded_file = st.file_uploader("Upload your resume (PDF)", type="pdf")
job_desc = st.text_area("Paste Job Description here")

if uploaded_file and job_desc:
    resume_text = extract_text_from_pdf(uploaded_file)
    match_score = calculate_similarity(resume_text, job_desc)
    
    st.markdown(f"### ✅ Match Score: **{match_score}%**")

    if match_score > 80:
        st.success("Great match! Your resume fits this job well.")
    elif match_score > 50:
        st.info("Decent match. Consider adding some more relevant skills.")
    else:
        st.warning("Low match. You may want to tailor your resume better.")
