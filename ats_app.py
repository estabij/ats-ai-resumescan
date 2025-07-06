import streamlit as st
from openai import OpenAI
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import docx2txt
import PyPDF2
import os
from dotenv import load_dotenv

# Load .env file
load_dotenv()
# Access the API key securely
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)
# Streamlit app UI
st.title("📄 ATS Resume Checker")
st.write(
    "Upload your resume (.docx or .pdf) and paste a job description below. "
    "The app will simulate how an ATS and GPT-4 might evaluate your fit."
)
uploaded_file = st.file_uploader("Upload your resume (.docx or .pdf)", type=["pdf", "docx"])
job_desc = st.text_area("Paste job description here")
resume = ""
# Resume text extraction
if uploaded_file:
    if uploaded_file.name.endswith(".pdf"):
        pdf_reader = PyPDF2.PdfReader(uploaded_file)
        for page in pdf_reader.pages:
            resume += page.extract_text() or ""
    elif uploaded_file.name.endswith(".docx"):
        resume = docx2txt.process(uploaded_file)
# Main logic on button press
if st.button("Check ATS Match") and resume and job_desc:
    # Cosine Similarity Scoring
    vectorizer = TfidfVectorizer(stop_words="english")
    vectors = vectorizer.fit_transform([resume, job_desc])
    cos_sim = cosine_similarity(vectors[0:1], vectors[1:2])[0][0] * 100
    # GPT-4 Resume Evaluation Prompt
    prompt = (
        "Evaluate the following resume against the job description. "
        "Give a score out of 100, a short rationale, and improvement suggestions if any.\n\n"
        f"Resume:\n{resume}\n\nJob Description:\n{job_desc}"
    )
    # GPT-4 API Call
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )
    evaluation = response.choices[0].message.content
    passed = cos_sim >= 60
    # Output Section
    st.subheader("Results")
    st.write(f"Cosine Similarity Score: {cos_sim:.2f} %")
    st.markdown("**GPT-4 Evaluation:**")
    st.write(evaluation)
    st.markdown(f"### {'✅ PASS' if passed else '❌ FAIL'}")
    st.markdown("---")
    # Improved Resume Generator
    st.subheader("📝 Generate ATS-Optimized Resume")
    if st.button("Generate Improved Resume"):
        improve_prompt = f"""
You are an expert resume coach and editor. Rewrite the following resume to be optimized for the job description below.
- Incorporate missing skills, tools, or responsibilities based on the job.
- Keep original experiences factual but improve alignment.
- Use clear, ATS-friendly formatting and job-aligned language.
- Do not invent roles or exaggerate.
Resume:
{resume}
Job Description:
{job_desc}
"""
        improved_response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": improve_prompt}],
            temperature=0.4
        )
        improved_resume = improved_response.choices[0].message.content
        st.subheader("Improved Resume (ATS-Optimized)")
        st.code(improved_resume, language="markdown")
        st.download_button(
            label="📥 Download Improved Resume",
            data=improved_resume,
            file_name="Improved_Resume.txt",
            mime="text/plain"
        )
