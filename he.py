import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from groq import Groq
import os
from dotenv import load_dotenv

# ---------------------------
# 🔑 LOAD API KEY
# ---------------------------
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    st.error("❌ GROQ_API_KEY not found. Add it in .env file")
    st.stop()

client = Groq(api_key=GROQ_API_KEY)

# ---------------------------
# 🎯 AI Resume Generator
# ---------------------------
def generate_resume(name, skills, projects, experience, education, certifications):

    prompt = f"""
    Create a professional ATS-friendly resume.

    Name: {name}
    Skills: {skills}
    Projects: {projects}
    Experience: {experience}
    Education: {education}
    Certifications: {certifications}

    Format properly with headings.
    """

    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
    )

    return response.choices[0].message.content


# ---------------------------
# 💌 Cover Letter Generator
# ---------------------------
def generate_cover_letter(name, job_role, skills):

    prompt = f"""
    Write a professional cover letter for the role of {job_role}.
    Candidate Name: {name}
    Skills: {skills}
    Keep it concise and impactful.
    """

    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
    )

    return response.choices[0].message.content


# ---------------------------
# 🌐 Portfolio Generator
# ---------------------------
def generate_portfolio(name, skills, projects):

    html = f"""
    <html>
    <head>
        <title>{name} Portfolio</title>
        <style>
            body {{
                font-family: Arial;
                background: #f4f4f4;
                padding: 40px;
            }}
            .card {{
                background: white;
                padding: 30px;
                border-radius: 10px;
                box-shadow: 0 5px 15px rgba(0,0,0,0.1);
            }}
            h1 {{
                color: #4A90E2;
            }}
        </style>
    </head>
    <body>
        <div class="card">
            <h1>{name}</h1>
            <h3>Skills</h3>
            <p>{skills}</p>
            <h3>Projects</h3>
            <p>{projects}</p>
        </div>
    </body>
    </html>
    """

    return html


# ---------------------------
# 🤖 Job Match Score (ML)
# ---------------------------
def job_match_score(resume_text, job_description):
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([resume_text, job_description])
    score = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]
    return round(score * 100, 2)


# ---------------------------
# 🌟 STREAMLIT UI
# ---------------------------
st.set_page_config(page_title="AI Resume Builder", layout="wide")
st.title("🚀 AI Resume & Portfolio Builder")

name = st.text_input("Full Name")
skills = st.text_area("Skills (comma separated)")
projects = st.text_area("Projects")
experience = st.text_area("Experience")
education = st.text_area("Education")
certifications = st.text_area("Certifications")
job_role = st.text_input("Target Job Role")
job_description = st.text_area("Paste Job Description (Optional)")

if st.button("Generate Resume & Portfolio"):

    resume = generate_resume(name, skills, projects, experience, education, certifications)
    cover_letter = generate_cover_letter(name, job_role, skills)
    portfolio = generate_portfolio(name, skills, projects)

    st.subheader("📄 Resume")
    st.write(resume)

    st.subheader("💌 Cover Letter")
    st.write(cover_letter)

    st.subheader("🌐 Portfolio Preview")
    st.components.v1.html(portfolio, height=400)

    # Job Match Score
    if job_description:
        score = job_match_score(resume, job_description)
        st.success(f"🎯 Job Match Score: {score}%")

    # Download Buttons
    st.download_button("Download Resume", resume, "resume.txt")
    st.download_button("Download Cover Letter", cover_letter, "cover_letter.txt")
    st.download_button("Download Portfolio", portfolio, "portfolio.html")