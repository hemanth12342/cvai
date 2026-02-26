import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from groq import Groq

# ---------------------------
# 🔑 ENTER YOUR GEMINI API KEY HERE
# ---------------------------
GROQ_API_KEY = "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"

client = Groq(api_key=GROQ_API_KEY)


# ---------------------------
# 🎯 AI Resume Generator
# ---------------------------
def generate_resume(name, skills, projects, experience, education,certifications):
    prompt = f"""
    Create a professional ATS-friendly resume.

    Name: {name}
    Skills: {skills}
    Projects: {projects}
    Experience: {experience}
    Education: {education}
    certifications: {certifications}

    Make it clean, structured and professional.
    """

    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7
    )

    return response.choices[0].message.content


# ---------------------------
# 💌 AI Cover Letter Generator
# ---------------------------
def generate_cover_letter(name, job_role, skills):
    prompt = f"""
    Write a professional cover letter for {job_role}.
    Candidate Name: {name}
    Skills: {skills}
    Make it concise and impactful.
    """

    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7
    )

    return response.choices[0].message.content


# ---------------------------
# 🌐 Portfolio Builder
# ---------------------------
def generate_portfolio(name, skills, projects, experience="", education="", certifications=""):
    """Generate a professional portfolio HTML with enhanced styling and layout."""

    skills_list = [s.strip() for s in skills.split(",") if s.strip()]
    skills_html = "".join([
        f'<span class="skill-badge">{skill}</span>'
        for skill in skills_list
    ])

    projects_list = [p.strip() for p in projects.split("\n") if p.strip()]
    projects_html = "".join([
        f'<div class="project-item"><strong>• {proj}</strong></div>'
        for proj in projects_list
    ])

    certifications_list = [c.strip() for c in certifications.split("\n") if c.strip()]
    certifications_html = "".join([
        f'<div class="project-item"><strong>🏆 {cert}</strong></div>'
        for cert in certifications_list
    ])

    html_template = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>{name} | Professional Portfolio</title>
        <style>
            * {{
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }}
            body {{
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: #333;
                line-height: 1.6;
                padding: 20px;
            }}
            .container {{
                max-width: 900px;
                margin: 0 auto;
                background: white;
                border-radius: 12px;
                box-shadow: 0 10px 40px rgba(0,0,0,0.2);
                overflow: hidden;
            }}
            .header {{
                background: linear-gradient(135deg, #2c3e50 0%, #34495e 100%);
                color: white;
                padding: 60px 40px;
                text-align: center;
            }}
            .header h1 {{
                font-size: 2.5em;
                margin-bottom: 10px;
                font-weight: 700;
            }}
            .header p {{
                font-size: 1.1em;
                opacity: 0.9;
            }}
            .content {{
                padding: 40px;
            }}
            .section {{
                margin-bottom: 40px;
            }}
            .section h2 {{
                font-size: 1.8em;
                color: #2c3e50;
                border-bottom: 3px solid #667eea;
                padding-bottom: 12px;
                margin-bottom: 20px;
                font-weight: 700;
            }}
            .skill-badge {{
                display: inline-block;
                background: #667eea;
                color: white;
                padding: 8px 16px;
                border-radius: 25px;
                margin: 6px 8px 6px 0;
                font-size: 0.95em;
                font-weight: 600;
                transition: all 0.3s;
            }}
            .skill-badge:hover {{
                background: #764ba2;
                transform: translateY(-2px);
                box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
            }}
            .project-item {{
                background: #f8f9fa;
                padding: 16px;
                margin-bottom: 12px;
                border-left: 4px solid #667eea;
                border-radius: 4px;
                transition: all 0.3s;
            }}
            .project-item:hover {{
                background: #e8ecff;
                transform: translateX(4px);
            }}
            .footer {{
                background: #f8f9fa;
                padding: 20px 40px;
                text-align: center;
                color: #666;
                border-top: 1px solid #ddd;
            }}
            .footer p {{
                font-size: 0.9em;
                margin: 5px 0;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>{name}</h1>
                <p>Professional Portfolio</p>
            </div>
            <div class="content">
                <div class="section">
                    <h2>💡 Skills & Expertise</h2>
                    <div>{skills_html if skills_html else '<p style="color:#999;">No skills listed</p>'}</div>
                </div>
                <div class="section">
                    <h2>🚀 Projects & Achievements</h2>
                    <div>{projects_html if projects_html else '<p style="color:#999;">No projects listed</p>'}</div>
                </div>
                {f'<div class="section"><h2>📚 Experience</h2><p>{experience}</p></div>' if experience else ''}
                {f'<div class="section"><h2>🎓 Education</h2><p>{education}</p></div>' if education else ''}
                {f'<div class="section"><h2>🏆 Certifications</h2><div>{certifications_html if certifications_html else "<p style=\"color:#999;\">No certifications listed</p>"}</div></div>' if certifications else ''}
            </div>
            <div class="footer">
                <p><strong>Generated Portfolio</strong></p>
                <p>This portfolio was generated using AI Resume & Portfolio Builder</p>
            </div>
        </div>
    </body>
    </html>
    """
    return html_template





# ---------------------------
# 🤖 Job Match Score (ML)
# ---------------------------
def job_match_score(resume_text, job_description):
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([resume_text, job_description])

    score = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]
    return round(score * 100, 2)


# ---------------------------
# 🌟 Streamlit UI
# ---------------------------
st.set_page_config(page_title="AI Resume & Portfolio Builder", layout="wide")
st.title("🚀 AI Resume & Portfolio Builder ")

# Initialize session state to persist data after download
if "resume" not in st.session_state:
    st.session_state.resume = None
if "cover_letter" not in st.session_state:
    st.session_state.cover_letter = None
if "portfolio" not in st.session_state:
    st.session_state.portfolio = None

name = st.text_input("Full Name")
skills = st.text_area("Skills (comma separated)")
projects = st.text_area("Projects")
experience = st.text_area("Experience")
education = st.text_area("Education")
certifications = st.text_area("Certifications ")
job_role = st.text_input("Target Job Role")
job_description = st.text_area("Paste Job Description (Optional)")

if st.button("Generate Resume & Portfolio"):
    # Generate and store in session state
    st.session_state.resume = generate_resume(name, skills, projects, experience, education,certifications)
    st.session_state.cover_letter = generate_cover_letter(name, job_role, skills)
    st.session_state.portfolio = generate_portfolio(name, skills, projects, experience, education,certifications)

# Display saved content if it exists
if st.session_state.resume:
    st.subheader("📄 AI Resume")
    st.write(st.session_state.resume)

    st.subheader("💌 AI Cover Letter")
    st.write(st.session_state.cover_letter)

    st.subheader("🌐 Portfolio Preview")
    st.components.v1.html(st.session_state.portfolio, height=400)

    # ----------
    # Downloads Section
    # ----------
    st.divider()
    st.subheader("📥 Download Your Files")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.download_button(
            "📄 Resume (CV)",
            st.session_state.resume,
            file_name="AI_Resume.txt",
            mime="text/plain",
        )

    with col2:
        st.download_button(
            "📝 Cover Letter",
            st.session_state.cover_letter,
            file_name="AI_Cover_Letter.txt",
            mime="text/plain",
        )

    with col3:
        st.download_button(
            "🌐 Portfolio",
            st.session_state.portfolio,
            file_name="portfolio.html",
            mime="text/html",
        )

    # ML Job Matching Score
    