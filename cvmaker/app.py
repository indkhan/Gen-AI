import streamlit as st
import groq
import os
import base64
import shutil
import PyPDF2
from pathlib import Path
from tenacity import retry, stop_after_attempt, wait_exponential

# Set page config
st.set_page_config(
    page_title="AI Job Application Customizer", page_icon="📄", layout="centered"
)


class GroqApp:
    def __init__(self):
        self.temp_dir = Path("groq_temp")
        self._validate_env()
        self.client = groq.Groq(api_key=os.getenv("GROQ_API_KEY"))
        self._setup_directory()

    def _validate_env(self):
        """Validate required environment variables"""
        if not os.getenv("GROQ_API_KEY"):
            st.error("❌ Missing GROQ_API_KEY in environment variables")
            st.stop()

    def _setup_directory(self):
        """Create and clean working directory"""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
        self.temp_dir.mkdir(exist_ok=True)

    def run(self):
        st.title("🚀 Groq-Powered Job Application Customizer")

        # File upload section
        cv_file = st.file_uploader("Upload CV (PDF)", type=["pdf"])
        cl_file = st.file_uploader("Upload Cover Letter (PDF)", type=["pdf"])
        job_desc = st.text_area("Paste Job Description", height=200)

        if st.button("✨ Generate Documents", use_container_width=True):
            if cv_file and cl_file and job_desc:
                with st.spinner("Analyzing and customizing..."):
                    try:
                        self._process_documents(cv_file, cl_file, job_desc)
                    except Exception as e:
                        st.error(f"🚨 Error: {str(e)}")
            else:
                st.warning("⚠️ Please upload all files and enter job description")

    def _process_documents(self, cv_file, cl_file, job_desc):
        """Main processing workflow"""
        # Save uploaded files
        cv_path = self._save_file(cv_file)
        cl_path = self._save_file(cl_file)
        jd_path = self.temp_dir / "job_description.txt"

        with open(jd_path, "w") as f:
            f.write(job_desc)

        # Analyze job description
        with st.spinner("🔍 Analyzing job requirements..."):
            analysis = self._analyze_jd(jd_path)

        # Customize CV
        with st.spinner("📄 Optimizing CV..."):
            cv_text = self._read_pdf(cv_path)
            custom_cv = self._customize_cv(cv_text, analysis)

        # Customize Cover Letter
        with st.spinner("📝 Crafting cover letter..."):
            cl_text = self._read_pdf(cl_path)
            custom_cl = self._customize_cl(cl_text, analysis)

        # Display results
        self._show_results(custom_cv, custom_cl, analysis)

    @retry(
        stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    def _analyze_jd(self, jd_path):
        """Analyze job description with Groq"""
        with open(jd_path, "r") as f:
            jd_text = f.read()

        response = self.client.chat.completions.create(
            model="mixtral-8x7b-32768",
            messages=[
                {
                    "role": "user",
                    "content": f"""Analyze this job description and extract:
                - Required skills
                - Key qualifications
                - Industry keywords
                - Company culture cues
                - Success metrics
                
                {jd_text[:3000]}""",
                }
            ],
            temperature=0.2,
        )
        return response.choices[0].message.content

    def _customize_cv(self, cv_text, analysis):
        """Customize CV with Groq"""
        response = self.client.chat.completions.create(
            model="mixtral-8x7b-32768",
            messages=[
                {
                    "role": "user",
                    "content": f"""Customize this CV based on job analysis:
                {analysis}
                
                Original CV:
                {cv_text[:3000]}
                
                Focus on:
                - Matching required skills
                - Adding relevant metrics
                - Using industry keywords""",
                }
            ],
            temperature=0.3,
        )
        return response.choices[0].message.content

    def _customize_cl(self, cl_text, analysis):
        """Customize cover letter with Groq"""
        response = self.client.chat.completions.create(
            model="mixtral-8x7b-32768",
            messages=[
                {
                    "role": "user",
                    "content": f"""Create a cover letter using this analysis:
                {analysis}
                
                Original Letter:
                {cl_text[:2000]}
                
                Include:
                - Value proposition
                - Cultural alignment
                - Specific examples""",
                }
            ],
            temperature=0.4,
        )
        return response.choices[0].message.content

    def _show_results(self, cv_content, cl_content, analysis):
        """Display results and download options"""
        st.success("✅ Documents Ready!")

        # Create download buttons
        col1, col2 = st.columns(2)
        with col1:
            st.download_button(
                label="📥 Download CV",
                data=cv_content,
                file_name="custom_cv.txt",
                mime="text/plain",
            )
        with col2:
            st.download_button(
                label="📥 Download Cover Letter",
                data=cl_content,
                file_name="custom_cl.txt",
                mime="text/plain",
            )

        # Show analysis summary
        with st.expander("🔍 View Job Analysis"):
            st.write(analysis)

    def _save_file(self, uploaded_file):
        """Save uploaded file to temp directory"""
        save_path = self.temp_dir / uploaded_file.name
        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        return save_path

    def _read_pdf(self, file_path):
        """Read PDF content with error handling"""
        try:
            text = ""
            with open(file_path, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                if len(reader.pages) == 0:
                    raise ValueError("PDF file is empty")
                for page in reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
            return text
        except Exception as e:
            raise ValueError(f"Error reading PDF: {str(e)}") from e


if __name__ == "__main__":
    GroqApp().run()
