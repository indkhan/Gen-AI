import os
import re
import difflib
import base64
import yaml
import groq
import streamlit as st
import PyPDF2
from fpdf import FPDF
from pathlib import Path
from tenacity import retry, stop_after_attempt, wait_exponential
from sentence_transformers import SentenceTransformer, util


# Configuration
class Config:
    TEMP_DIR = Path("temp_files")
    OUTPUT_DIR = Path("output")
    MAX_ADDITIONS = 5
    MAX_REMOVALS = 3
    SEMANTIC_SIMILARITY_THRESHOLD = 0.75


# Initialize environment
Config.TEMP_DIR.mkdir(exist_ok=True, parents=True)
Config.OUTPUT_DIR.mkdir(exist_ok=True, parents=True)


class PDFProcessor:
    """Handles PDF input/output with validation"""

    @staticmethod
    def pdf_to_text(file_path):
        try:
            text = ""
            with open(file_path, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                for page in reader.pages:
                    text += page.extract_text() + "\n"
            return text.strip()
        except Exception as e:
            raise ValueError(f"PDF read error: {str(e)}")

    @staticmethod
    def text_to_pdf(text, output_path):
        try:
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", size=11)
            pdf.multi_cell(0, 6, text)
            pdf.output(output_path)
        except Exception as e:
            raise ValueError(f"PDF creation error: {str(e)}")


class ContentValidator:
    """Ensures all content remains grounded in source material"""

    def __init__(self, cv_text, personal_data):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.source_embeddings = self.model.encode(personal_data + cv_text)

    def validate_addition(self, new_content):
        new_embeddings = self.model.encode(new_content)
        similarity = util.cos_sim(self.source_embeddings, new_embeddings)
        if similarity.max() < Config.SEMANTIC_SIMILARITY_THRESHOLD:
            raise ValueError("Addition contains non-source material")


class AIEditor:
    """Handles AI-powered document editing"""

    def __init__(self):
        self.client = groq.Groq(api_key=os.getenv("GROQ_API_KEY"))

    @retry(
        stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    def optimize_cv(self, cv_text, job_desc, personal_data):
        prompt = f"""Optimize this CV for the job description below.
        ONLY use information from the CV and personal data.
        ALLOWED:
        - Reordering sections
        - Rephrasing existing content
        - Removing irrelevant information
        PROHIBITED:
        - Adding new facts
        - Changing numbers/metrics
        - Inventing experiences
        
        Personal Data: {personal_data[:2000]}
        Job Description: {job_desc[:2000]}
        Original CV: {cv_text[:4000]}
        """

        response = self.client.chat.completions.create(
            model="mixtral-8x7b-32768",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )
        return response.choices[0].message.content


class ApplicationBuilder:
    """Orchestrates the document processing workflow"""

    def __init__(self):
        self.validator = None
        self.ai_editor = AIEditor()

    def build(self, cv_path, job_desc, personal_data, adjustments):
        # Read and validate inputs
        cv_text = PDFProcessor.pdf_to_text(cv_path)
        self.validator = ContentValidator(cv_text, personal_data)

        # Apply AI optimization
        optimized_cv = self.ai_editor.optimize_cv(cv_text, job_desc, personal_data)

        # Validate and apply manual adjustments
        final_cv = self._apply_adjustments(optimized_cv, adjustments)

        # Generate output PDF
        output_path = Config.OUTPUT_DIR / "optimized_application.pdf"
        PDFProcessor.text_to_pdf(final_cv, output_path)

        return output_path, cv_text, optimized_cv

    def _apply_adjustments(self, content, adjustments):
        modified = content
        for add in adjustments.get("add", [])[: Config.MAX_ADDITIONS]:
            self.validator.validate_addition(add)
            modified += f"\n{add}"

        for remove in adjustments.get("remove", [])[: Config.MAX_REMOVALS]:
            modified = modified.replace(remove, "")

        return modified


class StreamlitInterface:
    """Provides the user interface"""

    def __init__(self):
        self.builder = ApplicationBuilder()

    def run(self):
        st.set_page_config(page_title="CV Optimizer Pro", layout="wide")
        st.title("Professional CV Optimization Suite")

        with st.sidebar:
            st.header("Upload Documents")
            cv_file = st.file_uploader("Current CV (PDF)", type=["pdf"])
            personal_file = st.file_uploader("Personal Data (TXT)", type=["txt"])
            job_desc = st.text_area("Job Description", height=150)
            adjustments_file = st.file_uploader("Adjustments (YAML)", type=["yaml"])

            if st.button("Optimize Documents", use_container_width=True):
                self.process_documents(
                    cv_file, personal_file, job_desc, adjustments_file
                )

    def process_documents(self, cv_file, personal_file, job_desc, adjustments_file):
        try:
            with st.spinner("Processing documents..."):
                # Save uploaded files
                cv_path = self._save_uploaded_file(cv_file, Config.TEMP_DIR)
                personal_data = personal_file.getvalue().decode()
                adjustments = (
                    yaml.safe_load(adjustments_file) if adjustments_file else {}
                )

                # Process documents
                output_path, original, optimized = self.builder.build(
                    cv_path, job_desc, personal_data, adjustments
                )

                # Display results
                self._show_results(output_path, original, optimized)

        except Exception as e:
            st.error(f"Processing error: {str(e)}")

    def _show_results(self, output_path, original, optimized):
        st.success("Document optimization complete!")

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Original CV")
            st.text(original[:2000] + "...")
        with col2:
            st.subheader("Optimized CV")
            st.text(optimized[:2000] + "...")

        st.download_button(
            "Download Enhanced Application",
            data=open(output_path, "rb").read(),
            file_name="enhanced_application.pdf",
            mime="application/pdf",
        )

        st.subheader("Modification Analysis")
        self._show_diff(original, optimized)

    def _show_diff(self, original, modified):
        diff = difflib.ndiff(
            original.splitlines(keepends=True), modified.splitlines(keepends=True)
        )
        st.text("".join(diff))

    def _save_uploaded_file(self, file, directory):
        if file is None:
            raise ValueError("No file uploaded")

        save_path = directory / file.name
        with open(save_path, "wb") as f:
            f.write(file.getbuffer())
        return save_path


if __name__ == "__main__":
    StreamlitInterface().run()
