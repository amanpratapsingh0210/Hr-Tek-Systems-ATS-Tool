import streamlit as st

# Page config must be the first Streamlit command
st.set_page_config(page_title="ATS Resume Checker", page_icon="���", layout="wide")

import PyPDF2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import io
import requests
import re
from urllib.parse import urlparse, parse_qs

# OCR imports
try:
    import pytesseract
    from pdf2image import convert_from_bytes
    from PIL import Image
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False

@st.cache_resource
def downloadNltkData():
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('punkt_tab', quiet=True)
    nltk.download('omw-1.4', quiet=True)
    
downloadNltkData()

# Predefined Job Descriptions
PREDEFINED_JOB_DESCRIPTIONS = {
    "Cloud / DevOps Intern (AWS Focused)": """Cloud / DevOps Intern (AWS Focused)
About the Role: Help HR-Tek optimize cloud usage and deploy the product on AWS with scalability, monitoring, and automation in mind.
Responsibilities:
Deploy staging and production environments on AWS.
Set up auto-scaling groups, load balancers, and monitoring (CloudWatch).
Work on CI/CD pipelines using AWS CodePipeline / Jenkins.
Optimize AWS credits for cost efficiency.
Preferred Skills: AWS EC2, S3, RDS, VPC, IAM, CloudFormation/Terraform, Linux basics.""",

    "Full Stack Developer": """Full Stack Developer
We are looking for a Full Stack Developer to join our development team.
Responsibilities:
Develop and maintain web applications using modern frameworks.
Design and implement RESTful APIs.
Work with databases (SQL and NoSQL).
Collaborate with frontend and backend teams.
Implement responsive web designs.
Required Skills: JavaScript, React, Node.js, Python, SQL, Git, REST APIs.
Preferred: Experience with cloud platforms (AWS/Azure), Docker, CI/CD.""",

    "Data Science & Analytics Intern": """
Python (Pandas, NumPy, Scikit-learn), SQL, PowerBI, Tableau,
statistics, data visualization.""",

    "Software Engineer (Backend)": """Software Engineer (Backend)
We need a backend engineer to build scalable server-side applications.
Responsibilities:
Design and develop backend services and APIs.
Optimize database performance and queries.
Implement security best practices.
Work with microservices architecture.
Write unit tests and documentation.
Required Skills: Java, Python, Spring Boot, SQL, REST APIs, Git.
Preferred: Experience with Docker, Kubernetes, AWS, message queues.""",

    "Frontend Developer (React)": """Frontend Developer (React)
Join our frontend team to create amazing user experiences.
Responsibilities:
Develop responsive web applications using React.
Implement modern UI/UX designs.
Optimize application performance.
Work with state management libraries (Redux/Context).
Collaborate with designers and backend developers.
Required Skills: JavaScript, React, HTML5, CSS3, Git, REST APIs.
Preferred: Experience with TypeScript, Next.js, testing frameworks, CI/CD.""",

    "Product Manager": """Product Manager
Lead product development and strategy for our platform.
Responsibilities:
Define product roadmap and feature priorities.
Gather and analyze user requirements.
Coordinate with engineering, design, and marketing teams.
Conduct market research and competitive analysis.
Manage product launches and iterations.
Required Skills: Product Strategy, User Research, Agile/Scrum, Analytics, Communication.
Preferred: Experience with B2B SaaS, technical background, MBA.""",

    "Custom Job Description": ""
}

def extract_file_id_from_gdrive_url(url):
    """Extract file ID from Google Drive URL"""
    patterns = [
        r'/file/d/([a-zA-Z0-9-_]+)',
        r'id=([a-zA-Z0-9-_]+)',
        r'/d/([a-zA-Z0-9-_]+)'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None

def download_file_from_gdrive(file_id):
    """Download file from Google Drive using file ID"""
    try:
        # Use the direct download URL format
        download_url = f"https://drive.google.com/uc?export=download&id={file_id}"
        
        # Set headers to mimic a browser request
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        # First request to get the file
        response = requests.get(download_url, headers=headers, stream=True, allow_redirects=True)
        
        if response.status_code == 200:
            # Check if we got HTML content (Google Drive warning page)
            content_type = response.headers.get('content-type', '').lower()
            
            if 'text/html' in content_type:
                # We got an HTML page, likely a warning page
                # Try to extract the confirm token
                html_content = response.text
                confirm_match = re.search(r'confirm=([a-zA-Z0-9-_]+)', html_content)
                
                if confirm_match:
                    confirm_token = confirm_match.group(1)
                    # Make another request with the confirm token
                    confirm_url = f"https://drive.google.com/uc?export=download&confirm={confirm_token}&id={file_id}"
                    response = requests.get(confirm_url, headers=headers, stream=True, allow_redirects=True)
                    
                    # Check content type again
                    content_type = response.headers.get('content-type', '').lower()
                    if 'text/html' in content_type:
                        return None  # Still getting HTML, file might be restricted
            
            # Check if we have PDF content
            if 'application/pdf' in content_type or file_id in response.content[:100].decode('utf-8', errors='ignore'):
                return response.content
            else:
                # Check if content starts with PDF header
                if response.content.startswith(b'%PDF'):
                    return response.content
                else:
                    return None
        else:
            return None
            
    except Exception as e:
        st.error(f"Error downloading from Google Drive: {e}")
        return None

def extract_text_with_ocr(pdf_content):
    """Extract text from PDF using OCR (for scanned documents)"""
    if not OCR_AVAILABLE:
        return None
    
    try:
        # Convert PDF to images
        images = convert_from_bytes(pdf_content, dpi=300)
        
        extracted_text = ""
        for i, image in enumerate(images):
            # Use OCR to extract text from each page
            page_text = pytesseract.image_to_string(image, lang='eng')
            extracted_text += page_text + "\n"
            
        return extracted_text.strip()
    except Exception as e:
        st.error(f"OCR processing failed: {e}")
        return None

def preprocessText(text):
    """Cleans and preprocesses the input text."""
    if not text:
        return ""
    tokens = word_tokenize(text.lower())
    tokens = [word for word in tokens if word.isalpha()]
    
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    
    return " ".join(tokens)

def extract_text_from_pdf(uploaded_file):
    """Extracts text from an uploaded PDF file."""
    try:
        reader = PyPDF2.PdfReader(uploaded_file)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
        
        # If no text extracted, try OCR
        if not text.strip() and OCR_AVAILABLE:
            st.info("��� No selectable text found. Attempting OCR extraction...")
            uploaded_file.seek(0)  # Reset file pointer
            pdf_content = uploaded_file.read()
            text = extract_text_with_ocr(pdf_content)
            if text:
                st.success("✅ Text extracted using OCR!")
        
        return text.strip() if text else None
        
    except Exception as e:
        st.error(f"Error reading PDF file: {e}")
        return None

def extract_text_from_gdrive_pdf(file_content):
    """Extracts text from PDF content downloaded from Google Drive."""
    try:
        # Check if content is actually a PDF
        if not file_content.startswith(b'%PDF'):
            st.error("Downloaded content is not a valid PDF file. Please check the Google Drive link and sharing permissions.")
            return None
            
        pdf_file = io.BytesIO(file_content)
        reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
        
        # If no text extracted, try OCR
        if not text.strip() and OCR_AVAILABLE:
            st.info("��� No selectable text found. Attempting OCR extraction...")
            text = extract_text_with_ocr(file_content)
            if text:
                st.success("✅ Text extracted using OCR!")
        
        if text.strip():
            return text.strip()
        else:
            if OCR_AVAILABLE:
                st.error("❌ Could not extract text even with OCR. The PDF might be corrupted or have very poor image quality.")
            else:
                st.warning("⚠️ PDF file is empty or contains only images. OCR is not available. Please install pytesseract and pdf2image for scanned PDF support.")
            return None
            
    except Exception as e:
        st.error(f"Error reading PDF from Google Drive: {e}")
        return None
    
def get_keyword_analysis(processed_resume, processed_jd):
    """Identifies keywords present and missing from the resume."""
    jd_words = set(processed_jd.split())
    resume_words = set(processed_resume.split())
    
    found_keywords = list(resume_words.intersection(jd_words))
    missing_keywords = list(jd_words.difference(resume_words))
    
    return found_keywords, missing_keywords

#MAIN APP

st.title("ATS Resume Compatibility Checker")
st.markdown("""
Welcome! This tool helps you check how well your resume matches a job description.
Upload your resume manually or provide a Google Drive link, then select a predefined job description or paste your own.
""")

# Show OCR availability status
if OCR_AVAILABLE:
    st.success("��� OCR support is available for scanned PDFs")
else:
    st.warning("⚠️ OCR support is not available. Install pytesseract and pdf2image for scanned PDF support.")

col1, col2 = st.columns(2)

with col1:
    st.header("Your Resume")
    
    # Choice between manual upload and Google Drive link
    upload_method = st.radio(
        "Choose how to provide your resume:",
        ["Upload File", "Google Drive Link"],
        help="Select whether to upload a file directly or provide a Google Drive link"
    )
    
    resume_text = None
    
    if upload_method == "Upload File":
        uploadedResume = st.file_uploader("Upload your resume in PDF format", type=["pdf"])
        if uploadedResume is not None:
            resume_text = extract_text_from_pdf(uploadedResume)
    else:
        gdrive_url = st.text_input(
            "Google Drive Link:",
            placeholder="https://drive.google.com/file/d/1ABC.../view?usp=sharing",
            help="Paste the Google Drive shareable link to your resume PDF"
        )
        
        if gdrive_url:
            if "drive.google.com" in gdrive_url:
                file_id = extract_file_id_from_gdrive_url(gdrive_url)
                if file_id:
                    with st.spinner('Downloading file from Google Drive...'):
                        file_content = download_file_from_gdrive(file_id)
                        if file_content:
                            resume_text = extract_text_from_gdrive_pdf(file_content)
                        else:
                            st.error("❌ Failed to download file from Google Drive. Please check the link and sharing permissions.")
                else:
                    st.error("❌ Could not extract file ID from the Google Drive URL. Please check the link format.")
            else:
                st.error("❌ Please provide a valid Google Drive URL")

with col2:
    st.header("Job Description")
    
    # Job description selection
    selected_jd = st.selectbox(
        "Choose a predefined job description:",
        options=list(PREDEFINED_JOB_DESCRIPTIONS.keys()),
        help="Select from predefined job descriptions or choose 'Custom Job Description' to paste your own"
    )
    
    # Show the selected job description
    if selected_jd != "Custom Job Description":
        st.subheader("Selected Job Description:")
        st.text_area("", value=PREDEFINED_JOB_DESCRIPTIONS[selected_jd], height=200, disabled=True, key="preview_jd")
        jobDescription = PREDEFINED_JOB_DESCRIPTIONS[selected_jd]
    else:
        st.subheader("Custom Job Description:")
        jobDescription = st.text_area("Paste your custom job description here", height=200, key="custom_jd")
    
if st.button("Analyze Compatibility", type="primary", use_container_width=True):
    if resume_text and jobDescription:
        with st.spinner('Analyzing your documents...'):
            processedResume = preprocessText(resume_text)
            processedJd = preprocessText(jobDescription)
            
            if not processedResume or not processedJd:
                st.error("Could not extract meaningful text from one or both documents. Please check the content.")
            else:
                text_corpus = [processedResume, processedJd]
                
                vectorizer = TfidfVectorizer()
                try:
                    tfidf_matrix = vectorizer.fit_transform(text_corpus)
                    similarity_score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
                    
                    st.header("Analysis Results")
                    
                    # Show which job description was analyzed
                    st.info(f"Analyzed against: **{selected_jd}**")
                    
                    score_percentage = similarity_score * 100
                    if score_percentage > 30:
                        st.metric(label="Compatibility Score", value=f"{score_percentage:.2f}%", delta="Good Match!")
                    elif score_percentage > 15:
                        st.metric(label="Compatibility Score", value=f"{score_percentage:.2f}%", delta="Could be improved", delta_color="off")
                    else:
                        st.metric(label="Compatibility Score", value=f"{score_percentage:.2f}%", delta="Poor Match", delta_color="inverse")

                    #KEWORD ANALYSIS (HIDDEN)
                    st.subheader("Keyword Analysis")
                    found, missing = get_keyword_analysis(processedResume, processedJd)

                    expander_found = st.expander(f"✅ Keywords Found ({len(found)})")
                    expander_found.success(", ".join(sorted(found)))

                    expander_missing = st.expander(f"❌ Keywords Missing ({len(missing)})")
                    expander_missing.warning(", ".join(sorted(missing)))
                
                except ValueError as e:
                    st.error(f"An error occurred during vectorization. This can happen if one of the documents has no unique words after processing. Details: {e}")
    else:
        st.warning("Please provide your resume and select/paste a job description to proceed.")
