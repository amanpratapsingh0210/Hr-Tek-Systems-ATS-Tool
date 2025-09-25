import nltk
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('omw-1.4', quiet=True)
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import PyPDF2
import string

def preprocessText(text):
    if not text:
        return ""
    tokens = word_tokenize(text.lower())
    tokens = [word for word in tokens if word.isalpha()]
    stopWords = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stopWords]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    
    return " ".join(tokens)

def extractTextFromPdf(pdfPath):
    try:
        text = ""
        with open(pdfPath, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                page_text = page.extract_text() or ""
                text += page_text
        return text.strip()
    except Exception as e:
        return f"Error reading PDF: {e}"

# PATH TO THE RESUME FILE BELOW
pdfPath = r'' 

# PASTE JOB DESCRIPTION BELOW
jobDescription = """ 

"""

resumeText = extractTextFromPdf(pdfPath)
if "Error" in resumeText:
    print(resumeText)
else:
    processedResume = preprocessText(resumeText)
    processedJD = preprocessText(jobDescription)
    
    textCorpus = [processedResume, processedJD]
    
    vectorizer = TfidfVectorizer()
    tfidfMatrix = vectorizer.fit_transform(textCorpus)
    
    similarityScore = cosine_similarity(tfidfMatrix[0:1], tfidfMatrix[1:2])[0][0]
    
    print(f"Resume and Job Description Compatibility Score : {similarityScore:.2%}")
    
    if similarityScore > 0.30:
        print("This looks like a good match!")
    elif similarityScore > 0.15:
        print("This could be a potential match. Consider tailoring your resume.")
    else:
        print("This may not be a strong match.")