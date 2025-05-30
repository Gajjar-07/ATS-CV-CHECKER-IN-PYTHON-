from docx import Document
import streamlit as st
import re
from collections import Counter
import nltk
# We will import stopwords and word_tokenize *after* ensuring resources are ready
nltk.download('punkt_tab')  # Ensure punkt is downloaded for tokenization

# --- NLTK Resource Download (Console Output) ---

def download_nltk_resources():
    """
    Downloads NLTK resources if they are not already present.
    Uses st.cache_resource to run only once per session.
    Prints status to the console.
    Returns True if all essential resources are confirmed available, False otherwise.
    """
    resources_to_check = {
        "stopwords": "corpora/stopwords.zip",
        "punkt": "tokenizers/punkt"
    }
    all_resources_ready = True

    print("\n--- NLTK Resource Check ---") # Console header

    for resource_name, resource_path_fragment in resources_to_check.items():
        try:
            nltk.data.find(resource_path_fragment)
            print(f"[NLTK INFO] Resource '{resource_name}' is available.") # Console
        except LookupError:
            print(f"[NLTK WARNING] Resource '{resource_name}' not found. Attempting download...") # Console
            try:
                nltk.download(resource_name, quiet=True)
                nltk.data.find(resource_path_fragment) # Verify again
                print(f"[NLTK SUCCESS] Resource '{resource_name}' downloaded successfully.") # Console
            except Exception as e:
                print(f"[NLTK ERROR] Failed to download or verify resource '{resource_name}'. Error: {e}") # Console
                # Also show an error in Streamlit UI as this is critical
                st.error(
                    f"Critical NLTK resource '{resource_name}' could not be made available. "
                    "The application might not function correctly. Check console for details."
                )
                all_resources_ready = False
        except Exception as e:
            print(f"[NLTK ERROR] Unexpected error checking resource '{resource_name}'. Error: {e}") # Console
            st.error(
                f"An unexpected error occurred while checking NLTK resource '{resource_name}'. "
                "The application might not function correctly. Check console for details."
            )
            all_resources_ready = False
            
    if not all_resources_ready:
        print("[NLTK STATUS] One or more essential NLTK resources are missing.") # Console
        st.warning("One or more essential NLTK resources are missing. Functionality will be limited. Check console logs.")
    else:
        print("[NLTK STATUS] All essential NLTK resources are ready.") # Console
        # Optionally, you can still show a success message in UI if you want
        # st.success("NLTK resources loaded successfully.")
        
    print("--- End NLTK Resource Check ---\n")
    return all_resources_ready

# --- Initialize NLTK resources and dependent imports ---
# This function will now print to console
NLTK_RESOURCES_READY = download_nltk_resources()

if NLTK_RESOURCES_READY:
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    STOP_WORDS = set(stopwords.words('english'))
    print("[APP INFO] NLTK stopwords and word_tokenize loaded.")
else:
    st.error("Application cannot proceed effectively without NLTK resources. Please resolve the download issues (see console).")
    STOP_WORDS = set()
    def word_tokenize(text):
        print("[APP WARNING] word_tokenize is using a fallback due to missing NLTK resources.")
        st.warning("word_tokenize is not fully available due to missing NLTK resources.")
        return text.split() if text else []
    # Consider st.stop() if the app is truly unusable

# (The rest of your helper functions and Streamlit app code remains the same)
# ... (extract_text_from_pdf, extract_text_from_docx, preprocess_text, etc.)
# ... (Streamlit UI layout: title, columns, inputs, button, results)

# Make sure your `preprocess_text` function still checks `NLTK_RESOURCES_READY`
# or handles potential errors if `word_tokenize` is the dummy version.

# Example of how preprocess_text might look:
def preprocess_text(text):
    if not text:
        return []
    if not NLTK_RESOURCES_READY:
        print("[PREPROCESS WARNING] NLTK resources not ready, preprocessing quality will be affected.")
        # Fallback to basic processing if NLTK is not available
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        tokens = text.split()
        # STOP_WORDS might be an empty set if NLTK failed, which is fine
        filtered_tokens = [word for word in tokens if word not in STOP_WORDS and (len(word) > 1 or word.isdigit())]
        return filtered_tokens

    # Normal processing if NLTK is ready
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    try:
        tokens = word_tokenize(text)
    except Exception as e:
        print(f"[PREPROCESS ERROR] Error during word tokenization: {e}. Using basic split.")
        st.error(f"Error during word tokenization (likely NLTK punkt issue): {e}. Using fallback.")
        tokens = text.split()

    filtered_tokens = [
        word for word in tokens
        if word not in STOP_WORDS and (len(word) > 1 or word.isdigit())
    ]
    return filtered_tokens

def extract_text_from_pdf(file):
    reader = PyPDF2.PdfReader(file)
    return " ".join([page.extract_text() or "" for page in reader.pages])

def extract_text_from_docx(file):
    doc = docx.Document(file)
    return "\n".join([para.text for para in doc.paragraphs])
def calculate_match_score(resume_tokens, jd_keywords, min_freq=1, top_n=75):
    """
    Calculate the match score between resume tokens and job description keywords.
    Returns a tuple of (score, matched_keywords, missing_keywords).
    """
    if not resume_tokens or not jd_keywords:
        return 0.0, set(), set()

    # Count frequency of each token in the resume
    resume_counter = Counter(resume_tokens)
    
    # Filter JD keywords based on frequency
    jd_keywords_filtered = {word for word, count in Counter(jd_keywords).items() if count >= min_freq}
    
    # Get matched keywords
    matched_keywords = jd_keywords_filtered.intersection(resume_counter.keys())
    
    # Get missing keywords
    missing_keywords = jd_keywords_filtered.difference(matched_keywords)
    
    # Calculate score
    score = (len(matched_keywords) / len(jd_keywords_filtered)) * 100 if jd_keywords_filtered else 0.0
    
    # Limit to top N keywords if specified
    if top_n > 0:
        matched_keywords = set(sorted(matched_keywords, key=lambda x: resume_counter[x], reverse=True)[:top_n])
        missing_keywords = set(sorted(missing_keywords, key=lambda x: resume_counter[x], reverse=True)[:top_n])
    
    return round(score, 2), matched_keywords, missing_keywords

def get_keywords(text, min_freq=2, top_n=50):
    """Extracts keywords based on frequency, excluding common words."""
    tokens = preprocess_text(text)
    if not tokens:
        return []
    
    counts = Counter(tokens)
    keywords = [word for word, count in counts.most_common(top_n * 2)
                if count >= min_freq and not word.isdigit()]
    return keywords[:top_n]
# --- Streamlit App UI (ensure it starts after NLTK check) ---
st.set_page_config(page_title="ATS Resume Checker",page_icon="nand.png", layout="wide")
st.title("📄 ATS Resume Checker")
st.markdown("""
    Analyze your resume against a job description to improve your chances!
    This tool provides a basic keyword-based matching.
""")

if not NLTK_RESOURCES_READY:
    st.error("NLTK resources failed to load. The application may not work correctly. Check the console output for details.")
    # Optionally st.stop() if the app is completely broken without NLTK
    # st.stop()


# The rest of your Streamlit UI code (col1, col2, button, etc.)
# ... (paste the UI part of your previous code here)
# --- Input Columns ---
col1, col2 = st.columns(2)

with col1:
    st.header("📝 Job Description")
    jd_text_input = st.text_area("Paste the Job Description here:", height=300, key="jd_text")

with col2:
    st.header("📄 Your Resume")
    resume_source = st.radio("Resume Source:", ("Upload File", "Paste Text"), key="resume_source_radio")

    resume_text_input = ""
    uploaded_resume_file = None

    if resume_source == "Upload File":
        uploaded_resume_file = st.file_uploader("Upload your Resume (PDF, DOCX, TXT):", type=['txt', 'pdf', 'docx'], key="resume_upload")
        if uploaded_resume_file:
            st.success(f"Uploaded {uploaded_resume_file.name}")
    else:
        resume_text_input = st.text_area("Paste your Resume text here:", height=300, key="resume_text")

# --- Analysis Button ---
if st.button("🔍 Analyze Resume", type="primary", use_container_width=True):
    if not NLTK_RESOURCES_READY:
        st.error("Cannot analyze: NLTK resources are not available. Check errors in the console.")
        st.stop()

    # (Your existing resume text extraction logic)
    # ...
    resume_text = ""
    if uploaded_resume_file:
        if uploaded_resume_file.type == "application/pdf":
            # Ensure extract_text_from_pdf is defined and PyPDF2 imported
            import PyPDF2 
            resume_text = extract_text_from_pdf(uploaded_resume_file)
        elif uploaded_resume_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document": # DOCX
            # Ensure extract_text_from_docx is defined and python-docx imported
            from docx import Document
            resume_text = extract_text_from_docx(uploaded_resume_file)
        elif uploaded_resume_file.type == "text/plain":
            resume_text = uploaded_resume_file.read().decode()
        else:
            st.error("Unsupported file type for resume.")
            st.stop()
    elif resume_text_input:
        resume_text = resume_text_input
    else:
        st.warning("Please provide resume content (either upload or paste).")
        st.stop()

    jd_text = jd_text_input
    if not jd_text:
        st.warning("Please paste the Job Description.")
        st.stop()

    if not resume_text: 
        st.warning("Could not extract text from resume or resume is empty.")
        st.stop()
    
    # (Your existing analysis logic: get_keywords, calculate_match_score)
    # ...
    with st.spinner("Analyzing... This might take a moment."):
        # Ensure get_keywords and calculate_match_score are defined
        jd_keywords_extracted = get_keywords(jd_text, min_freq=1, top_n=75) # get_keywords calls preprocess_text
        resume_tokens_processed = preprocess_text(resume_text) 

        if not jd_keywords_extracted:
            st.error("Could not extract any meaningful keywords from the Job Description. Please check its content.")
            st.stop()
        if not resume_tokens_processed:
            st.error("Could not extract any meaningful tokens from the Resume. Please check its content or NLTK setup (see console).")
            st.stop()

        score, matched, missing = calculate_match_score(resume_tokens_processed, jd_keywords_extracted)


    # (Your existing results display logic: st.metric, columns for matched/missing, tips)
    # ...
    st.subheader("📊 Analysis Results")
    st.metric(label="**Overall Match Score**", value=f"{score:.2f}%")
    if score < 50:
        st.warning("Your resume has a lower match. Consider incorporating more keywords from the job description.")
    elif score < 75:
        st.info("Good match! A few tweaks might improve it further.")
    else:
        st.success("Excellent match! Your resume aligns well with the job description's keywords.")

    st.markdown("---")
    col_match, col_miss = st.columns(2)

    with col_match:
        st.subheader("✅ Keywords Found in Your Resume")
        if matched:
            st.success(f"Found {len(matched)} matching keywords out of {len(jd_keywords_extracted)} from the JD.")
            for keyword in sorted(matched):
                st.markdown(f"- `{keyword}`")
        else:
            st.info("No specific keywords from the job description were found prominently in your resume.")

    with col_miss:
        st.subheader("⚠️ Keywords Missing From Your Resume")
        if missing:
            st.warning(f"Consider adding these {len(missing)} keywords if relevant to your experience:")
            for keyword in sorted(missing):
                st.markdown(f"- `{keyword}`")
        else:
            st.balloons()
            st.success("Great! No critical keywords seem to be missing based on this analysis.")
    
    st.markdown("---")
    st.subheader("💡 Tips for Improvement:")
    tips = [
        "**Tailor Your Resume:** Always customize your resume for each specific job application.",
        "**Use Keywords Naturally:** Integrate keywords from the job description into your experience, skills, and summary sections. Don't just stuff them in.",
        "**Quantify Achievements:** Use numbers and data to showcase your accomplishments (e.g., 'Increased sales by 15%').",
        "**Action Verbs:** Start bullet points with strong action verbs (e.g., 'Managed', 'Developed', 'Implemented').",
        "**Check for Typos:** Proofread carefully for any grammatical errors or typos.",
        "**ATS-Friendly Formatting:** Use a clean, simple format. Avoid tables, columns, images, headers/footers that might confuse some ATS.",
        "**Relevance is Key:** Only include keywords and skills that are genuinely relevant to your experience and the role."
    ]
    for tip in tips:
        st.markdown(f"- {tip}")

    st.markdown("---")
    st.info("**Disclaimer:** This tool provides a basic keyword analysis. ATS systems vary, and human recruiters also play a crucial role. This is a helper, not a guarantee.")

    with st.expander("🔍 View Processed Texts (for debugging)"):
        st.markdown("#### Job Description Keywords (Top from JD)")
        st.json(jd_keywords_extracted)
        st.markdown("#### Resume Tokens (First 200 after processing)")
        st.json(resume_tokens_processed[:200])

else:
    if NLTK_RESOURCES_READY:
        st.info("Enter the Job Description and your Resume details, then click 'Analyze Resume'.")

st.markdown("---")
st.markdown("Built with ❤️ by Nand Gajjar")
