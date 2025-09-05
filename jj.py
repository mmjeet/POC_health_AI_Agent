import re
import spacy
import PyPDF2
import streamlit as st
import google.generativeai as genai
from typing import Union
from io import BytesIO

# Load spaCy NER
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    st.error("Please install the spaCy English model: 'python -m spacy download en_core_web_sm'")
    st.stop()

def extract_text_from_pdf(pdf_file) -> Union[str, None]:
    try:
        text = ""
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        
        # Extract text from each page
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            text += page.extract_text() + "\n\n"
    
        return text.strip()
    
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
        return None

def clean_sensitive_info(text):
    # Define comprehensive list of medical terms to preserve
    medical_prefixes = [
        'Tab\.?', 'Tablet', 'Cap\.?', 'Capsule', 'Syrup', 'Inj\.?', 'Injection', 
        'Cream', 'Gel', 'Paint', 'Drop', 'Drops', 'Ointment', 'Lotion', 
        'Suspension', 'Solution', 'Powder', 'Spray', 'Patch', 'Suppository',
        'Rx\.?', 'Adv:', 'Advice:'
    ]
    
    # Common medicine name patterns and ingredients to preserve
    medicine_patterns = [
        r'Tab\.?\s+\w+',  # Tab. [MedicineName]
        r'Cap\.?\s+\w+',  # Cap. [MedicineName]
        r'\w+(cillin|mycin|floxacin|prazole|tide|zole|pine|lam|tin|fen|ol)\b',  # Common drug suffixes
        r'\w+\s+\d+(mg|mcg|g|ml|cc)',  # Medicine with dosage
        r'\b(Augmentin|Enzoflam|Pand|Hexigel)\w*\b',  # Specific medicines from your text
        r'\b\w*pain(t)?\b',  # gel paint, etc.
    ]
    
    # Medical terms and dosages to preserve
    medical_terms = [
        r'\bmg\b', r'\bmcg\b', r'\bg\b', r'\bml\b', r'\bcc\b',
        r'\bdays?\b', r'\bweeks?\b', r'\bmonths?\b',
        r'\bmassage\b', r'\bmeals?\b', r'\bbefore\b', r'\bafter\b',
        r'\bx\s*\d+', r'\d+\s*x\s*\d+', r'--\d+\s*x\s*\d+',
        r'\bgum\b', r'\bpaint\b', r'\bgel\b'
    ]
    
    # Create a list to store all medical terms found in text
    protected_terms = set()
    
    # Find and protect all medical terms
    for pattern in medicine_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        protected_terms.update(matches)
    
    # --- Remove phone numbers ---
    text = re.sub(r"\+?\d[\d\-\s]{7,}\d", "[PHONE]", text)

    # --- Remove emails ---
    text = re.sub(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", "[EMAIL]", text)

    # --- Remove websites ---
    text = re.sub(r"(www\.[^\s]+|https?://[^\s]+)", "[WEB]", text)

    # --- Remove explicit doctor titles and names (only when clearly marked) ---
    text = re.sub(r"\b(Dr\.?|DR\.?|Doctor)\s+[A-Za-z]+(?:\s+[A-Za-z]+)*", "[DOCTOR_NAME]", text, flags=re.IGNORECASE)
    
    # --- Remove explicit patient titles and names (only when clearly marked) ---
    text = re.sub(r"\b(Mr\.?|Mrs\.?|Ms\.?|Miss|Master)\s+[A-Za-z]+(?:\s+[A-Za-z]+)*", "[PATIENT_NAME]", text, flags=re.IGNORECASE)
    
    # --- Remove Me./Mme./Mlle. titles followed by names ---
    text = re.sub(r"\b(Me\.)\s*[A-Za-z]+(?:\s+[A-Za-z]+)*", "[PATIENT_NAME]", text, flags=re.IGNORECASE)
    
    # --- Use spaCy NER but be very selective ---
    doc = nlp(text)
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            # Only replace if it's NOT a medical term and NOT preceded by medical prefixes
            ent_text = ent.text.strip()
            
            # Skip if it's a known medical term
            is_medical = False
            for med_pattern in medicine_patterns:
                if re.search(med_pattern, ent_text, re.IGNORECASE):
                    is_medical = True
                    break
            
            # Skip if it contains medical dosage or common medical words
            if (re.search(r'\d+(mg|mcg|g|ml|cc)', ent_text, re.IGNORECASE) or 
                any(term in ent_text.lower() for term in ['tab', 'gel', 'paint', 'cream'])):
                is_medical = True
            
            # Skip if it's already been processed or is clearly medical
            if (not is_medical and 
                "[" not in ent_text and 
                not any(prefix.replace(r'\.?', '').lower() in text[max(0, text.find(ent_text)-10):text.find(ent_text)].lower() 
                        for prefix in medical_prefixes)):
                
                # Additional check: only replace if it looks like a person name
                # (standalone names that aren't part of medical context)
                context_start = max(0, text.find(ent_text) - 20)
                context_end = min(len(text), text.find(ent_text) + len(ent_text) + 20)
                context = text[context_start:context_end].lower()
                
                # Don't replace if it's in medical context
                if not any(med_word in context for med_word in ['tab', 'mg', 'gel', 'paint', 'cream', 'rx', 'adv']):
                    text = text.replace(ent_text, "[PERSON_NAME]")
                    
        elif ent.label_ == "DATE":
            text = text.replace(ent.text, "[DATE]")

    # --- Remove age patterns ---
    text = re.sub(r"\b\d{1,2}\s?(yrs|years|year|y|/m|/f)\b", "[AGE]", text, flags=re.IGNORECASE)
    text = re.sub(r"Age[:\s]*\d{1,2}", "Age: [AGE]", text, flags=re.IGNORECASE)

    # --- Remove patient ID numbers ---
    text = re.sub(r"\bPatient\s+ID[:\s]*\w+", "Patient ID: [PATIENT_ID]", text, flags=re.IGNORECASE)
    text = re.sub(r"\bID[:\s]*\d+", "ID: [ID]", text, flags=re.IGNORECASE)

    return text

def init_gemini(api_key):
    """Initialize the Gemini model"""
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash')
        return model
    except Exception as e:
        st.error(f"Failed to initialize Gemini: {e}")
        return None

def chat_with_gemini(model, prompt, context):
    """Send a prompt to Gemini with the medical report context"""
    try:
        full_prompt = f"""
        You are a medical assistant. Based on the following anonymized medical report, 
        please respond to the user's question. Be helpful, professional, and focused on 
        medical information.
        
        Medical Report:
        {context}
        
        User Question: {prompt}
        
        Response:
        """
        response = model.generate_content(full_prompt)
        return response.text
    except Exception as e:
        return f"Error communicating with Gemini: {e}"

def count_tokens(text):
    """Estimate token count (approximate)"""
    # Simple approximation: tokens ≈ words * 1.33
    words = len(text.split())
    return int(words * 1.33)

def calculate_total_tokens(cleaned_text, chat_history):
    """Calculate total tokens for the entire conversation"""
    # Count tokens for the medical report
    report_tokens = count_tokens(cleaned_text) if cleaned_text else 0
    
    # Count tokens for all messages in chat history
    chat_tokens = 0
    for message in chat_history:
        chat_tokens += count_tokens(message["content"])
    
    return report_tokens + chat_tokens

def main():
    st.set_page_config(
        page_title="Medical Report Assistant",
        page_icon="🏥",
        layout="wide",
        initial_sidebar_state="expanded"  # Changed to expanded to show sidebar by default
    )
    
    # Initialize session state
    if "cleaned_text" not in st.session_state:
        st.session_state.cleaned_text = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "gemini_model" not in st.session_state:
        st.session_state.gemini_model = None
    if "api_key" not in st.session_state:
        st.session_state.api_key = ""
    if "total_tokens" not in st.session_state:
        st.session_state.total_tokens = 0
    if "estimated_cost" not in st.session_state:
        st.session_state.estimated_cost = 0.0
    
    # Sidebar for API configuration
    with st.sidebar:
        st.title("🔑 API Configuration")
        
        api_key = st.text_input(
            "Enter your Gemini API key:", 
            type="password", 
            value=st.session_state.get("api_key", ""),
            help="Get your API key from Google AI Studio"
        )
        
        if st.button("Initialize Gemini", type="primary"):
            if api_key:
                st.session_state.api_key = api_key
                with st.spinner("Initializing Gemini..."):
                    st.session_state.gemini_model = init_gemini(api_key)
                if st.session_state.gemini_model:
                    st.success("✅ Gemini initialized!")
                else:
                    st.error("Failed to initialize Gemini")
            else:
                st.error("Please enter your API key")
        
        st.divider()
        
        # Token count information
        if st.session_state.get("cleaned_text"):
            st.subheader("📊 Token Information")
            
            # Calculate total tokens dynamically
            total_tokens = calculate_total_tokens(
                st.session_state.cleaned_text, 
                st.session_state.chat_history
            )
            
            # Update session state
            st.session_state.total_tokens = total_tokens
            
            # Calculate cost (input + output)
            # Assuming 50% of tokens are input and 50% are output for simplicity
            input_tokens = total_tokens * 0.5
            output_tokens = total_tokens * 0.5
            input_cost = (input_tokens / 1000) * 0.000035  # $0.035 per 1K tokens for input
            output_cost = (output_tokens / 1000) * 0.000175  # $0.175 per 1K tokens for output
            total_cost = input_cost + output_cost
            
            # Update session state
            st.session_state.estimated_cost = total_cost
            
            # Display metrics
            st.metric("Total Tokens", f"{total_tokens:,}")
            st.metric("Estimated Cost", f"${total_cost:.6f}")
            
            # Add pricing information
            st.caption("Based on Gemini 1.5 Flash pricing: $0.035/1K input, $0.175/1K output tokens")
        
        st.divider()
        
        # Application info
        st.subheader("ℹ️ About This App")
        st.markdown("""
        This application helps you:
        - Anonymize medical reports
        - Analyze medical content
        - Chat with AI about reports
        
        **Privacy-focused**: All sensitive information is removed before processing.
        """)
    
    # Main content area
    st.title("🏥 Medical Report Anonymization & Analysis")
    st.markdown("Upload a medical report PDF to anonymize sensitive information and chat with our AI assistant about it.")
    
    st.divider()
    
    # Main layout: Left column (upload + extracted text) and Right column (chat)
    col_left, col_right = st.columns([1, 1], gap="large")
    
    # LEFT COLUMN - PDF Upload and Extracted Text
    with col_left:
        # PDF Upload Section
        st.subheader("📄 Upload Medical Report")
        uploaded_file = st.file_uploader(
            "Choose a medical report PDF", 
            type="pdf",
            help="Upload your medical report PDF to extract and anonymize the text"
        )
        
        if uploaded_file is not None:
            col_process1, col_process2 = st.columns(2)
            
            with col_process1:
                if st.button("📝 Process PDF", type="primary"):
                    with st.spinner("Extracting text from PDF..."):
                        raw_text = extract_text_from_pdf(uploaded_file)
                        
                    if raw_text:
                        with st.spinner("Anonymizing sensitive information..."):
                            cleaned_text = clean_sensitive_info(raw_text)
                            st.session_state.cleaned_text = cleaned_text
                        
                        st.success("✅ PDF processed successfully!")
            
            with col_process2:
                if st.session_state.cleaned_text:
                    st.download_button(
                        label="💾 Download Report",
                        data=st.session_state.cleaned_text,
                        file_name="anonymized_medical_report.txt",
                        mime="text/plain",
                        type="secondary"
                    )
        
        st.divider()
        
        # Extracted Text Section
        st.subheader("📋 Anonymized Medical Report")
        
        if st.session_state.cleaned_text:
            # Show text statistics
            col_stats1, col_stats2, col_stats3 = st.columns(3)
            
            with col_stats1:
                st.metric("Characters", len(st.session_state.cleaned_text))
            
            with col_stats2:
                word_count = len(st.session_state.cleaned_text.split())
                st.metric("Words", word_count)
            
            with col_stats3:
                anonymized_count = st.session_state.cleaned_text.count("[")
                st.metric("Anonymized Items", anonymized_count)
            
            # Display the extracted text
            st.text_area(
                "Extracted and Anonymized Text", 
                st.session_state.cleaned_text, 
                height=500,
                help="This text has been automatically anonymized to remove sensitive personal information while preserving medical content."
            )
        else:
            st.info("📁 No medical report uploaded yet. Please upload a PDF file to see the extracted text here.")
    
    # RIGHT COLUMN - Chat Interface
    with col_right:
        st.subheader("💬 Medical Assistant Chat")
        
        if not st.session_state.api_key or not st.session_state.gemini_model:
            st.warning("⚠️ Please configure your Gemini API key to enable chat functionality")
        elif not st.session_state.cleaned_text:
            st.info("📄 Please upload and process a medical report to start chatting")
        else:
            st.success("🤖 Ready to answer questions about your medical report!")
            
            # Clear chat button
            if st.button("🗑️ Clear Chat History", type="secondary"):
                st.session_state.chat_history = []
                st.rerun()
        
        # Chat container
        chat_container = st.container(height=600)
        
        with chat_container:
            # Display chat history
            for message in st.session_state.chat_history:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
        
        # Chat input
        if st.session_state.cleaned_text and st.session_state.gemini_model:
            if prompt := st.chat_input("Ask me anything about the medical report..."):
                # Add user message to chat history
                st.session_state.chat_history.append({"role": "user", "content": prompt})
                
                # Get AI response
                with st.spinner("🤔 Analyzing your question..."):
                    response = chat_with_gemini(
                        st.session_state.gemini_model, 
                        prompt, 
                        st.session_state.cleaned_text
                    )
                
                # Add assistant response to chat history
                st.session_state.chat_history.append({"role": "assistant", "content": response})
                
                # Update token count in sidebar by rerunning
                st.rerun()
    
    # Footer with information
    st.divider()
    with st.expander("ℹ️ About This Application"):
        col_info1, col_info2 = st.columns(2)
        
        with col_info1:
            st.markdown("""
            **🔒 Privacy Features:**
            - Removes patient names and doctor names
            - Anonymizes phone numbers, emails, and websites
            - Replaces ages, dates, and ID numbers
            - Preserves all medical terminology and dosages
            """)
        
        with col_info2:
            st.markdown("""
            **🏥 Medical Content Preserved:**
            - Medicine names and dosages
            - Medical procedures and advice
            - Symptoms and diagnoses
            - Treatment recommendations
            """)

if __name__ == "__main__":
    main()