import streamlit as st
import psycopg2
from psycopg2.extras import RealDictCursor
import google.generativeai as genai
import PyPDF2
import io
import json
from datetime import datetime
import re

# Page configuration
st.set_page_config(
    page_title="Health AI Agent",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize Gemini - Check if API key is valid
GEMINI_API_KEY = "AIzaSyAZJHtWCI9LBqYVz3FMBfuJqsmo7"
if GEMINI_API_KEY and len(GEMINI_API_KEY) > 20:  # Basic validation
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        GEMINI_AVAILABLE = True
    except:
        GEMINI_AVAILABLE = False
else:
    GEMINI_AVAILABLE = False

# Database connection
@st.cache_resource
def init_connection():
    try:
        conn = psycopg2.connect(
            host="localhost",
            database="Health_ai",
            user="postgres",
            password="jeet",
            cursor_factory=RealDictCursor
        )
        return conn
    except Exception as e:
        st.error(f"Database connection failed: {e}")
        return None

conn = init_connection()

# Initialize database tables
def init_db():
    if conn is not None:
        try:
            with conn.cursor() as cur:
                # Create families table
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS families (
                        id SERIAL PRIMARY KEY,
                        phone_number VARCHAR(20) UNIQUE NOT NULL,
                        head_name VARCHAR(100) NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Create family_members table
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS family_members (
                        id SERIAL PRIMARY KEY,
                        family_id INTEGER REFERENCES families(id) ON DELETE CASCADE,
                        name VARCHAR(100) NOT NULL,
                        age INTEGER NOT NULL,
                        sex VARCHAR(10) NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Create medical_reports table
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS medical_reports (
                        id SERIAL PRIMARY KEY,
                        member_id INTEGER REFERENCES family_members(id) ON DELETE CASCADE,
                        report_text TEXT,
                        report_date DATE,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                conn.commit()
        except Exception as e:
            st.error(f"Database initialization failed: {e}")

init_db()

# Session state initialization
if "current_family" not in st.session_state:
    st.session_state.current_family = None
if "current_member" not in st.session_state:
    st.session_state.current_member = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "registration_step" not in st.session_state:
    st.session_state.registration_step = 0
if "new_member_data" not in st.session_state:
    st.session_state.new_member_data = {"name": "", "age": "", "sex": ""}
if "processing" not in st.session_state:
    st.session_state.processing = False
if "file_processed" not in st.session_state:
    st.session_state.file_processed = False

# Utility functions
def extract_text_from_pdf(uploaded_file):
    try:
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(uploaded_file.read()))
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        return text
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
        return ""

def get_gemini_insight(report_text, previous_reports=None):
    if not GEMINI_AVAILABLE:
        return "Gemini AI service is currently unavailable. Please check your API key configuration."
    
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        if previous_reports:
            # Timeline analysis with previous reports
            prompt = f"""
            Analyze this medical report in the context of the patient's history. 
            Provide a concise one-liner insight focusing on changes, trends, or important observations.
            
            Previous reports context: {previous_reports}
            
            Current report: {report_text}
            
            One-liner insight:
            """
        else:
            # First-time analysis
            prompt = f"""
            Analyze this medical report and provide a concise one-liner insight.
            Focus on the most important finding or observation.
            
            Report: {report_text}
            
            One-liner insight:
            """
        
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"Error generating insight: {str(e)}"

def get_family_by_phone(phone_number):
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM families WHERE phone_number = %s", (phone_number,))
            return cur.fetchone()
    except Exception as e:
        st.error(f"Database error: {e}")
        return None

def create_family(phone_number, head_name):
    try:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO families (phone_number, head_name) VALUES (%s, %s) RETURNING *",
                (phone_number, head_name)
            )
            conn.commit()
            return cur.fetchone()
    except Exception as e:
        st.error(f"Error creating family: {e}")
        return None

def get_family_members(family_id):
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM family_members WHERE family_id = %s", (family_id,))
            return cur.fetchall()
    except Exception as e:
        st.error(f"Database error: {e}")
        return []

def create_family_member(family_id, name, age, sex):
    try:
        with conn.cursor() as cur:
            cur.execute(
                """INSERT INTO family_members (family_id, name, age, sex) 
                VALUES (%s, %s, %s, %s) RETURNING *""",
                (family_id, name, age, sex)
            )
            conn.commit()
            return cur.fetchone()
    except Exception as e:
        st.error(f"Error creating family member: {e}")
        return None

def save_medical_report(member_id, report_text, report_date=None):
    if report_date is None:
        report_date = datetime.now().date()
    
    try:
        with conn.cursor() as cur:
            cur.execute(
                """INSERT INTO medical_reports (member_id, report_text, report_date) 
                VALUES (%s, %s, %s) RETURNING *""",
                (member_id, report_text, report_date)
            )
            conn.commit()
            return cur.fetchone()
    except Exception as e:
        st.error(f"Error saving medical report: {e}")
        return None

def get_medical_reports(member_id):
    try:
        with conn.cursor() as cur:
            cur.execute(
                """SELECT * FROM medical_reports 
                WHERE member_id = %s 
                ORDER BY report_date DESC""",
                (member_id,)
            )
            return cur.fetchall()
    except Exception as e:
        st.error(f"Database error: {e}")
        return []

# UI Components
def render_sidebar():
    with st.sidebar:
        st.title("🏥 Health AI Agent")
        
        if st.session_state.current_family:
            st.success(f"Logged in as: {st.session_state.current_family['head_name']}")
            st.write(f"Phone: {st.session_state.current_family['phone_number']}")
            
            if st.button("Logout"):
                st.session_state.current_family = None
                st.session_state.current_member = None
                st.session_state.chat_history = []
                st.session_state.registration_step = 0
                st.session_state.file_processed = False
                st.rerun()
            
            # Family members management
            st.subheader("Family Members")
            members = get_family_members(st.session_state.current_family['id'])
            
            for member in members:
                if st.button(
                    f"{member['name']} ({member['age']}, {member['sex']})", 
                    key=f"member_{member['id']}",
                    use_container_width=True
                ):
                    st.session_state.current_member = member
                    st.session_state.chat_history = []
                    st.session_state.file_processed = False
                    st.rerun()
            
            if st.button("+ Add New Member"):
                st.session_state.registration_step = 2
                st.session_state.new_member_data = {"name": "", "age": "", "sex": ""}
                
        else:
            st.info("Please enter your phone number to get started")

def render_chat_interface():
    st.header("Health Analysis Chat")
    
    # Chat container
    chat_container = st.container(height=400)
    
    with chat_container:
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
    
    # User input
    if prompt := st.chat_input("Type your message here...", key="chat_input"):
        # Prevent processing if already processing
        if st.session_state.processing:
            st.warning("Please wait, processing your previous request...")
            return
            
        # Add user message to chat history
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        st.session_state.processing = True
        
        # Process the message
        process_user_message(prompt)
        
        # Reset processing flag
        st.session_state.processing = False
        st.rerun()

def process_user_message(message):
    # If no family is registered
    if not st.session_state.current_family:
        st.session_state.chat_history.append({
            "role": "assistant", 
            "content": "I couldn't find your family profile. Please register with your head family name."
        })
        st.session_state.registration_step = 1
        return
    
    # If family is registered but no member is selected
    if not st.session_state.current_member:
        # Check if the message is a member name
        members = get_family_members(st.session_state.current_family['id'])
        member_names = [m['name'].lower() for m in members]
        
        if message.lower() in member_names:
            # Select the member
            for member in members:
                if member['name'].lower() == message.lower():
                    st.session_state.current_member = member
                    st.session_state.chat_history.append({
                        "role": "assistant", 
                        "content": f"Now analyzing reports for {member['name']}. You can upload a medical report PDF."
                    })
                    break
        elif "add" in message.lower() and "member" in message.lower():
            st.session_state.registration_step = 2
            st.session_state.chat_history.append({
                "role": "assistant", 
                "content": "Let's add a new family member. Please fill out the form below."
            })
        else:
            # Ask to select or create a member
            member_list = ", ".join([m['name'] for m in members])
            st.session_state.chat_history.append({
                "role": "assistant", 
                "content": f"I found these family members: {member_list}. Please type a name to select or say 'add new member' to create a new one."
            })
        return
    
    # If we have both family and member, process medical report uploads
    if "upload" in message.lower() or "report" in message.lower():
        st.session_state.chat_history.append({
            "role": "assistant", 
            "content": "Please upload a medical report PDF using the file uploader below."
        })

def render_file_uploader():
    if st.session_state.current_member:
        st.subheader("Upload Medical Report")
        
        # Use a unique key for the file uploader to prevent re-processing
        uploaded_file = st.file_uploader("Choose a PDF file", type="pdf", key=f"report_uploader_{st.session_state.current_member['id']}")
        
        if uploaded_file is not None and not st.session_state.file_processed:
            st.session_state.file_processed = True
            with st.spinner("Analyzing report..."):
                # Extract text from PDF
                report_text = extract_text_from_pdf(uploaded_file)
                
                if report_text:
                    # Get previous reports for timeline analysis
                    previous_reports = get_medical_reports(st.session_state.current_member['id'])
                    previous_texts = [r['report_text'] for r in previous_reports] if previous_reports else None
                    
                    # Get insight from Gemini
                    insight = get_gemini_insight(report_text, previous_texts)
                    
                    # Save the report
                    save_medical_report(st.session_state.current_member['id'], report_text)
                    
                    # Add to chat history
                    st.session_state.chat_history.append({
                        "role": "assistant", 
                        "content": f"**Report Analysis:** {insight}"
                    })
                    
                    # Show timeline if previous reports exist
                    if previous_reports:
                        st.session_state.chat_history.append({
                            "role": "assistant", 
                            "content": f"**Timeline Insight:** Compared to {len(previous_reports)} previous report(s), I've noted changes in your health metrics."
                        })
                    
                    st.rerun()
                else:
                    st.error("Could not extract text from the PDF. Please try another file.")
                    st.session_state.file_processed = False

def render_registration_form():
    # Family registration
    if st.session_state.registration_step == 1:
        with st.form("phone_registration", clear_on_submit=True):
            st.subheader("Register Your Family")
            phone_number = st.text_input("Phone Number", placeholder="Enter your phone number")
            head_name = st.text_input("Head of Family Name", placeholder="Enter the head of family name")
            
            if st.form_submit_button("Register"):
                if phone_number and head_name:
                    # Check if phone number already exists
                    existing_family = get_family_by_phone(phone_number)
                    if existing_family:
                        st.error("This phone number is already registered. Please use a different number.")
                    else:
                        # Create new family
                        family = create_family(phone_number, head_name)
                        if family:
                            st.session_state.current_family = family
                            st.session_state.registration_step = 2
                            st.session_state.chat_history.append({
                                "role": "assistant", 
                                "content": f"Welcome {head_name}! Your family profile has been created. Now let's add a family member."
                            })
                            st.rerun()
                        else:
                            st.error("Failed to create family profile. Please try again.")
                else:
                    st.error("Please fill in all fields")
    
    # Family member registration
    elif st.session_state.registration_step == 2:
        with st.form("member_registration", clear_on_submit=True):
            st.subheader("Add Family Member")
            name = st.text_input("Name", value=st.session_state.new_member_data['name'])
            age = st.number_input("Age", min_value=0, max_value=120, 
                                 value=int(st.session_state.new_member_data['age']) 
                                 if st.session_state.new_member_data['age'] else 0)
            sex = st.selectbox("Sex", options=["", "Male", "Female", "Other"], 
                              index=["", "Male", "Female", "Other"].index(st.session_state.new_member_data['sex']) 
                              if st.session_state.new_member_data['sex'] else 0)
            
            if st.form_submit_button("Add Member"):
                if name and age and sex:
                    member = create_family_member(
                        st.session_state.current_family['id'], 
                        name, age, sex
                    )
                    if member:
                        st.session_state.current_member = member
                        st.session_state.registration_step = 0
                        st.session_state.new_member_data = {"name": "", "age": "", "sex": ""}
                        st.session_state.chat_history.append({
                            "role": "assistant", 
                            "content": f"Added {name} to your family. You can now upload medical reports for analysis."
                        })
                        st.rerun()
                    else:
                        st.error("Failed to add family member. Please try again.")
                else:
                    st.error("Please fill in all fields")

# Main app logic
def main():
    # Show API key warning if not available
    if not GEMINI_AVAILABLE:
        st.warning("⚠️ Gemini API key is not properly configured. Some features may not work correctly.")
    
    # Initial phone number input if not logged in
    if not st.session_state.current_family:
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.title("🏥 Health AI Agent")
            st.write("Enter your phone number to get started with medical report analysis")
            
            with st.form("phone_input", clear_on_submit=True):
                phone_number = st.text_input("Phone Number", placeholder="Enter your phone number")
                if st.form_submit_button("Continue"):
                    if phone_number:
                        # Check if family exists
                        family = get_family_by_phone(phone_number)
                        if family:
                            st.session_state.current_family = family
                            st.session_state.chat_history.append({
                                "role": "assistant", 
                                "content": f"Welcome back {family['head_name']}! Please select a family member to analyze reports."
                            })
                            st.rerun()
                        else:
                            st.session_state.registration_step = 1
                            st.session_state.chat_history.append({
                                "role": "assistant", 
                                "content": "I couldn't find your family profile. Please register with your head family name."
                            })
                            st.rerun()
                    else:
                        st.error("Please enter a phone number")
    
    # Render sidebar
    render_sidebar()
    
    # Render main content based on state
    if st.session_state.current_family:
        if st.session_state.registration_step > 0:
            render_registration_form()
        else:
            render_chat_interface()
            render_file_uploader()
    else:
        # Show registration form if in registration process
        if st.session_state.registration_step > 0:
            render_registration_form()

if __name__ == "__main__":
    main()