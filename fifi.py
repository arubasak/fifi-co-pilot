import streamlit as st
from pinecone import Pinecone
from pinecone.apps.assistant.models import Message # Changed from pinecone_plugins...
from pinecone.core.client.exceptions import ApiException, PineconeException # Changed
import traceback
import datetime
from fpdf import FPDF

# --- Configuration from Streamlit Secrets ---
try:
    API_KEY = st.secrets["PINECONE_API_KEY"]
    ASSISTANT_NAME = st.secrets.get("PINECONE_ASSISTANT_NAME", "fifi-co-pilot")
    REGION = st.secrets.get("PINECONE_REGION", "us")
except KeyError as e:
    st.error(f"Missing critical secret: {e}. Please ensure this secret is configured in your Streamlit Cloud app settings or local secrets.toml.")
    st.stop()
except Exception as e:
    st.error(f"Error loading secrets: {e}. This app requires secrets to be configured.")
    st.stop()

# --- Theme Configuration ---
st.set_page_config(
    page_title="FiFi Co-Pilot",
    page_icon="🍊",
    layout="wide"
)

# --- Initialize Pinecone client and assistant object ---
@st.cache_resource
def initialize_pinecone_assistant_object():
    if not API_KEY:
        st.error("Pinecone API Key is not available. Cannot initialize.")
        return None
    try:
        pc = Pinecone(api_key=API_KEY) # Assuming region is handled by API key context or default
        try:
            pc.apps.assistant.describe_assistant(name=ASSISTANT_NAME)
            assistant_instance = pc.apps.assistant.Assistant(ASSISTANT_NAME)
            return assistant_instance
        except ApiException as e:
            if hasattr(e, 'status') and e.status == 404:
                st.error(f"Pinecone Assistant '{ASSISTANT_NAME}' not found. Please ensure it's created.")
            else:
                st.error(f"Pinecone API Error for assistant '{ASSISTANT_NAME}': {e}")
            return None
    except PineconeException as e:
        st.error(f"Pinecone client initialization error: {e}")
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred during assistant initialization: {e}")
        return None

assistant_object = initialize_pinecone_assistant_object()

# --- PDF Generation Function (Revised) ---
def generate_pdf_from_chat(chat_messages):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15) # Enable auto page break

    # Calculate effective width for cells, considering page margins
    page_width = pdf.w # Total page width (default A4 is 210mm)
    left_margin = pdf.l_margin # Default 10mm
    right_margin = pdf.r_margin # Default 10mm
    effective_cell_width = page_width - left_margin - right_margin

    # Title
    pdf.set_font("Arial", 'B', 12)
    pdf.set_fill_color(243, 112, 33) # 1-2-Taste Orange
    pdf.set_text_color(255, 255, 255) # White text
    pdf.cell(effective_cell_width, 10, "FiFi Co-Pilot Chat Transcript", 1, 1, 'C', True)
    pdf.ln(5)

    pdf.set_text_color(0, 0, 0) # Black text for messages
    line_height = 6 # Increased slightly for better readability

    for message in chat_messages:
        role = message["role"].capitalize()
        content = str(message["content"]) # Ensure content is a string

        # Role styling
        pdf.set_font("Arial", 'B', 10) # Bold for role
        try:
            pdf.multi_cell(effective_cell_width, line_height, f"{role}:", 0, 'L', False)
        except Exception as e:
            pdf.multi_cell(effective_cell_width, line_height, f"{role}: [Error rendering role: {e}]", 0, 'L', False)


        # Content styling
        pdf.set_font("Arial", '', 10) # Regular for content
        try:
            encoded_content = content.encode('latin-1', 'replace').decode('latin-1')
        except Exception:
            encoded_content = "[Content has characters not supported by PDF font encoding]"
        
        try:
            pdf.multi_cell(effective_cell_width, line_height, encoded_content, 0, 'L', False)
        except RuntimeError as e: # Catch the specific "Not enough horizontal space"
            if "Not enough horizontal space" in str(e):
                # Try to render a placeholder if content is too problematic for the cell
                error_placeholder = "[This message contained content too wide or complex for PDF rendering at this width]"
                pdf.multi_cell(effective_cell_width, line_height, error_placeholder, 0, 'L', False)
            else: # Re-raise other runtime errors
                raise 
        except Exception as e_gen: # Catch other general errors during multi_cell
             pdf.multi_cell(effective_cell_width, line_height, f"[Error rendering content: {e_gen}]", 0, 'L', False)


        pdf.ln(3) # Add a little space between messages

    return pdf.output(dest='S').encode('latin-1')


# --- Initialize session state ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Function to handle message sending and processing ---
def handle_user_query(user_query: str):
    if not user_query:
        return
    if not assistant_object:
        st.error("Assistant is not available. Please check configuration or try again later.")
        return

    st.session_state.messages.append({"role": "user", "content": user_query})

    sdk_messages = []
    conversion_error_message = None
    try:
        for msg_dict in st.session_state.messages:
            sdk_messages.append(Message(role=msg_dict.get("role"), content=str(msg_dict.get("content")))) # Ensure content is string
    except Exception as e:
        conversion_error_message = f"Error preparing messages for assistant: {e}"

    assistant_reply_content = ""
    if conversion_error_message:
        assistant_reply_content = conversion_error_message
    else:
        try:
            with st.spinner("FiFi is thinking..."):
                response_from_sdk = assistant_object.chat(messages=sdk_messages)
            if response_from_sdk and hasattr(response_from_sdk, 'content'):
                assistant_reply_content = response_from_sdk.content
            elif hasattr(response_from_sdk, 'message') and hasattr(response_from_sdk.message, 'content'):
                 assistant_reply_content = response_from_sdk.message.content
            else:
                assistant_reply_content = "(FiFi's response structure was unexpected.)"
            if not assistant_reply_content:
                 assistant_reply_content = "(FiFi returned an empty reply.)"
        except ApiException as e:
            assistant_reply_content = f"Sorry, an API error occurred with Pinecone: {e}"
        except PineconeException as e:
            assistant_reply_content = f"Sorry, a Pinecone client error occurred: {e}"
        except Exception as e:
            assistant_reply_content = f"Sorry, an unexpected error occurred while FiFi was thinking: {e}"
            
    st.session_state.messages.append({"role": "assistant", "content": assistant_reply_content})

# --- Streamlit App UI ---
st.title("🍊 1-2-Taste FiFi Co-Pilot")

if not assistant_object:
    st.warning("FiFi Co-Pilot is currently unavailable. This may be due to configuration issues or service downtime.")
else:
    st.sidebar.markdown("## Quick Questions")
    preview_questions = [
        "Help me with my recipe for a new juice drink",
        "Suggest me some strawberry flavours for beverage",
        "I need vanilla flavours for ice-cream"
    ]
    for question in preview_questions:
        if st.sidebar.button(question, key=f"preview_{question}", use_container_width=True):
            handle_user_query(question)

    user_prompt = st.chat_input("Ask FiFi Co-Pilot...", key="main_chat_input", disabled=(not assistant_object))
    if user_prompt:
        handle_user_query(user_prompt)

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

st.sidebar.markdown("---")
if st.session_state.messages:
    chat_export_data_txt = "\n\n".join([f"{msg['role'].capitalize()}: {str(msg['content'])}" for msg in st.session_state.messages])
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    st.sidebar.download_button(
        label="📥 Download Chat (TXT)",
        data=chat_export_data_txt,
        file_name=f"fifi_chat_{current_time}.txt",
        mime="text/plain",
        use_container_width=True
    )
    try:
        pdf_data = generate_pdf_from_chat(st.session_state.messages)
        st.sidebar.download_button(
            label="📄 Download Chat (PDF)",
            data=pdf_data,
            file_name=f"fifi_chat_{current_time}.pdf",
            mime="application/pdf",
            use_container_width=True
        )
    except Exception as e:
        st.sidebar.error(f"PDF generation failed: {e}")
        # For server-side logs during development:
        # print(f"PDF Error: {e}")
        # traceback.print_exc()


if st.sidebar.button("🧹 Clear Chat History", use_container_width=True):
    st.session_state.messages = []

st.sidebar.markdown("---")
st.sidebar.info("💡 Tip: Ask about specific ingredients, applications, or technical details!")
