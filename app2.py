import socketio
import subprocess
import streamlit as st
import os
import fitz  # PyMuPDF
import pandas as pd
import torch
import whisper
from pinecone import Pinecone
from dotenv import load_dotenv
import nest_asyncio
import tempfile
import uuid
import json
import time
import re

# Fix asyncio issue with nest_asyncio
nest_asyncio.apply()

# Load environment variables
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
CHAT_HISTORY_FILE = "chat_history.json"


# Initialize Pinecone client
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX_NAME)

# Session state initialization
if "chats" not in st.session_state:
    st.session_state.chats = {}  # Stores all chats: {chat_id: {"namespace": namespace, "messages": [], "created_at": timestamp}}
if "chat_titles" not in st.session_state:
    st.session_state.chat_titles = {}  # Stores user-defined chat titles separately
if "active_chat" not in st.session_state:
    st.session_state.active_chat = None  # Tracks the currently active chat
if "chat_order" not in st.session_state:
    st.session_state.chat_order = []  # List to maintain chat order with newest first



# Function to create a new chat
def create_new_chat():
    chat_id = str(uuid.uuid4())  # Generate a unique ID for the chat
    namespace = f"chat_{chat_id}"  # Unique namespace for Pinecone
    
    # Track creation time for sorting
    created_at = time.time()
    
    # Add to the beginning of chat_order list (newest first)
    st.session_state.chat_order.insert(0, chat_id)
    
    # Create chat data with timestamp
    st.session_state.chats[chat_id] = {
        "namespace": namespace, 
        "messages": [],
        "created_at": created_at
    }
    
    st.session_state.chat_titles[chat_id] = f"Chat {chat_id[:8]}"  # Default chat title
    st.session_state.active_chat = chat_id
    # save_chats()  # Save the new chat


def initialize_empty_chats():
    """Initialize empty chat state"""
    st.session_state.chats = {}
    st.session_state.chat_titles = {}
    st.session_state.active_chat = None
    st.session_state.chat_order = []


# Button to create a new chat
if st.sidebar.button("➕ New Chat"):
    create_new_chat()


if "popover_open" not in st.session_state:
    st.session_state.popover_open = None

st.sidebar.title("Chat History")


# Sidebar chat list with popover menu - Using chat_order to display newest first
for chat_id in st.session_state.chat_order:
    chat_data = st.session_state.chats.get(chat_id)
    if not chat_data:
        continue  # Skip if chat doesn't exist
        
    with st.sidebar:
        col1, col2 = st.columns([2, 8], gap="small")

        # Chat Button (Truncated text)
        is_active = chat_id == st.session_state.active_chat
        bg_color = "#4CAF50" if is_active else "#f0f0f0"  # Green for active, Gray for inactive
        text_color = "white" if is_active else "black"

        truncate_length = 30
        chat_name = st.session_state.chat_titles[chat_id]
        truncated_name = (chat_name[:truncate_length] + "…") if len(chat_name) > truncate_length else chat_name

        with col2.container():  # Wrap button in a container for styling
            if st.button(st.session_state.chat_titles[chat_id], key=chat_id):  # To select a particular chat, and highlight it
                st.session_state.active_chat = chat_id

            st.markdown(
                f'<div style="background-color: {bg_color}; padding: 1px; border-radius: 1px; text-align: right;">',
                unsafe_allow_html=True
            )
            st.markdown("</div>", unsafe_allow_html=True)

        # Popover Menu
        with col1:
            with st.popover(""):  # Popover to hold actions
                # Rename Chat
                new_title = st.text_input("✏️ Rename", chat_name, key=f"rename_{chat_id}")
                if new_title.strip() and new_title.strip() != chat_name:
                    st.session_state.chat_titles[chat_id] = new_title.strip()
               
                if st.button("🗑 Delete", key=f"delete_{chat_id}", use_container_width=False):
                    try:
                        # Simulate Pinecone delete
                        index.delete(delete_all=True, namespace=chat_data["namespace"])
                        st.sidebar.success(f"Deleted '{chat_name}' from Pinecone.")
                    except Exception:
                        st.sidebar.error(f"Error deleting namespace '{chat_data['namespace']}'")

                    # Remove from chats dictionary
                    del st.session_state.chats[chat_id]
                    del st.session_state.chat_titles[chat_id]
                    
                    # Remove from chat_order
                    if chat_id in st.session_state.chat_order:
                        st.session_state.chat_order.remove(chat_id)
                        
                    if st.session_state.active_chat == chat_id:
                        st.session_state.active_chat = None 
                    # save_chats()
                    st.rerun()
            
                st.download_button("📥 Download Chat", 
                                json.dumps(st.session_state.chats[chat_id]["messages"], indent=4), 
                                file_name=f"chat_{chat_name}.json",
                                mime="application/json")

    
def chunk_text(text, chunk_size=500, overlap_size=50):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap_size
    return chunks

def chunk_text_pdf(pdf_content, chunk_size=500, overlap_size=50):
    chunks = []
    for item in pdf_content:
        page = str(item["page_number"])
        text = item["text"]
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunks.append("'PDF Page No:"+ page+ "'"+ text[start:end])
            start += chunk_size - overlap_size

    return chunks

# Function to add text to Pinecone index
def add_to_index(text, filename, filetype, namespace):

    if filename[-4:] == ".pdf":
        chunks = chunk_text_pdf(text, chunk_size=500, overlap_size = 50)
    else:
        chunks = chunk_text(text, chunk_size=500, overlap_size=50)
    
    records = []
    file_context = f"filetype:{filetype} filename: {filename} chunk: "

    for i, chunk in enumerate(chunks):
        if chunk:
            record = {
                "id": f"{filename.encode('ascii', 'ignore').decode()}#{i}",
                "chunk_text": file_context+chunk  # Store the chunk text
            }
            # st.write(record)
            records.append(record)
    
    if records:
        batch_size = 91
        try:
            for i in range(0, len(records), batch_size):
                batch = records[i:i + batch_size]
                index.upsert_records(namespace, batch)
            st.success(f"✅ Indexed {len(records)} chunks from '{filename}'")
        except Exception as e:
            st.error(f"❌ Failed to upsert vectors: {e}, {namespace}")

# Function to generate audio text using Whisper
def generate_audio_text(file):
    model = whisper.load_model("base")
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(file.getbuffer())
        temp_file_path = temp_file.name
    result = model.transcribe(temp_file_path)
    return result["text"]

def read_pdf_with_pages(file):
    """Reads a PDF and extracts text along with page numbers."""
    doc = fitz.open(stream=file.read(), filetype="pdf")  # Open the PDF
    pdf_data = []

    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text("text")  # Extract text from the page
        
        if text.strip():  # Ignore empty pages
            pdf_data.append({
                "page_number": page_num + 1,  # Page numbers start from 1
                "text": text
            })
    
    return pdf_data


# To get text from image
import pytesseract
from PIL import Image
import io
pytesseract.pytesseract.tesseract_cmd = r"C:\Users\Shivanshu\Softwares\tesserect\tesseract.exe"



# Process uploaded files
def process_files(files, namespace):
    all_links = []  # Store all extracted links

    for file in files:
        file_type = file.type
        filename = file.name

        if file_type == "application/pdf":
            text = read_pdf_with_pages(file)
            add_to_index(text, filename,file_type, namespace)

        elif file_type == "text/csv":
            df = pd.read_csv(file)
            text = df.to_string(index=False)
            add_to_index(text, filename,file_type, namespace)

        elif file_type == "text/plain":
            text = file.getvalue().decode("utf-8")
            add_to_index(text, filename,file_type, namespace)

        elif file_type in ["audio/mpeg", "audio/wav", "audio/opus", "audio/ogg"]:
            text = generate_audio_text(file)
            add_to_index(text, filename,file_type, namespace)

        elif file_type in ["image/png", "image/jpeg", "image/jpg"]:
            image = Image.open(io.BytesIO(file.getvalue()))
            text = pytesseract.image_to_string(image)
            add_to_index(text, filename, file_type, namespace)

        

# Function to retrieve context from Pinecone
def retrieve_context(query, namespace):
    try:
        results = index.search(
            namespace=namespace,
            query={
                "top_k": 15,
                "inputs": {
                    'text': query
                }
            }
        )
        retrieved_chunks = [hit['fields']['chunk_text'] for hit in results['result']['hits']]
        return "\n\n".join(retrieved_chunks)
    except Exception as e:
        st.error(f"❌ Retrieval error: {e}")
        return ""


import threading
speaking_complete_flag = threading.Event()

sio = socketio.Client()
try:
    sio.connect("http://127.0.0.1:5000/")
except Exception as e:
    st.error("Failed to connect to SocketIO server.")

@sio.on('speaking_complete')
def on_speaking_complete():
    print("check")
    speaking_complete_flag.set()

# Path to Piper
piper_exe = r"C:\Users\Shivanshu\Desktop\hack\piper\piper\piper.exe"
model_path = r"C:\Users\Shivanshu\Desktop\hack\piper\piper\en_US-hfc_female-medium.onnx"
output_file = r"C:\Users\Shivanshu\Desktop\hack\combining\11-working_girlfriend_emotional\avi_sl\audio\audio.wav"


def text_to_audio(sentence):
    # Run Piper as a subprocess
    process = subprocess.run(
        [piper_exe, "--model", model_path, "--output_file", output_file],
        input=sentence.encode(),  # Send text as input
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )

    # Check for errors
    if process.returncode == 0:
        print(f"Speech saved to {output_file}")
    else:
        print(f"Error: {process.stderr.decode()}")
                

if st.session_state.active_chat:
    chat_id = st.session_state.active_chat
    with st._bottom:
        left_col, right_col, _ = st.columns([6,28,1], gap="small")
        with left_col:
            with st.popover("📎Attach"):
                uploaded_files = st.file_uploader("file_upload", accept_multiple_files=True,
                    key=f"uploader_{st.session_state.active_chat}",
                    label_visibility="hidden"
                )
                if uploaded_files:
                    namespace = st.session_state.chats[st.session_state.active_chat]["namespace"]
                    process_files(uploaded_files, namespace)
        
        with right_col:
            prompt = st.chat_input("Type your message...")
            if prompt:
                # Retrieve context from Pinecone
                namespace = st.session_state.chats[st.session_state.active_chat]["namespace"]
                context = retrieve_context(prompt, namespace)
                context += "\nGive the response of above context with emotions in this list: [Happy, Sad, Angry, Excited, Surprised, Neutral]"
                context += "\nExample for format only, don't take context from here: [Neutral] The café buzzed with quiet chatter and the hum of an old espresso machine as Daniel stirred his coffee, the spoon clinking softly against the ceramic. [Sad] He traced the rim of the cup absentmindedly, the bitter scent mingling with the sweetness of pastries he had no appetite for. [Sad] Across the room, couples leaned close, laughter spilling between them like shared secrets, a stark contrast to the emptiness gnawing at his chest. [Surprised] Just as he sank deeper into his thoughts, the bell above the door jingled, and a familiar figure stepped inside, her smile lighting up the dim space. [Happy] His heart stumbled, warmth rushing through him as their eyes met, the weight on his shoulders lifting just enough to breathe. [Happy] Without thinking, he raised his hand in a small wave, and when she smiled back, the air seemed a little lighter."
                context += "\n\nDo not say or give example of how you would respond. Give a natural answer."

                # Generate response using Gemini
                from google import genai
                client = genai.Client(api_key=GEMINI_API_KEY)
                full_prompt = f"Context:\n{context}\n\nQuestion: {prompt}\nAnswer:"
                response = client.models.generate_content(
                    model="gemini-2.0-flash",
                    contents=full_prompt
                )

                output = response.text

                st.session_state.chats[st.session_state.active_chat]["messages"].append({"role": "user", "content": prompt})
                st.session_state.chats[st.session_state.active_chat]["messages"].append({"role": "assistant", "content": output})
                chat_html = """

                <div class='chat-history' id='chat-history'>
                """
                for msg in reversed(st.session_state.chats[st.session_state.active_chat]["messages"]):
                    role_class = "assistant" if msg["role"] == "assistant" else "user"
                    chat_html += f"<div class='chat-bubble {role_class}'>{msg['content']}</div>"

                chat_html += """
                </div>

                """
                # <script type="text/javascript">
                #     var chatDiv = document.getElementById('chat-history');
                #     chatDiv.scrollTop = chatDiv.scrollHeight;
                # </script>

                st.markdown(chat_html, unsafe_allow_html=True)

                # sentences = re.split('. |? |! |: |\n', output)
                output = output.replace("?", ".")
                output = output.replace("!", ".")
                output = output.replace(":", ".")
                output = output.replace("\n", ".")
                sentences = output.split(". ")
                for sentence in sentences:
                    emotion = "Neutral"
                    if "[Happy]" in sentence:
                        emotion = "Happy"
                        sentence = sentence.replace("[Happy]", "")
                    elif "[Neutral]" in sentence:
                        emotion = "Neutral"
                        sentence = sentence.replace("[Neutral]", "")
                    elif "[Sad]" in sentence:
                        emotion = "Sad"
                        sentence = sentence.replace("[Sad]", "")
                    elif "[Excited]" in sentence:
                        emotion = "Excited"
                        sentence = sentence.replace("[Excited]", "")
                    elif "[Angry]" in sentence:
                        emotion = "Angry"
                        sentence = sentence.replace("[Angry]", "")
                    elif "[Surprised]" in sentence:
                        emotion = "Surprised"
                        sentence = sentence.replace("[Surprised]", "")

                    text_to_audio(sentence)
                    speaking_complete_flag.clear()
                    sio.emit('request_audio', {'emotion': emotion, 'audioPath': '/audio/audio.wav'})
                    speaking_complete_flag.wait()



# ---------------------- Safe --------------------------------------
# ========== 1. LOAD CUSTOM CSS ==========
def load_custom_css():
    with open("style.css", "r") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_custom_css()

# ========== 2. SOCKET.IO CLIENT (OPTIONAL) ==========
# sio = socketio.Client()
# try:
#     sio.connect("http://127.0.0.1:5000/")
# except Exception as e:
#     st.error("Failed to connect to SocketIO server.")

# ========== 3. SESSION STATE FOR CHAT HISTORY ==========
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# ========== 4. VRM BACKGROUND IFRAME ==========
st.markdown("""
<div class="vrm-container">
    <iframe src="http://127.0.0.1:5000/" scrolling="no"></iframe>
</div>
""", unsafe_allow_html=True)

# # ========== 5. SIMPLE CHATBOT LOGIC ==========
# def chatbot_response(user_input):
#     """Hardcoded responses; replace with real AI if needed."""
#     if "hello" in user_input.lower():
#         return "Hi there! How can I assist you?"
#     elif "how are you" in user_input.lower():
#         return "I'm doing well, thank you!"
#     else:
#         return "I'm a simple chatbot. I can't answer that. Try asking something else."

# # ========== 6. CAPTURE USER INPUT ==========
# user_input = st.chat_input("Type your message...")

# if user_input:
#     # Save user's message
#     st.session_state["messages"].append({"role": "user", "content": user_input})
#     # Generate bot reply
#     bot_reply = chatbot_response(user_input)
#     st.session_state["messages"].append({"role": "bot", "content": bot_reply})

# # ========== 7. RENDER CHAT MESSAGES ON THE LEFT PANEL ==========
# chat_html = "<div class='chat-history'>"
# for msg in st.session_state["messages"]:
#     role_class = "bot" if msg["role"] == "bot" else "user"
#     chat_html += f"<div class='chat-bubble {role_class}'>{msg['content']}</div>"
# chat_html += "</div>"

# st.markdown(chat_html, unsafe_allow_html=True)
# # st.rerun()