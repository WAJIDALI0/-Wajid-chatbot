import os
from dotenv import load_dotenv
import streamlit as st
import streamlit.components.v1 as components
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain

# Load environment variables
load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

# Initialize Gemini model
llm = ChatGoogleGenerativeAI(model="models/gemini-2.0-flash")

# Memory for chat (last 10 messages)
memory = ConversationBufferMemory(return_messages=True)
chat = ConversationChain(llm=llm, memory=memory, verbose=False)

# Page config
st.set_page_config(page_title="WAJID - Your Voice Assistant 🤖", page_icon="🤖")
st.markdown("""
    <style>
    .main {
        background-color: #f0f2f6;
        padding: 2rem;
        font-family: 'Segoe UI', sans-serif;
    }
    .title {
        color: #4a4a4a;
        text-align: center;
        font-size: 2.5rem;
        margin-bottom: 1rem;
    }
    .subtitle {
        text-align: center;
        font-size: 1.2rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .stTextInput > div > input {
        font-size: 18px;
        padding: 10px;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("<div class='main'>", unsafe_allow_html=True)
st.markdown("<div class='title'>🤖 Meet WAJID - Your Smart Voice Chatbot</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Ask anything by typing or speaking – WAJID replies in voice and text.</div>", unsafe_allow_html=True)

# --- SPEECH INPUT BUTTON WITH AUTO-SUBMIT ---
components.html("""
<script>
    let recognition;
    function startRecognition() {
        recognition = new (window.SpeechRecognition || window.webkitSpeechRecognition)();
        recognition.lang = 'en-US';
        recognition.interimResults = false;
        recognition.maxAlternatives = 1;

        recognition.onresult = function(event) {
            const transcript = event.results[0][0].transcript;
            const inputs = window.parent.document.querySelectorAll('input[type="text"]');
            const buttons = window.parent.document.querySelectorAll('button');

            for (let input of inputs) {
                input.value = transcript;
                input.dispatchEvent(new Event('input', { bubbles: true }));
            }

            // Find and click the submit button
            for (let btn of buttons) {
                if (btn.innerText.includes("📨 Submit")) {
                    btn.click();
                    break;
                }
            }
        };

        recognition.onerror = function(event) {
            alert("Speech Recognition Error: " + event.error);
        };

        recognition.start();
    }
</script>
<button onclick="startRecognition()" style="background-color:#4CAF50;color:white;padding:10px 20px;font-size:16px;border:none;border-radius:5px;cursor:pointer;margin-bottom:10px;">🎤 Human Ask (Speak)</button>
""", height=100)

# --- TEXT INPUT ---
user_input = st.text_input("Enter your message to WAJID:")
submit = st.button("📨 Submit")

# --- WAJID'S RESPONSE ---
if submit and user_input:
    response = chat.predict(input=user_input)
    st.success("WAJID says:")
    st.text_area("", value=response, height=180, max_chars=None)

    # Voice output
    st.markdown("Click below to hear WAJID's voice:")
    st.button("🔊 Speak WAJID's Reply", on_click=lambda: components.html(f"""
        <script>
            var msg = new SpeechSynthesisUtterance({response!r});
            window.speechSynthesis.speak(msg);
        </script>
    """, height=0))

# --- CHAT HISTORY ---
with st.expander("🕒 Chat History"):
    history = memory.chat_memory.messages[-10:]
    for msg in history:
        role = "🧑‍💻 You" if msg.type == "human" else "🤖 WAJID"
        st.markdown(f"**{role}:** {msg.content}")

st.markdown("</div>", unsafe_allow_html=True)
