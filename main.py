import os
from dotenv import load_dotenv
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain

# Load environment variables
load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

# Initialize Gemini model
llm = ChatGoogleGenerativeAI(model="models/gemini-2.0-flash")

# Set up memory (limit to last 10 messages: 5 human + 5 AI)
class LimitedConversationBufferMemory(ConversationBufferMemory):
    def save_context(self, inputs, outputs):
        super().save_context(inputs, outputs)
        # Keep only last 10 messages
        if len(self.chat_memory.messages) > 10:
            self.chat_memory.messages = self.chat_memory.messages[-10:]

memory = LimitedConversationBufferMemory(return_messages=True)

# Create the conversation chain
chat = ConversationChain(llm=llm, memory=memory, verbose=False)

# Streamlit UI setup
st.set_page_config(page_title="MY Gemini ChatBot with WAJID", page_icon="🤖")
st.title("🤖 Gemini ChatBot With Wajid")
st.markdown("Ask me anything! Technical, fun, career, or random questions.")

# Chat UI with Streamlit chat-style elements
for msg in memory.chat_memory.messages:
    if msg.type == "human":
        with st.chat_message("user"):
            st.markdown(msg.content)
    else:
        with st.chat_message("assistant"):
            st.markdown(msg.content)

# User input at the bottom
prompt = st.chat_input("Type your message here...")

if prompt:
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get Gemini's response
    response = chat.predict(input=prompt)

    # Display AI response
    with st.chat_message("assistant"):
        st.markdown(response)

# Optional: Show expandable history view (limited to last 10 messages)
with st.expander("🕒 Full Chat Memory (Last 10 Messages)"):
    for msg in memory.chat_memory.messages:
        role = "🧑‍💻 You" if msg.type == "human" else "🤖 Gemini"
        st.markdown(f"**{role}:** {msg.content}")
