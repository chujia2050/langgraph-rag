import streamlit as st
from agentic_rag import graph

st.title("💬 Chatbot")

prompt = st.text_input("Enter your question here:")

if prompt:
        user_message = st.chat_message("human")
        user_message.write(prompt)

        ai_message = st.chat_message("ai")
        ai_reply = graph.invoke({"messages": [prompt]})
        ai_message.write(ai_reply["messages"][-1].content)

