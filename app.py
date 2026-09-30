import os
import tempfile
import streamlit as st
from dotenv import load_dotenv

# Page configuration MUST be the first Streamlit command
st.set_page_config(page_title="AI Document Assistant", page_icon="📄", layout="wide")

# Load environment variables
load_dotenv()

st.title("📄 Smart PDF Q&A Assistant (Powered by Gemini)")
st.write("Upload a PDF document and ask questions grounded in its content.")

# Sidebar setup
with st.sidebar:
    st.header("Configuration")
    api_key = st.text_input("Google Gemini API Key", type="password", value=os.getenv("GOOGLE_API_KEY", ""))
    uploaded_file = st.file_uploader("Upload a PDF document", type=["pdf"])

if not api_key:
    st.info("👈 Please enter your Google Gemini API key in the sidebar (or `.env` file) to begin.")
    st.stop()

def process_pdf(file, api_key):
    from langchain_community.document_loaders import PyPDFLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_chroma import Chroma
    from langchain_google_genai import GoogleGenerativeAIEmbeddings

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(file.read())
        tmp_path = tmp_file.name

    try:
        loader = PyPDFLoader(tmp_path)
        docs = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        chunks = text_splitter.split_documents(docs)

        # Updated to explicit Google AI Studio embedding identifier
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/text-embedding-004", 
            google_api_key=api_key
        )
        vector_store = Chroma.from_documents(documents=chunks, embedding=embeddings)
        return vector_store
    finally:
        os.remove(tmp_path)

if uploaded_file:
    with st.spinner("Analyzing document..."):
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
            from langchain_core.prompts import ChatPromptTemplate
            from langchain_core.output_parsers import StrOutputParser
            from langchain_core.runnables import RunnablePassthrough

            vector_store = process_pdf(uploaded_file, api_key)
            st.success("Document indexed successfully!")

            retriever = vector_store.as_retriever(search_kwargs={"k": 3})

            llm = ChatGoogleGenerativeAI(
                model="gemini-1.5-flash", 
                temperature=0.2, 
                google_api_key=api_key
            )

            prompt = ChatPromptTemplate.from_template(
                """You are an AI assistant. Answer using strictly the context below. 
If the answer isn't in the context, say 'Information not found in document.'

Context:
{context}

Question: {question}"""
            )

            def format_docs(docs):
                return "\n\n".join(doc.page_content for doc in docs)

            rag_chain = (
                {"context": retriever | format_docs, "question": RunnablePassthrough()}
                | prompt
                | llm
                | StrOutputParser()
            )

            if "messages" not in st.session_state:
                st.session_state.messages = []

            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

            if user_query := st.chat_input("Ask a question about your PDF..."):
                st.session_state.messages.append({"role": "user", "content": user_query})
                with st.chat_message("user"):
                    st.markdown(user_query)

                with st.chat_message("assistant"):
                    with st.spinner("Searching document..."):
                        response = rag_chain.invoke(user_query)
                        st.markdown(response)

                st.session_state.messages.append({"role": "assistant", "content": response})

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
else:
    st.info("👈 Upload a PDF document in the sidebar to start asking questions!")