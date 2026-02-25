import os
import zipfile
import tempfile
from typing import List

import streamlit as st
from dotenv import load_dotenv

# Updated imports for latest langchain
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI

# Load environment variables from .env file
load_dotenv()

# -----------------------------
# Config
# -----------------------------

SUPPORTED_EXT = (".py", ".js", ".ts", ".java", ".md", ".txt", ".json", ".cpp", ".c", ".h", ".go", ".rs", ".html", ".css", ".jsx", ".tsx")


@st.cache_resource
@st.cache_resource(show_spinner=False)
def get_embeddings():
    """Load HuggingFace embeddings with auto-download"""
    import os
    from langchain_community.embeddings import HuggingFaceEmbeddings
    
    # Disable symlinks warning on Windows
    os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
    
    with st.spinner("⏳ Loading AI model... First time may take 2-3 minutes"):
        try:
            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': False}
            )
            return embeddings
        except Exception as e:
            st.error(f"Model load failed: {str(e)}")
            st.info("Attempting manual download...")
            
            # Force download using sentence-transformers directly
            from sentence_transformers import SentenceTransformer
            SentenceTransformer('all-MiniLM-L6-v2')
            
            # Retry loading
            return HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={'device': 'cpu'}
            )


def extract_code_files(zip_file) -> List[Document]:
    """
    Extract supported code files from uploaded ZIP
    Returns list of Document objects with content and metadata
    """
    temp_dir = tempfile.mkdtemp()
    
    # Extract ZIP to temp directory
    with zipfile.ZipFile(zip_file, "r") as zip_ref:
        zip_ref.extractall(temp_dir)

    docs = []
    file_count = 0
    
    # Walk through extracted files
    for root, _, files in os.walk(temp_dir):
        for file in files:
            if file.endswith(SUPPORTED_EXT):
                path = os.path.join(root, file)
                try:
                    with open(path, "r", encoding="utf-8", errors="ignore") as f:
                        text = f.read()
                    
                    # Skip empty files
                    if text.strip():
                        # Use relative filename for cleaner display
                        display_name = os.path.relpath(path, temp_dir)
                        docs.append(
                            Document(
                                page_content=text,
                                metadata={"source": display_name, "filename": file}
                            )
                        )
                        file_count += 1
                        
                except Exception as e:
                    continue
    
    return docs


def chunk_documents(docs: List[Document]) -> List[Document]:
    """Split documents into smaller chunks for better retrieval"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        separators=["\n\n", "\n", " ", ""]
    )
    return splitter.split_documents(docs)


def build_index(zip_file):
    """
    Build vector index from uploaded ZIP file
    Returns Chroma vector database
    """
    docs = extract_code_files(zip_file)
    
    if not docs:
        raise ValueError(
            f"No supported code files found in ZIP.\n"
            f"Supported extensions: {', '.join(SUPPORTED_EXT)}"
        )
    
    chunks = chunk_documents(docs)
    
    # Create in-memory Chroma vector store
    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=get_embeddings()
    )
    
    return vectordb, len(docs)


def ask_gemini(question: str, docs: List[Document]) -> str:
    """
    Query Google's Gemini model with code context
    """
    # Check for API key
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        return "⚠️ ERROR: GOOGLE_API_KEY not found. Please check your .env file."

    # Initialize Gemini model
    try:
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            google_api_key=api_key,
            temperature=0.1,
            convert_system_message_to_human=True
        )
    except Exception as e:
        return f"❌ Error initializing Gemini: {str(e)}"

    # Build context from retrieved documents
    context_parts = []
    for i, doc in enumerate(docs, 1):
        content = doc.page_content[:2500]
        context_parts.append(
            f"=== FILE {i}: {doc.metadata['source']} ===\n{content}\n"
        )
    
    context = "\n".join(context_parts)

    # Create prompt
    prompt = f"""You are an expert software engineer analyzing a codebase.

Your task is to answer questions about the provided code clearly and accurately.

CODE FILES:
{context}

QUESTION: {question}

INSTRUCTIONS:
- Provide a clear, technical explanation
- Reference specific files when relevant
- If you don't know, say so honestly
- Keep your answer concise but informative

ANSWER:"""

    # Generate response
    try:
        response = llm.invoke(prompt)
        return response.content
    except Exception as e:
        error_msg = str(e)
        if "API key not valid" in error_msg:
            return "❌ Invalid API key. Please check your GOOGLE_API_KEY in .env file."
        elif "quota" in error_msg.lower():
            return "⚠️ API quota exceeded. Free tier allows 1,500 requests/day."
        else:
            return f"❌ Error generating response: {error_msg}"


# -----------------------------
# Streamlit UI
# -----------------------------

def main():
    # Page configuration
    st.set_page_config(
        page_title="AI Codebase Explainer",
        page_icon="💻",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Header
    st.title("💻 AI Codebase Explainer")
    st.caption("🚀 Powered by Google Gemini | 🆓 Free tier: 1,500 requests/day")
    
    # Check API key on startup
    if not os.getenv("GOOGLE_API_KEY"):
        st.error("⚠️ GOOGLE_API_KEY not found! Please create a .env file with your API key.")
        st.info("Get your free API key at: https://makersuite.google.com/app/apikey")
        st.stop()

    # Sidebar info
    with st.sidebar:
        st.header("📋 About")
        st.markdown("""
        This tool helps you understand codebases quickly:
        
        1. 📁 Upload a ZIP file with code
        2. 🔨 Build the search index
        3. ❓ Ask any question about the code
        
        **Supported files:**
        """)
        st.code(", ".join(SUPPORTED_EXT))
        
        st.markdown("---")
        st.markdown("**Model:** Gemini 1.5 Flash")
        st.markdown("**Embeddings:** all-MiniLM-L6-v2")
        
        if st.button("🔄 Reset Application"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

    st.markdown("---")
    
    # File upload section
    st.subheader("1️⃣ Upload Code")
    zip_upload = st.file_uploader(
        "Upload a ZIP file containing your project",
        type=["zip"],
        help="ZIP should contain code files (Python, JS, Java, etc.)"
    )

    if zip_upload:
        st.subheader("2️⃣ Build Index")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            build_clicked = st.button("🔨 Build Code Index", use_container_width=True)
        
        with col2:
            st.info("💡 This extracts files and creates a searchable vector index")

        if build_clicked:
            with st.spinner("📂 Extracting files and building index..."):
                try:
                    vectordb, file_count = build_index(zip_upload)
                    st.session_state.vectordb = vectordb
                    st.session_state.file_count = file_count
                    st.success(f"✅ Successfully indexed {file_count} files!")
                except ValueError as ve:
                    st.error(f"❌ {str(ve)}")
                except Exception as e:
                    st.error(f"❌ Unexpected error: {str(e)}")

    if st.session_state.get("vectordb"):
        st.markdown("---")
        st.subheader("3️⃣ Ask Questions")
        
        with st.expander("💡 Example questions you can ask"):
            st.markdown("""
            - What does this project do?
            - Explain the main entry point
            - How does authentication work?
            - What are the key classes/functions?
            - Find potential bugs or issues
            - Explain the database models
            - How is the API structured?
            """)

        question = st.text_input(
            "Your question:",
            placeholder="e.g., What is the main purpose of this codebase?",
            key="question_input"
        )

        col_ask, col_status = st.columns([1, 3])
        
        with col_ask:
            ask_clicked = st.button("🔍 Ask Gemini", use_container_width=True)
        
        with col_status:
            if st.session_state.get("file_count"):
                st.caption(f"📊 Indexed {st.session_state['file_count']} files ready for query")

        if ask_clicked and question:
            if not question.strip():
                st.warning("⚠️ Please enter a question!")
            else:
                with st.spinner("🤔 Gemini is analyzing your code..."):
                    hits = st.session_state.vectordb.similarity_search(question, k=5)
                    answer = ask_gemini(question, hits)
                    
                    st.markdown("---")
                    st.subheader("📝 Answer")
                    st.markdown(answer)
                    
                    with st.expander("📄 Source Files Referenced"):
                        for i, doc in enumerate(hits, 1):
                            st.markdown(f"**{i}. `{doc.metadata['source']}`**")
                            preview = doc.page_content[:300].replace("\n", " ")
                            if len(doc.page_content) > 300:
                                preview += "..."
                            st.caption(f"Preview: {preview}")
                            st.divider()

    elif not zip_upload:
        st.info("👆 Upload a ZIP file to get started!")
        
        st.markdown("---")
        st.subheader("🎯 How it works")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("### 1️⃣ Upload")
            st.markdown("ZIP your project folder and upload it here")
        
        with col2:
            st.markdown("### 2️⃣ Index")
            st.markdown("We extract and index all code files for semantic search")
        
        with col3:
            st.markdown("### 3️⃣ Ask")
            st.markdown("Ask natural language questions about the codebase")


if __name__ == "__main__":
    main()