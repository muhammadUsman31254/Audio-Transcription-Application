import streamlit as st
import assemblyai as aai
import tempfile
import os
import yt_dlp
import logging
from urllib.parse import urlparse, parse_qs

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# Set up logging
logging.basicConfig(level=logging.INFO)

aai.settings.api_key = os.getenv('ASSEMBLYAI_API_KEY')
GROQ_API_KEY = os.getenv('GROQ_API_KEY')

# Set page configuration
st.set_page_config(page_title="Audio Transcription & Q&A", page_icon="🎙️")

# Initialize session state
def init_session_state():
    """Initialize or reset session state variables"""
    session_keys = [
        'transcription', 
        'qa_chain', 
        'show_transcription', 
        'audio_source', 
        'current_page'
    ]
    
    for key in session_keys:
        if key not in st.session_state:
            if key == 'current_page':
                st.session_state[key] = 'Transcribe'
            elif key == 'audio_source':
                st.session_state[key] = 'Local Audio File'
            else:
                st.session_state[key] = None

def setup_sidebar():
    """Create and manage the sidebar navigation"""
    st.sidebar.title("🎙️ Audio Transcription App")
    
    # Navigation
    st.session_state.current_page = st.sidebar.radio(
        "Navigation", 
        ["Transcribe", "Q&A"],
        index=["Transcribe", "Q&A"].index(st.session_state.current_page)
    )
    
    # Audio Source Selection
    st.sidebar.header("Audio Source")
    st.session_state.audio_source = st.sidebar.radio(
        "Choose Source", 
        ["Local Audio File", "YouTube Video"]
    )

def setup_rag(text):
    """Set up the RAG system with the transcribed text"""
    # Split the text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=100,
        chunk_overlap=25
    )
    chunks = text_splitter.create_documents([text])
    
    # Create embeddings and vector store
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vectorstore = FAISS.from_documents(chunks, embeddings)
    
    # Set up retriever
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3}
    )
    
    # Initialize LLM and QA chain
    llm = ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model_name="llama3-8b-8192"
    )
    
    # Refined system prompt
    system_prompt = (
        "You are an expert assistant specialized in analyzing audio transcriptions. Your task is to:\n\n"
        "1. Carefully review the provided context (transcription text) and answer questions with precision and clarity.\n\n"
        "2. Base your answers strictly on the information contained within the transcription:\n"
        "   - If the answer is directly available in the text, provide a concise and accurate response.\n"
        "   - Use the exact language and details from the transcription when possible.\n\n"
        "3. If the question cannot be answered using the transcription:\n"
        "   - Respond with 'I cannot find information about this in the transcription.'\n"
        "   - Do not fabricate or guess information\n"
        "   - Suggest rephrasing the question or checking the transcription content\n\n"
        "4. Maintain a professional and helpful tone.\n\n"
        "5. If multiple speakers are mentioned in the transcription, include speaker context in your answer when relevant.\n\n"
        "Context: {context}"
    )
    
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(retriever, question_answer_chain)

def is_valid_youtube_url(url):
    """Validate YouTube URL"""
    try:
        parsed_url = urlparse(url)
        if parsed_url.netloc in ['www.youtube.com', 'youtube.com', 'youtu.be']:
            if parsed_url.path == '/watch' and 'v' in parse_qs(parsed_url.query):
                return True
            elif parsed_url.netloc == 'youtu.be':
                return True
    except:
        return False
    return False

def download_youtube_audio(url):
    """Download audio from YouTube video using yt-dlp"""
    try:
        # Create temporary directory
        temp_dir = tempfile.mkdtemp()
        temp_path = os.path.join(temp_dir, "audio.mp3")
        
        # Configure yt-dlp options
        ydl_opts = {
            'format': 'bestaudio/best',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': '192',
            }],
            'outtmpl': temp_path.replace('.mp3', ''),  # yt-dlp will add extension
            'quiet': True,
            'no_warnings': True
        }
        
        # Download with progress bar
        with st.spinner("Downloading YouTube audio..."):
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
                st.info(f"Downloaded: {info.get('title', 'Video')}")
        
        # The actual file will have .mp3 extension added
        actual_path = temp_path.replace('.mp3', '') + '.mp3'
        return actual_path, temp_dir
    
    except Exception as e:
        st.error(f"Error downloading YouTube video: {str(e)}")
        return None, None

def process_transcription(audio_path):
    """Process audio transcription"""
    try:
        with st.spinner("Transcribing... This may take a few minutes."):
            # Initialize transcriber
            transcriber = aai.Transcriber()
            
            # Transcribe the audio file
            transcript = transcriber.transcribe(audio_path)
            
            # Store transcription in session state
            st.session_state.transcription = transcript.text
            
            # Set up RAG system
            st.session_state.qa_chain = setup_rag(transcript.text)
            
            # Switch to Q&A page
            st.session_state.current_page = 'Q&A'
            
            # Display success message
            st.success("Transcription completed successfully!")
            
            return transcript.text
    
    except Exception as e:
        st.error(f"An error occurred during transcription: {str(e)}")
        return None

def transcribe_page():
    """Render the transcription page"""
    st.header("Upload Audio for Transcription")
    
    # Use the audio source from session state
    if st.session_state.audio_source == "Local Audio File":
        # File uploader
        uploaded_file = st.file_uploader("Upload an audio file", type=['mp3', 'wav', 'm4a', 'flac'])
        
        if uploaded_file is not None:
            # Create a temporary file to store the uploaded audio
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                temp_filename = tmp_file.name
            
            # Add a transcribe button
            if st.button("Transcribe Audio"):
                try:
                    transcription = process_transcription(temp_filename)
                
                finally:
                    # Clean up the temporary file
                    if os.path.exists(temp_filename):
                        os.unlink(temp_filename)

    else:  # YouTube Video option
        # Add YouTube URL input
        youtube_url = st.text_input("Enter YouTube URL")
        
        if youtube_url:
            if not is_valid_youtube_url(youtube_url):
                st.error("Please enter a valid YouTube URL")
            else:
                if st.button("Download and Transcribe"):
                    # Download YouTube audio
                    temp_audio_path, temp_dir = download_youtube_audio(youtube_url)
                    
                    if temp_audio_path and os.path.exists(temp_audio_path):
                        try:
                            # Process transcription
                            transcription = process_transcription(temp_audio_path)
                        finally:
                            # Clean up temporary files
                            try:
                                os.unlink(temp_audio_path)
                                os.rmdir(temp_dir)
                            except Exception as e:
                                logging.error(f"Error cleaning up temporary files: {str(e)}")

def qa_page():
    """Render the Q&A page"""
    st.header("📝 Transcription Q&A")
    
    # Check if transcription exists
    if not st.session_state.transcription:
        st.warning("No transcription available. Please transcribe an audio file first.")
        return
    
    # Transcription preview
    with st.expander("Show Transcription"):
        st.write(st.session_state.transcription)
    
    # Q&A Section
    st.subheader("Ask a Question")
    user_question = st.text_input("Enter your question about the transcription:")
    
    if user_question:
        with st.spinner("Generating answer..."):
            try:
                # Run the retrieval-augmented generation
                response = st.session_state.qa_chain.invoke({
                    "input": user_question
                })
                
                # Display the answer
                st.write("### Answer:")
                st.write(response['answer'])
            
            except Exception as e:
                st.error(f"An error occurred while processing your question: {str(e)}")

def main():
    # Initialize session state
    init_session_state()
    
    # Set up sidebar
    setup_sidebar()
    
    # Render page based on current navigation
    if st.session_state.current_page == 'Transcribe':
        transcribe_page()
    elif st.session_state.current_page == 'Q&A':
        qa_page()

if __name__ == "__main__":
    main()