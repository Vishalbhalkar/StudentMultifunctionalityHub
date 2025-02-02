from typing import Callable, TypeVar
import os
from groq import Groq
import inspect
import streamlit as st
import streamlit_analytics2 as streamlit_analytics
from dotenv import load_dotenv
from streamlit_chat import message
from streamlit_pills import pills
from streamlit.runtime.scriptrunner import add_script_run_ctx, get_script_run_ctx
from streamlit.delta_generator import DeltaGenerator
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from custom_callback_handler import CustomStreamlitCallbackHandler
from agents import define_graph
import shutil
import google.generativeai as genai
import pdfplumber
import requests
import zipfile
import io

    
load_dotenv()

# Set environment variables from Streamlit secrets or .env
os.environ["LINKEDIN_EMAIL"] = st.secrets.get("LINKEDIN_EMAIL", "")
os.environ["LINKEDIN_PASS"] = st.secrets.get("LINKEDIN_PASS", "")
os.environ["LANGCHAIN_API_KEY"] = st.secrets.get("LANGCHAIN_API_KEY", "")
os.environ["LANGCHAIN_TRACING_V2"] = os.getenv("LANGCHAIN_TRACING_V2") or st.secrets.get("LANGCHAIN_TRACING_V2", "")
os.environ["LANGCHAIN_PROJECT"] = st.secrets.get("LANGCHAIN_PROJECT", "")
os.environ["GROQ_API_KEY"] = st.secrets.get("GROQ_API_KEY", "")
os.environ["SERPER_API_KEY"] = st.secrets.get("SERPER_API_KEY", "")
os.environ["FIRECRAWL_API_KEY"] = st.secrets.get("FIRECRAWL_API_KEY", "")
os.environ["LINKEDIN_SEARCH"] = st.secrets.get("LINKEDIN_JOB_SEARCH", "")

# Page configuration
st.set_page_config(layout="wide")
st.title("Student Multi-Functionality Hub ")

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home","DocComparator" ,"Chatbot","Coder Friend","E-mail bot","ReadMe for GitHub" ,"Others"])

if page == "Home":
    st.markdown(
        """
        <div style="padding:10px 0;">
            <p style="text-align:center;">Welcome to the Student Multi-Functionality Hub! This app provides a range of tools and agents to assist students with various tasks, including chatbots, code generation, resume analysis, and more.</p>
            <p style="text-align:center;">Use the sidebar to navigate to different tools and services.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

if page == "Others":
    streamlit_analytics.start_tracking()

    # Setup directories and paths
    temp_dir = "temp"
    dummy_resume_path = os.path.abspath("dummy_resume.pdf")

    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)

    # Add dummy resume if it does not exist
    if not os.path.exists(dummy_resume_path):
        default_resume_path = "path/to/your/dummy_resume.pdf"
        shutil.copy(default_resume_path, dummy_resume_path)

    # Sidebar - File Upload
    uploaded_document = st.sidebar.file_uploader("Upload Your Resume", type="pdf")

    if not uploaded_document:
        uploaded_document = open(dummy_resume_path, "rb")
        st.sidebar.write("Using a dummy resume for demonstration purposes. ")
        st.sidebar.markdown(f"[View Dummy Resume]({'\dummy_resume.pdf'})", unsafe_allow_html=True)

    bytes_data = uploaded_document.read()

    filepath = os.path.join(temp_dir, "resume.pdf")
    with open(filepath, "wb") as f:
        f.write(bytes_data)

    st.markdown("**Resume uploaded successfully!**")

    # Sidebar - Service Provider Selection
    service_provider = st.sidebar.selectbox(
        "Service Provider",
        ("groq (llama-3.1-70b-versatile)", "openai"),
    )
    streamlit_analytics.stop_tracking()

    # Not to track the key
    if service_provider == "openai":
        # Sidebar - OpenAI Configuration
        api_key_openai = st.sidebar.text_input(
            "OpenAI API Key",
            st.session_state.get("OPENAI_API_KEY", ""),
            type="password",
        )
        model_openai = st.sidebar.selectbox(
            "OpenAI Model",
            ("gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"),
        )
        settings = {
            "model": model_openai,
            "model_provider": "openai",
            "temperature": 0.3,
        }
        st.session_state["OPENAI_API_KEY"] = api_key_openai
        os.environ["OPENAI_API_KEY"] = st.session_state["OPENAI_API_KEY"]

    else:
        # Toggle visibility for Groq API Key input
        if "groq_key_visible" not in st.session_state:
            st.session_state["groq_key_visible"] = False

        if st.sidebar.button("Enter Groq API Key (optional)"):
            st.session_state["groq_key_visible"] = True

        if st.session_state["groq_key_visible"]:
            api_key_groq = st.sidebar.text_input("Groq API Key", type="password")
            st.session_state["GROQ_API_KEY"] = api_key_groq
            os.environ["GROQ_API_KEY"] = api_key_groq

        settings = {
            "model": "llama-3.1-70b-versatile",
            "model_provider": "groq",
            "temperature": 0.3,
        }

    # Sidebar - Service Provider Note
    st.sidebar.markdown(
        """
        **Note:** \n
        This multi-agent system works best with OpenAI. llama 3.1 may not always produce optimal results.\n
        Any key provided will not be stored or shared it will be used only for the current session.
        """
    )
    st.sidebar.markdown(
        """
        <div style="padding:10px 0;">
            If you like the project, give a 
            <a href="https://github.com/SagarNikam09/Student-MultiFunctionality-Hub" target="_blank" style="text-decoration:none;">
                ⭐ on GitHub
            </a>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Create the agent flow
    flow_graph = define_graph()
    message_history = StreamlitChatMessageHistory()

    # Initialize session state variables
    if "active_option_index" not in st.session_state:
        st.session_state["active_option_index"] = None
    if "interaction_history" not in st.session_state:
        st.session_state["interaction_history"] = []
    if "response_history" not in st.session_state:
        st.session_state["response_history"] = ["Hello! How can I assist you today?"]
    if "user_query_history" not in st.session_state:
        st.session_state["user_query_history"] = ["Hi there! 👋"]

    # Containers for the chat interface
    conversation_container = st.container()
    input_section = st.container()

    # Define functions used above
    def initialize_callback_handler(main_container: DeltaGenerator):
        V = TypeVar("V")

        def wrap_function(func: Callable[..., V]) -> Callable[..., V]:
            context = get_script_run_ctx()

            def wrapped(*args, **kwargs) -> V:
                add_script_run_ctx(ctx=context)
                return func(*args, **kwargs)

            return wrapped

        streamlit_callback_instance = CustomStreamlitCallbackHandler(
            parent_container=main_container
        )

        for method_name, method in inspect.getmembers(
            streamlit_callback_instance, predicate=inspect.ismethod
        ):
            setattr(streamlit_callback_instance, method_name, wrap_function(method))

        return streamlit_callback_instance

    def execute_chat_conversation(user_input, graph):
        callback_handler_instance = initialize_callback_handler(st.container())
        callback_handler = callback_handler_instance
        try:
            output = graph.invoke(
                {
                    "messages": list(message_history.messages) + [user_input],
                    "user_input": user_input,
                    "config": settings,
                    "callback": callback_handler,
                },
                {"recursion_limit": 30},
            )
            message_output = output.get("messages")[-1]
            messages_list = output.get("messages")
            message_history.clear()
            message_history.add_messages(messages_list)

        except Exception as exc:
            return ":( Sorry, Some error occurred. Can you please try again?"
        return message_output.content

    # Clear Chat functionality
    if st.button("Clear Chat"):
        st.session_state["user_query_history"] = []
        st.session_state["response_history"] = []
        message_history.clear()
        st.rerun()  # Refresh the app to reflect the cleared chat

    # for tracking the query.
    streamlit_analytics.start_tracking()

    # Display chat interface
    with input_section:
        options = [
            "Identify top trends in the tech industry relevant to gen ai",
            "Find emerging technologies and their potential impact on job opportunities",
            "Summarize my resume",
            "Create a career path visualization based on my skills and interests from my resume",
            "GenAI Jobs at Microsoft",
            "Job Search GenAI jobs in India.",
            "Analyze my resume and suggest a suitable job role and search for relevant job listings",
            "Generate a cover letter for my resume.",
        ]
        icons = ["🔍", "🌐", "📝", "📈", "💼", "🌟", "✉️", "🧠  "]

        selected_query = pills(
            "Pick a question for query:",
            options,
            clearable=None,  # type: ignore
            icons=icons,
            index=st.session_state["active_option_index"],
            key="pills",
        )
        if selected_query:
            st.session_state["active_option_index"] = options.index(selected_query)

        # Display text input form
        with st.form(key="query_form", clear_on_submit=True):
            user_input_query = st.text_input(
                "Query:",
                value="ASK HERE",
                placeholder="📝 Write your query or select from the above",
                key="input",
            )
            submit_query_button = st.form_submit_button(label="Send")

        if submit_query_button:
            if not uploaded_document:
                st.error("Please upload your resume before submitting a query.")

            elif service_provider == "openai" and not st.session_state["OPENAI_API_KEY"]:
                st.error("Please enter your OpenAI API key before submitting a query.")

            elif user_input_query:
                # Process the query as usual if resume is uploaded
                chat_output = execute_chat_conversation(user_input_query, flow_graph)
                st.session_state["user_query_history"].append(user_input_query)
                st.session_state["response_history"].append(chat_output)
                st.session_state["last_input"] = user_input_query  # Save the latest input
                st.session_state["active_option_index"] = None

    # Display chat history
    if st.session_state["response_history"]:
        with conversation_container:
            for i in range(len(st.session_state["response_history"])):
                message(
                    st.session_state["user_query_history"][i],
                    is_user=True,
                    key=str(i) + "_user",
                    avatar_style="fun-emoji",
                )
                message(
                    st.session_state["response_history"][i],
                    key=str(i),
                    avatar_style="bottts",
                )

    streamlit_analytics.stop_tracking()

elif page == "Chatbot":


    # Configure page title
    st.title("Study-Buddy Chatbot")

    # Initialize session state for chat history
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    # Initialize Gemini with initial history
    def initialize_gemini():
        try:
            # Get API key from secrets.toml
            api_key = st.secrets["GEMINI_API_KEY"]

            # Validate API key
            if not api_key:
                st.error("API key is missing. Please check your secrets.toml file.")
                return None

            genai.configure(api_key=api_key)
            model = genai.GenerativeModel("gemini-1.5-flash")
            chat = model.start_chat(
                history=[
                    {"role": "user", "parts": "Hello"},
                    {"role": "model", "parts": "Great to meet you. What would you like to know?"},
                ]
            )
            return chat
        except Exception as e:
            st.error(f"Error initializing chat: {str(e)}")
            return None

    # Initialize chat instance
    if 'chat' not in st.session_state:
        st.session_state.chat = initialize_gemini()

    # Check if chat is properly initialized
    if st.session_state.chat is None:
        st.warning("Chat initialization failed. Please check your API key and try again.")
        st.stop()  # Stop execution if chat isn't initialized

    # Display chat history
    for message in st.session_state.chat_history:
        st.write(f"{message['role']}: {message['content']}")

    # Function to process user input
    def process_user_input():
        if st.session_state.user_input and st.session_state.chat:
            user_message = st.session_state.user_input

            # Add user message to history
            st.session_state.chat_history.append({"role": "User", "content": user_message})

            try:
                # Get model response with streaming
                response = st.session_state.chat.send_message(user_message, stream=True)

                # Container for streaming response
                response_placeholder = st.empty()
                full_response = ""

                # Stream the response
                for chunk in response:
                    full_response += chunk.text
                    response_placeholder.write()

                # Add final response to history
                st.session_state.chat_history.append({"role": "Assistant", "content": full_response})

            except Exception as e:
                st.error(f"Error during chat: {str(e)}")

    # Chat input with callback
    user_input = st.text_input("Type your message:", key="user_input", on_change=process_user_input)

    # Add a clear button
    if st.button("Clear Chat"):
        st.session_state.chat_history = []
        st.session_state.chat = initialize_gemini()

    # Display initialization status
    if st.session_state.chat:
        st.success("Chat is initialized and ready!")
    else:
        st.error("Chat is not properly initialized. Please check your configuration.")

elif page == "Coder Friend":

    # Configure the Gemini API
    GENAI_API_KEY = os.getenv("GEMINI_API_KEY")  # Ensure the environment variable is set
    if GENAI_API_KEY is None:
        st.error("Please set the GEMINI_API_KEY environment variable.")
        st.stop()

    genai.configure(api_key=GENAI_API_KEY)

    # Initialize the Gemini model
    model = genai.GenerativeModel(
        model_name='gemini-1.5-pro',
        tools='code_execution'
    )

    # Streamlit App UI
    st.title("Coder Buddy")
    st.write("This app allows you to input your query, and it will generate and execute code to provide results.")

    # User input section
    st.subheader("Enter Your Query")
    query = st.text_area(
        "Type your question or task that requires code execution:",
        placeholder="E.g., What is the sum of the first 50 prime numbers?",
    )

    # Generate and execute button
    if st.button("Generate and Execute"):
        if query.strip():  # Check if the query is not empty
            try:
                # Call the Gemini API
                with st.spinner("Generating and executing code..."):
                    response = model.generate_content(query)

                # Extract and display the code and output
                generated_text = response.text or "No result generated."  # Fallback if no text is generated

                # Assume the generated code is marked or indented in the text
                st.subheader("Generated Content:")
                st.text_area("Full Response from Coder Friend:", value=generated_text, height=300)

                # Optional: Use a delimiter or parsing method to separate the code from output
                st.subheader("Execution Output (if any):")
                st.write(generated_text)  # For simplicity, display the entire response

            except Exception as e:
                st.error(f"An error occurred while processing your query: {e}")
        else:
            st.warning("Please enter a valid query.")

    # Footer
    st.write("---")
    st.write("This code created by Student Multi-Functionality Hub")

elif page == 'E-mail bot':
    
    st.write("Click the link below to visit E-mail bot:")
    st.markdown("[Open Cold Mail gen](https://coldemailai.streamlit.app/)", unsafe_allow_html=True)

elif page == 'DocComparator':
    # Configure the Gemini API
    GENAI_API_KEY = os.getenv("GEMINI_API_KEY")  # Ensure the environment variable is set
    if GENAI_API_KEY is None:
        st.error("Please set the GEMINI_API_KEY environment variable.")
        st.stop()

    genai.configure(api_key=GENAI_API_KEY)

    # Initialize the Gemini model
    model = genai.GenerativeModel(model_name="gemini-1.5-flash")

    # Streamlit App UI
    st.title("Doc Comparator")
    st.write("Upload multiple PDF documents to summarize the differences between their thesis statements.")

    # File uploader for PDFs
    uploaded_files = st.file_uploader(
        "Upload your documents (PDF files only):",
        accept_multiple_files=True,
        type=["pdf"],
    )

    prompt = "Summarize the differences between the thesis statements for these documents."

    if st.button("Generate Summary"):
        if len(uploaded_files) < 2:
            st.warning("Please upload at least two PDF files.")
        else:
            try:
                # Extract text from uploaded PDF files using pdfplumber
                file_contents = []
                for uploaded_file in uploaded_files:
                    with pdfplumber.open(uploaded_file) as pdf:
                        content = ""
                        for page in pdf.pages:
                            content += page.extract_text()
                        file_contents.append(content)

                # Prepare the input for the Gemini model
                input_data = [prompt] + file_contents

                # Generate the response
                with st.spinner("Generating summary..."):
                    response = model.generate_content(input_data)

                # Display the summary
                st.subheader("Summary of Differences:")
                st.write(response.text or "No summary generated.")
            except Exception as e:
                st.error(f"An error occurred while processing the files: {e}")

elif page == 'ReadMe for GitHub':


    # Configure the Gemini API
    GENAI_API_KEY = os.getenv("GEMINI_API_KEY")  # Ensure the environment variable is set
    if GENAI_API_KEY is None:
        st.error("Please set the GEMINI_API_KEY environment variable.")
        st.stop()

    genai.configure(api_key=GENAI_API_KEY)
    model = genai.GenerativeModel(model_name="gemini-1.5-flash")

    # Function to fetch GitHub repository content as ZIP
    def fetch_github_repo(repo_url):
        try:
            # Extract the GitHub owner and repo name
            parts = repo_url.rstrip("/").split("/")
            if len(parts) < 5:
                st.error("Invalid GitHub repository URL.")
                return None

            owner = parts[-2]
            repo = parts[-1]
            api_url = f"https://api.github.com/repos/{owner}/{repo}/zipball"

            # Request the ZIP file
            response = requests.get(api_url, stream=True)
            if response.status_code == 200:
                return response.content
            else:
                st.error(f"Failed to fetch repository: {response.status_code}")
                return None
        except Exception as e:
            st.error(f"An error occurred: {e}")
            return None

    # Function to extract specific files (app.py or main.py) and read them
    def extract_and_read_file(zip_content, target_file="app.py"):
        with zipfile.ZipFile(io.BytesIO(zip_content)) as z:
            # Look for the target file (app.py or main.py)
            target_file_content = None
            for file in z.namelist():
                if target_file in file:
                    target_file_content = z.read(file).decode('utf-8')
                    break
            return target_file_content

    # Streamlit UI setup
    st.title("GitHub README Generator")
    st.write("Enter a GitHub repository URL to generate a README.md file based on the repository's main Python file.")

    # Input for GitHub repository URL
    repo_url = st.text_input("GitHub Repository URL", placeholder="https://github.com/username/repository")

    # Input for the file name (default to "app.py" or "main.py")
    target_file = st.text_input("File to Read (e.g., app.py, main.py)", value="app.py", placeholder="app.py or main.py")

    if st.button("Generate README"):
        if not repo_url:
            st.warning("Please enter a valid GitHub repository URL.")
        else:
            with st.spinner("Fetching repository data..."):
                zip_content = fetch_github_repo(repo_url)

            if zip_content:
                # Extract and read the target file (e.g., app.py)
                file_content = extract_and_read_file(zip_content, target_file)

                if file_content:
                    # Prepare the prompt for the Gemini API
                    prompt = (
                        f"Based on the following Python code from the file {target_file}, generate a professional `README.md` file.\n\n"
                        f"GitHub Repository URL: {repo_url}\n\n"
                        f"Content from {target_file}:\n{file_content}\n\n"
                        f"Ensure that the `README.md` includes sections like Introduction, Installation, Usage, Features, "
                        f"Contributing, License, and other relevant details based on the provided code content."
                    )

                    try:
                        with st.spinner("Generating README ..."):
                            response = model.generate_content(prompt)

                        # Display the generated README content
                        st.subheader("Generated README.md")
                        readme_content = response.text or "No content generated."
                        st.code(readme_content, language="markdown")

                        # Allow user to download the generated README
                        st.download_button(
                            label="Download README.md",
                            data=readme_content,
                            file_name="README.md",
                            mime="text/markdown",
                        )
                    except Exception as e:
                        st.error(f"An error occurred while generating the README: {e}")
                else:
                    st.error(f"The file {target_file} was not found in the repository.")


