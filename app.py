import os
import ast
import streamlit as st
from streamlit_mermaid import st_mermaid
from gtts import gTTS

# LangChain core contains main abstractions
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate

# LangChain community contains 3rd party integrations
from langchain_community.document_loaders import WebBaseLoader, YoutubeLoader
from langchain_community.vectorstores import Chroma # Embeddings Database
from langchain_community.chat_models import ChatAnthropic

# Recursive character splitter splits texts into smaller chunks
from langchain.text_splitter import RecursiveCharacterTextSplitter

# OpenAI
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

# Chains
from langchain.chains.combine_documents import create_stuff_documents_chain # prompt multi documents
from langchain.chains import create_history_aware_retriever, create_retrieval_chain


# App Config
st.set_page_config(page_title="AiDA", layout="wide")
st.title("AiDA: Your Genius Math Assistant!")


def get_vector_store(document):
    # Split the documents into chunks of text
    # Vectorise the text and store them in vector_store using openAI embeddings and Chroma
    # return the vector store
   
    # Split the text into smaller chunks to handle content larger than the model's context window
    text_splitter = RecursiveCharacterTextSplitter()
    document_chunks = text_splitter.split_documents(document)

    # Vectorize and store in Chroma
    vector_store = Chroma.from_documents(document_chunks, OpenAIEmbeddings())

    return vector_store

def classify_intent(user_input):
    # Return true if the user is trying to get a summary
    llm = ChatOpenAI()
    prompt = ChatPromptTemplate.from_messages([
            ("system","The following user request is out of context, is the user requesting a summary or something else?"),
            ("human", "{user_input}"),
            ("system", "Respond True if requesting summary, False for anything else")
    ])

    chain = prompt | llm
    response = chain.invoke({"user_input": user_input})
    return ast.literal_eval(response.content)

def get_context_retriever_chain(vector_store):
    # 1. Create a prompt that takes chat history
    # 2. Create a retriever chain with the prompt (chat history)
    # 3. return the retriever chain

    llm = ChatOpenAI()

    # call retriever method from vectore_store: an instance of class Chroma
    # .as_retriever() allows querying and retrieval operations on the stored vectors
    retriever = vector_store.as_retriever()

    # Pass the chat history to the prompt template via MessagesPlaceholder
    # Prepare a prompt that will use the context (chat history) and user's input to generate a search query
    # to search the vector store 
    prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"), # from session state
        ("human", "{input}"),
        ("human", "Given the above conversation, generate a search query to get the information relevant to the conversation")
    ])

    retriever_chain = create_history_aware_retriever(llm, retriever, prompt)
    # Note: the retriever is programmed now to generate variable context 
    return retriever_chain

def get_conversational_rag_chain(retriever_chain):
    # 1. Create a prompt that takes context (the output from retriever chain) and user input
    # 2. Create a document chain with prompt (context and user input)
    # 3. Create a retrieval chain with (the retriever chain for context and document chain)
    # 4. return the retrieval chain

    # llm = ChatOpenAI()
    llm = ChatAnthropic()

    # context is coming from the retriever chain
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer the below question based on the following context: \n\n{context} \n\n"),
        ("human", "{input}")
    ])

    stuff_documents_chain = create_stuff_documents_chain(llm, prompt)
    # Stuff document chains is now prompted to retrieve an answer to input based on context (form retriever)
    retrieval_chain = create_retrieval_chain(retriever_chain, stuff_documents_chain)
    # retrieval chain chained retriever chain <- stuff document chain
    return retrieval_chain

def get_response(user_input):
    # Contexts: Source and Chat history
    # 1. Create context retriever chain <- vector store: (Vectorized Document)
    # 2. Create conversational chain <- context retriever chain (source)
    # 3. Invoke the model (conversational chain) <- chat history and user's input
    # 4. return the llm's response

    # Note: The same user input is used twice, first to retrieve relevant information from vectore store
    # second to generate a response based on the context (retrieved information and chat history)
    
    retriever_chain = get_context_retriever_chain(st.session_state.vector_store)
    conversation_rag_chain = get_conversational_rag_chain(retriever_chain)

    response = conversation_rag_chain.invoke({
        "chat_history": st.session_state.chat_history,
        "input": user_input
    })

    return response['answer']

def strip_mermaid_wrapper(input_text):
    # Define the start and end markers of the mermaid content
    start_marker = "```mermaid"
    end_marker = "```"
    
    # Find the start and end positions of the actual content
    start_pos = input_text.find(start_marker) + len(start_marker)
    end_pos = input_text.rfind(end_marker)
    
    # Extract the content between the markers without trimming whitespace
    # The only change here is removing .strip() to preserve original formatting
    stripped_content = input_text[start_pos:end_pos]
    
    return stripped_content.strip()

def generate_chart(transcript):
    # Not in use due to inconsistent format 
    llm = ChatOpenAI()
    prompt = ChatPromptTemplate.from_messages([
            ("system","given this video script below, can you generate me a mermaid flowchart that describes the video script?"),
            ("human", "{transcript}"),
            ("system", "Produce a very accurate flowchart code"),
            ("system", "Output only the graph")
    ])

    chain = prompt | llm
    response = chain.invoke({"transcript": transcript})
    mermaid_code = strip_mermaid_wrapper(response.content)
    return mermaid_code

# Main
source = st.text_input("Paste your youtube video URL here:")


if source is None or source == "":
    st.info("Enter URL of a webpage, pdf, or a youtube video")

else:
    #if source == webpage:
    #loader = WebBaseLoader(source)

    # else if source == youtube:
    loader = YoutubeLoader.from_youtube_url(source)

    document = loader.load()
    transcript = document[0].page_content
    # Add / update chat_history and vector_store to session state
    if "chat_history" not in st.session_state:
        # Start of chat history
        st.session_state.chat_history = [
            AIMessage(content="Hello, I am AIDA, your AI Math Assistant, how can I help?")
        ]
    
    if "vector_store" not in st.session_state:
        # Chunk documents and store its vectors in vectore_store
        st.session_state.vector_store = get_vector_store(document)

    # User Input
    user_input = st.chat_input("Message AIDA")
    
    summary_intent = False
    if user_input is not None and user_input != "":
        # First, classify the intent
        summary_intent = classify_intent(user_input)
               
        # Get the response
        response = get_response(user_input)

        # Add the user query to the chat history
        st.session_state.chat_history.append(HumanMessage(content=user_input))

        # Add the AI response to chat history
        st.session_state.chat_history.append(AIMessage(content=response))

        
    # Video Frame
    col1, col2, col3 = st.columns([2,4,2])

    with col2: # This column is effectively controlling the width of the video
        st.video(source)
    # st.video(source)

    # Conversation
    for message in st.session_state.chat_history:
        # Fetch the AI's response if its the AI's turn
        if isinstance(message, AIMessage):
            with st.chat_message("AI"):
                st.write(message.content)

        # Write the user's input if its the user's turn
        elif isinstance(message, HumanMessage):
            with st.chat_message("Human"):
                st.write(message.content)
        
    # If the user wanted a summary, sketch an explainer chart
    if summary_intent:
        # This function is inconsistent, for demoing purposes we will use manual graph
        # mermaid_code = generate_chart(transcript)
        mermaid_code = """
        graph TD
        A[Start: Randomly Orient Cube] --> B[Consider Cube's Shadow]
        B --> C[Area of Shadow in 3D Space]
        C --> D{Is Light Source Infinite?}
        D -- Yes --> E[Shadow as Flat Projection onto xy-plane]
        D -- No --> F[Depends on Light Source Position]
        E --> G[Calculate Expected Value of Area]
        G --> H[Repeat Process]
        H --> I[Measure Areas]
        I --> J[Tally Up Areas]
        J --> K[Approach Empirical Mean]
        K --> L[Explore Problem-Solving Styles]
        L --> M[Discuss Bias in Mathematical Popularizations]
        M --> N[End: Shadow Puzzle Context]
        """
        st_mermaid(mermaid_code, key="example")

        # Play text to speech
        if isinstance(message, AIMessage):
            with st.chat_message("AI"):
                tts = gTTS(text=message.content, lang='en')
                audio_file = 'response.mp3'
                tts.save(audio_file)

                audio_file_path = os.path.join(os.getcwd(), audio_file)
                st.audio(audio_file_path, format='audio/mp3', start_time=0)


    
