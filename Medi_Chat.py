import os
import warnings
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.embeddings import HuggingFaceEmbeddings

from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

import gradio as gr


load_dotenv()
warnings.filterwarnings('ignore')

GENAI_API_KEY = os.getenv("GENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX_NAME)
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vectorstore = PineconeVectorStore.from_existing_index(index_name=PINECONE_INDEX_NAME, embedding=embedding_model)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

model = ChatGoogleGenerativeAI(
    model="models/gemini-2.0-flash",
    temperature=0.4,
    max_output_tokens=500,
    convert_system_message_to_human=True,
    google_api_key=GENAI_API_KEY
)

system_prompt = (
    # "You are an assistant for question-answering tasks for Medical Drug like a doctor. "
    # "Use the following pieces of retrieved context to answer the question "
    # "If you don't know the answer, say that you don't know."
    # "Use three sentences maximum and keep the answer concise."
    """ You are a helpful assistant that answers medical drug-related questions like a qualified doctor. 
    Use the retrieved context to provide concise, accurate answers in no more than three sentences. 
    If the question relates to a disease, symptom, or treatment beyond the context, politely respond with: 'I’m sorry, but I cannot provide medical advice. 
    Please consult a healthcare professional for proper diagnosis and treatment.' """

    "\n\n"
    "{context}"
)
retriever_prompt = (
    "Given a chat history and the latest user question which might reference context in the chat history,"
    "formulate a standalone question which can be understood without the chat history."
    "Do NOT answer the question, just reformulate it if needed and otherwise return it as is."
)

contextualize_q_prompt  = ChatPromptTemplate.from_messages(
    [
        ("system", retriever_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),


     ]
)
history_aware_retriever = create_history_aware_retriever(model,retriever,contextualize_q_prompt)
qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
question_answer_chain = create_stuff_documents_chain(model, qa_prompt)
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()

    return store[session_id]

conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
)

# while True:
#     user_input = input("Ask a question (or type 'exit' to quit): ")
#     if user_input.lower() == "exit":
#         print("Exiting chat...")
#         break
#     else:
#         try:
#             response = conversational_rag_chain.invoke(
#                 {"input": user_input},
#                 config={
#                     "configurable": {"session_id": "001"}
#                 },
#             )
#             print(f"AI: {response['answer']}")
#         except Exception as e:
#             print(f"An error occurred: {e}")
                       


session_id_counter = 1  
chat_sessions = {} 
session_titles = {} 

def chat_with_bot(user_input, chatbot, session_id_state, chat_history_list):
    try:
        session_id = session_id_state[0]
        if session_id not in chat_sessions:
            chat_sessions[session_id] = []  # Create new session history

        response = conversational_rag_chain.invoke(
            {"input": user_input},
            config={"configurable": {"session_id": session_id}},
        )
        chat_sessions[session_id].append((user_input, response['answer']))

       
        if session_id in session_titles and session_titles[session_id] == "New Chat":
            session_titles[session_id] = user_input[:15] + "..."  

        return (
            chat_sessions[session_id],
            session_id_state,
            gr.update(choices=list(session_titles.values()), value=session_titles[session_id])  # Ensure valid value
        )
    except Exception as e:
        return chat_sessions.get(session_id, []), session_id_state, gr.update(choices=list(session_titles.values()))

def new_chat(chatbot, session_id_state, chat_history_list):
    global session_id_counter
    session_id_counter += 1
    new_session_id = str(session_id_counter)
    chat_sessions[new_session_id] = []  # Store new session
    session_titles[new_session_id] = "New Chat"  

    session_id_state[0] = new_session_id  

    return (
        [],
        session_id_state,
        gr.update(choices=list(session_titles.values()), value="New Chat")  
    )

def load_chat_history(selected_title):
    session_id = next((sid for sid, title in session_titles.items() if title == selected_title), None)
    if session_id and session_id in chat_sessions:
        return chat_sessions[session_id], [session_id]
    return [], [session_id]

with gr.Blocks() as demo:
    gr.Markdown("# 🩺 Medical ChatBot")
    gr.Markdown("### Ask me anything about Medical.")
    gr.Markdown()
    
    with gr.Row():
        with gr.Column(scale=1, min_width=200): 
            gr.Markdown("### Chat History")
            chat_history_list = gr.Dropdown(choices=[], label="Previous Chats")  
            new_chat_button = gr.Button("New Chat")

        with gr.Column(scale=3):  
            chatbot = gr.Chatbot(label="Chat with the Medical Bot")
            msg = gr.Textbox(label="Your Query")
            clear = gr.Button("Clear")

    session_id_state = gr.State([str(session_id_counter)])  

    new_chat_button.click(new_chat, [chatbot, session_id_state, chat_history_list], [chatbot, session_id_state, chat_history_list])
    msg.submit(chat_with_bot, [msg, chatbot, session_id_state, chat_history_list], [chatbot, session_id_state, chat_history_list])
    chat_history_list.change(load_chat_history, chat_history_list, [chatbot, session_id_state])
    clear.click(lambda: [], None, chatbot, queue=False)

    demo.launch()
