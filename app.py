from flask import Flask, render_template,jsonify, request, session
from flask_session import Session
from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_groq import ChatGroq
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.messages import AIMessage,HumanMessage
from dotenv import load_dotenv
from src.prompt import *
import os


app = Flask(__name__)
app.secret_key = 'pcsk_2VxGKZ_Q1fXmNN66sdzST55PwFrAtCSicvedzB5YkUYcn3wjtR2hCfoS2PdsTdU3sD44F5' 
app.config['SESSION_TYPE'] = 'filesystem' 
Session(app)
load_dotenv()

GROQ_API_KEY="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"
PINECONE_API_KEY="pcsk_2VxGKZ_Q1fXmNN66sdzST55PwFrAtCSicvedzB5YkUYcn3wjtR2hCfoS2PdsTdU3sD44F5"

os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
os.environ["PINECONE_API_KEY"] = os.getenv("PINECONE_API_KEY")

embeddings=download_hugging_face_embeddings()

index_name="medicalbot"

docsearch= PineconeVectorStore.from_existing_index(
    embedding=embeddings,
    index_name=index_name
)
retriever=docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 2})
llm=ChatGroq(model_name="llama3-8b-8192",temperature=0.4,max_tokens=500)

prompt=ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}")
    ]
)

question_answer_chain= create_stuff_documents_chain(llm,prompt)
rag_chain= create_retrieval_chain(retriever,question_answer_chain)


hiostory_aware_retriever=create_history_aware_retriever(llm,retriever,contextualize_q_prompt)

question_answer_chain=create_stuff_documents_chain(llm,contextualize_c_prompt)
rag_chain=create_retrieval_chain(hiostory_aware_retriever, question_answer_chain)



@app.route('/')
def index():
    return render_template('chatbot.html')

@app.route("/get", methods=["GET", "POST"])
def get_response():
    msg = request.form["msg"]
    input_text = msg

    # Initialize chat history if not present
    if "chat_history" not in session:
        session["chat_history"] = []

    # Convert session's chat_history to actual LangChain message objects
    chat_history = [
        HumanMessage(content=entry["user"]) if i % 2 == 0 else AIMessage(content=entry["bot"])
        for i, entry in enumerate(session["chat_history"])
    ]

    # Get response from model
    response = rag_chain.invoke({
        "input": input_text,
        "chat_history": chat_history
    })

    # Append to chat history
    session["chat_history"].extend([
        {"user": input_text},
        {"bot": response['answer']}
    ])

    return response['answer']


if __name__ == "__main__":
    app.run(host="0.0.0.0",port=8080,debug=True)