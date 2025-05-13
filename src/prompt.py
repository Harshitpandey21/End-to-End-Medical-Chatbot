from langchain_core.prompts import MessagesPlaceholder
from langchain_core.prompts import ChatPromptTemplate
system_prompt=(
    """
    You are a medical assistant.
    You are given a question and some context.
    You need to answer the question based on the context.
    If you don't know the answer, say "I don't know".
    Keep tyhe answer concise.
    \n\n
    {context}
    """
)

contextualize_q_system_prompt=(
    "Given a chat history and the latets user question"
    "which might reference context in the chat history,"
    "formulate a standalone question which can be understood"
    "without the chat history.Do not answer the question."
    "just reformulate it if needed and otherwise return it as it is."
)

contextualize_q_prompt=ChatPromptTemplate.from_messages([
    ("system", contextualize_q_system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human",  "{input}")
])

contextualize_c_prompt=ChatPromptTemplate.from_messages([
    ("system",system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human",  "{input}")
])