import os
from dotenv import load_dotenv

load_dotenv()

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough



def create_rag_chain():

    # Embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Load vector DB
    vectordb = Chroma(
        persist_directory="chroma_db",
        embedding_function=embeddings
    )

    retriever = vectordb.as_retriever(search_kwargs={"k": 3})

    # LLM
    llm = ChatGroq(
        groq_api_key=os.getenv("GROQ_API_KEY"),
        model_name="llama-3.1-8b-instant",
    )

    # Prompt
    prompt = ChatPromptTemplate.from_template(
        """
        Answer the question using ONLY the context below.

        Context:
        {context}

        Question:
        {question}
        """
    )

    # Format retrieved docs into string
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # LCEL chain (Modern LangChain)
    rag_chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain
