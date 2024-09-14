from langchain_google_genai import GoogleGenerativeAI
from data_helper import format_docs
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

RAG_TEMPLATE = """
    You are an assistant for question-answering tasks. Use the following pieces of retrieved 
    context to answer the question. If you don't know the answer, just say that you don't know.
    
    <context>
    {context}
    </context>
    
    Answer the following question:
    
    {question}
    
"""

rag_prompt = ChatPromptTemplate.from_template(RAG_TEMPLATE)


class LlmHelper:
    def __init__(self, llm=None):
        self.llm = llm

    def initialize_gemini_llm(self):
        llm = GoogleGenerativeAI(model='gemini-1.5-flash', temperature=0)
        self.llm = llm

    def chat(self, query: str):
        response = self.llm.invoke(query)
        return response

    def create_qa_chain(self, vector_store):
        qa_chain = (
            {"context": vector_store.as_retriever() | format_docs, "question": RunnablePassthrough()}
            | rag_prompt
            | self.llm
            | StrOutputParser()
        )

        return qa_chain

