from data_helper import WebScrapper
from vector_store_helper import QdrantHelper
from chatbot import LlmHelper

URL = "https://indianexpress.com/article/political-pulse/pm-modi-haryana-rallies-bjp-assembly-elections-9567166/?ref=hometop_hp"
COLLECTION_NAME = "webscrapper_chatbot"


if __name__ == '__main__':
    web_scrapper = WebScrapper()
    qdrant_helper = QdrantHelper()
    data = web_scrapper.get_text_from_webpage(URL)
    langchain_data = web_scrapper.get_data_from_webpage(URL)
    qdrant_helper.ingest_data_with_langchain_qdrant_client(collection_name=COLLECTION_NAME, data=data)
    llm_helper = LlmHelper()
    llm_helper.initialize_gemini_llm()
    qa_chain = llm_helper.create_qa_chain(vector_store=qdrant_helper.get_vector_store(collection_name=COLLECTION_NAME))
    response = qa_chain.invoke(input="What is the importance of elections in Haryana")
