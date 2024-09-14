from data_helper import WebScrapper
from vector_store_helper import QdrantHelper
from chatbot import LlmHelper
import asyncio
from typing import List

URL = ("https://www2.deloitte.com/us/en/insights/economy/global-economic-outlook/weekly-update/"
       "weekly-update-2023-10.html?icid=archive_click")
COLLECTION_NAME = "economic_outlook"


async def call_llm_with_multiple_questions(questions_list: List[str]):
    # Create a list of coroutines
    response_coroutines = [qa_chain.ainvoke(input=each_question) for each_question in questions_list]

    # Gather all the coroutines and run them concurrently
    llm_response_list = await asyncio.gather(*response_coroutines)

    return llm_response_list


if __name__ == '__main__':
    web_scrapper = WebScrapper()
    qdrant_helper = QdrantHelper()
    data = web_scrapper.get_text_from_webpage(URL)
    langchain_data = web_scrapper.get_data_from_webpage(URL)
    qdrant_helper.ingest_data_with_langchain_qdrant_client(collection_name=COLLECTION_NAME, data=data)
    llm_helper = LlmHelper()
    llm_helper.initialize_gemini_llm()
    qa_chain = llm_helper.create_qa_chain(vector_store=qdrant_helper.get_vector_store(collection_name=COLLECTION_NAME))

    with open('questions.txt', 'r') as questions_file:
        input_questions_list = questions_file.readlines()

    response_list = asyncio.run(call_llm_with_multiple_questions(input_questions_list))

    with open('responses.txt', 'w') as response_file:
        for answer_no, each_answer in enumerate(response_list):
            response_file.write(f"Response {answer_no + 1}: \n {each_answer} \n")

    print("Responses generated Successfully")
