import requests
from bs4 import BeautifulSoup
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List


class WebScrapper:
    def __init__(self):
        pass

    @staticmethod
    def get_text_from_webpage(url: str) -> str:
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        text = ' '.join([p.text for p in soup.find_all('p')])
        text = text.strip()

        return text

    @staticmethod
    def get_data_from_webpage(url: str):
        loader = WebBaseLoader(url)
        data = loader.load()
        splits = split_data(data)

        return splits


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def split_text(data: str) -> List:
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    splits = text_splitter.split_text(data)

    return splits


def split_data(data) -> List:
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    splits = text_splitter.split_documents(data)

    return splits
