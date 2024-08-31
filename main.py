import warnings

warnings.filterwarnings("ignore")
from pprint import pprint

from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders.parsers import LanguageParser
from langchain_text_splitters import (
    Language,
    RecursiveCharacterTextSplitter,
)
loader = GenericLoader.from_filesystem(
    "/Users/markmcmurray/workspace/forks/helm-issue-13241",
    glob="**/*",
    suffixes=[".go"],
    parser=LanguageParser(language=Language.GO),
)
docs = loader.load()


go_splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.GO, chunk_size=60, chunk_overlap=0
)

result = go_splitter.split_documents(docs)

