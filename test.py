'''
向量库使用测试
'''
import sqlite3
import os
import chromadb
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from chromadb.utils.embedding_functions import CohereEmbeddingFunction
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFacePipeline
from langchain.schema import Document
from transformers import pipeline

# 配置路径
SQLITE_PATH = "rag_storage/storage/rag_vec_db/chroma.sqlite3"  # SQLite数据库路径
VECTOR_DB_DIR = "rag_storage/storage/rag_vec_db/"  # 向量数据库持久化路径
COLLECTION_NAME = "test"

embedding_model = HuggingFaceEmbeddings(
    # 支持多语言，384维向量
    model_name="models/sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={'device': 'cuda:0'},
    # 归一化向量，提升相似度计算精度
    encode_kwargs={'normalize_embeddings': True}
)

client = chromadb.PersistentClient(path="rag_storage/storage/rag_vec_db")
client.heartbeat()

vector_db = Chroma(
    client=client,
    collection_name="test",
    embedding_function=embedding_model  # 指定 embedding 模型
)

results = vector_db.similarity_search("什么是量子？", k=3)

print(results)
for i in results:
    print(i.page_content)
    print(i.metadata["source"])
    print(i.metadata["id"])








