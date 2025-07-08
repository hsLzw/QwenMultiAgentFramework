import os
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFacePipeline
from transformers import pipeline
import chromadb
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma


class RAGLib:
	'''未继承多库多collection处理 demo'''

	def __init__(self, storage_path):
		# 向量库位置
		self.storage_path = storage_path
		self.collection_name = "test"
		self.client = None
		self.vector_db = None
		self.init_storage()

	def init_storage(self):
		# 初始化客户端
		self.client = chromadb.PersistentClient(path="rag_storage/storage/rag_vec_db")
		# 心跳
		self.client.heartbeat()
		# embeddings模型 与创建向量数据库时保持一致
		embedding_model = HuggingFaceEmbeddings(
			# 支持多语言，384维向量
			model_name="models/sentence-transformers/all-MiniLM-L6-v2",
			model_kwargs={'device': 'cuda:0'},
			# 归一化向量，提升相似度计算精度
			encode_kwargs={'normalize_embeddings': True}
		)
		# 创建db检索
		self.vector_db = Chroma(
			client=self.client,
			collection_name=self.collection_name,
			embedding_function=embedding_model  # 指定 embedding 模型
		)

	def search_query(self, query, delimiter_start="<|object_ref_start|>", delimiter_end="<|object_ref_end|>"):
		results = self.vector_db.similarity_search(query, k=3)
		result_str = ""
		for i in results:
			result_str += f'''{delimiter_start}\ndoc_id:{i.metadata["id"]}\ndoc_content:{i.page_content}\n{delimiter_end}\n'''
		return result_str












