'''
创建向量库
单独demo，项目中仅使用向量库，不进行创建操作。
'''
import os
import json
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.schema import Document


def get_create_path(db_path, file_name, exist_to_raise=True):
	file_path = os.path.join(db_path, file_name)
	if os.path.exists(file_path):
		print(f"Warning: 存在向量库{file_name}")
		if exist_to_raise:
			raise Exception(f"{file_name}已存在！")
	return file_path


def get_file_to_json(documents_path, max_chunk_size, chunk_overlap):
	# 分割
	text_splitter = RecursiveCharacterTextSplitter(
		# 最大
		chunk_size=max_chunk_size,
		# 反向覆盖 防止语义断层
		chunk_overlap=chunk_overlap,
		# 中文分割符
		separators=["。", "；", "，", " "]
	)
	# 读取
	with open(documents_path, "r", encoding="utf-8") as f:
		data = f.readlines()
	# 提取数据
	documents = []
	for one_seq in data:
		# 不做异常处理
		this_data = json.loads(one_seq)
		# 提取
		doc_id = this_data["id"]
		doc = this_data["doc"]
		question = this_data["doc"]
		answer = this_data["answer"]
		# 组合
		content = f"文档数据：{doc}\n问题：{question}\n答案：{answer}"  # 补充QA对增强检索
		# 文档长度分割
		chunks = text_splitter.split_text(content)
		# 为每个片段添加元数据（原id+片段序号，方便溯源）
		for i, chunk in enumerate(chunks):
			documents.append({
				"id": f"{doc_id}_chunk{i}",
				"text": chunk,
				"source": doc_id
			})
	return documents


def text_embeddings(documents_path, db_path, collection_name, max_chunk_size, chunk_overlap):
	# 获取文档
	documents = get_file_to_json(documents_path, max_chunk_size, chunk_overlap)
	# 初始化嵌入模型（使用sentence-transformers的开源模型） 要跟程序中一致
	embedding_model = HuggingFaceEmbeddings(
		# 支持多语言，384维向量
		model_name="models/sentence-transformers/all-MiniLM-L6-v2",
		model_kwargs={'device': 'cuda:0'},
		# 归一化向量，提升相似度计算精度
		encode_kwargs={'normalize_embeddings': True}
	)
	# 转换为langchain的Document格式（需text和metadata）
	langchain_docs = [
		Document(
			page_content=chunk["text"],
			metadata={"id": chunk["id"], "source": chunk["source"]}
		) for chunk in documents
	]

	# 初始化向量数据库（持久化存储到本地目录./rag_chroma_db）
	vector_db = Chroma.from_documents(
		documents=langchain_docs,
		embedding=embedding_model,
		persist_directory=db_path,
		collection_name=collection_name
	)
	# 当前版本已自动持久化到磁盘
	# vector_db.persist()


if __name__ == "__main__":
	storage_path = "rag_storage/storage/"
	# 创建名称
	db_name = "rag_vec_db"
	# 集合名（方便多数据集管理）
	collection_name = "test"
	# 实际路径
	db_path = get_create_path(storage_path, db_name)
	# 文档路径 单文件提取，多文件需自行调整
	documents_path = "rag_storage/demo/PhysicBench.json"
	# 提取创建
	max_chunk_size = 200
	chunk_overlap = 20
	text_embeddings(documents_path, db_path, collection_name, max_chunk_size, chunk_overlap)















