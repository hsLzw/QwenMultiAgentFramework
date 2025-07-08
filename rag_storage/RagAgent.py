import re

from agents_core.agent_import import *
from agents_core.base_agent import BaseAgent
from models_core.chat_session import ChatSession
from rag_storage.storage_lib import RAGLib


class SearchRAG(BaseAgent):
	"""RAG 检索增强生成"""

	def __init__(self, model, name: str = "SearchRAG",
				 description="SearchRAG"):

		super().__init__(model, name=name, description=description)
		# 初始化向量库
		self.storage = RAGLib("rag_storage/storage/")

	def get_system_tip(self, chat_session):
		user_data = chat_session.get_near_user()
		content = user_data["content"]
		# 检索
		doc_content = self.storage.search_query(content)
		tips = f'''
		**请深度思考**
		请结合历史对话并根据下列内容，整理汇总与用户问题相关的内容，汇总最多200字：
		{doc_content}
		
		约束: 生成前请思考是否符合文档描述的逻辑。不要明文标注有多少字符。
		'''
		return tips

	def get_input(self, chat_session, *args):
		system_ = {"role": "system", "content": self.get_system_tip(chat_session)}
		# 提取最新的用户对话 包含历史对话
		this_user_input = [*chat_session.get_current_input()]
		return [system_, *this_user_input]

	def run(self, chat_session: ChatSession, *args) -> Any:
		response = self.generate(self.get_input(chat_session, *args), tools=self.tools,
								 prompt=self.get_prompt(chat_session, *args))

		# ret_data = f'''<|object_ref_start|>{self.deal_response(response)}<|object_ref_end|>'''
		# print(ret_data)
		return self.deal_response(response)



