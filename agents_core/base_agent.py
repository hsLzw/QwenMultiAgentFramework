import re

from agents_core.agent_import import *
from models_core.chat_session import ChatSession


class BaseAgent:
	"""基础Agent类，定义通用接口"""

	def __init__(self, model, name: str, description: str, tools: List[Tool] = None, memory=None):
		self.name = name
		self.description = description
		self.tools = tools or []
		self.tools_map = {}
		self.memory = memory or ConversationBufferMemory(memory_key="chat_history")
		self.agent_executor = None
		self.max_iterations = 3  # 最大迭代次数

		# 根据use_vllm参数选择LLM
		self.llm = model
		# generate
		self.generate = self.llm.generate

	def initialize(self):
		"""初始化Agent的推理能力"""
		self.agent_executor = initialize_agent(
			self.tools,
			self.llm,
			agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
			verbose=True,
			memory=self.memory
		)

	def get_prompt(self, chat_session, *args):
		return ""

	def get_input(self, chat_session, *args):
		return chat_session.get_current_input()

	def run(self, chat_session: ChatSession, *args) -> Any:
		"""
		执行推理并返回结果
		"""
		response = self.generate(self.get_input(chat_session, *args), tools=self.tools, prompt=self.get_prompt(chat_session, *args))
		return self.deal_response(response)

	def get_agent_info(self) -> Dict:
		"""返回Agent的基本信息"""
		return {
			"name": self.name,
			"tool_count": len(self.tools)
		}

	def get_tool_call(self, response):
		pattern = r'<tool_call>(.*?)</tool_call>'
		match = re.search(pattern, response, re.DOTALL)
		if not match:
			return {"response": response, "match": "", "tool_call": None}
		try:
			# 提取标签内的字符串并转换为 JSON
			tool_content = match.group(1).strip()
			json_data = json.loads(tool_content)
			return {"response": response, "match": tool_content, "tool_call": json_data}
		except json.JSONDecodeError as e:
			return {"response": response, "match": "", "tool_call": None}


	def deal_response(self, response):
		if "<think>" in response:
			response = response.split("<think>")[-1].strip()
		if "</think>" in response:
			response = response.split("</think>")[-1].strip()
		return response



