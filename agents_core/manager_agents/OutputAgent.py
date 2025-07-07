import re

from agents_core.agent_import import *
from agents_core.base_agent import BaseAgent
from models_core.chat_session import ChatSession


class OutputAgent(BaseAgent):

	def __init__(self, model, name: str = "OutputAgent", description="将过程总结输出"):
		super().__init__(model, name=name, description=description)

	def get_system_tip(self):
		return '''
		你是一个专业的整理归纳的助手，请你帮忙根据历史对话整理出能够回答用户问题的回复
		如果存在调用工具的情况，你需要注意以下几点：
		-. 请忽视assistant中途出现的问题(如xx失败,无法xx)，这时请你直接按照用户的诉求进行回复
		-. 根据对话历史，答复用户的问题(最近的一次)
		如果不存在调用工具的情况，请根据历史对话答复用户！
		'''

	def get_input(self, chat_session, *args):
		# 提取最新的用户对话 包含历史对话
		this_user_input = chat_session.get_current_input()
		# 创建system提示
		system_tip = {"role": "system", "content": self.get_system_tip()}
		return [system_tip, *this_user_input]

	def run(self, chat_session: ChatSession, *args) -> Any:
		"""
		执行推理并返回结果
		"""
		response = self.generate(self.get_input(chat_session, *args), tools=self.tools, temperature=0.6, prompt=self.get_prompt(chat_session, *args))
		return response

