from agents_core.agent_import import *
from agents_core.agents_tools.current_tools.time_tool import TimeTool
from agents_core.base_agent import BaseAgent
from models_core.chat_session import ChatSession


class CurrentAgent(BaseAgent):

	def __init__(self,
				 model,
				 name: str = "CurrentAgent",
				 description="即时信息Agent 包含功能:"
				 ):
		super().__init__(model, name=name, description=description)
		self.initialize_tools()

	def initialize_tools(self):
		# 加入时间工具
		self.tools.extend(TimeTool.get_tool_list())
		for i in self.tools:
			self.tools_map[i.name] = i
			self.description += f"{i.description},"

	def get_system_tip(self):
		return '''
		你是一个工具助手，帮助选出正确的工具
		请根据历史对话 和 tool_calls提供的工具，帮用户选出当前最需要的一个工具和参数
		输出需严格遵循下述要求
		'''

	def get_input(self, chat_session, *args):
		# 提取最新的用户对话 包含历史对话
		this_user_input = chat_session.get_current_input()
		# 创建system提示
		system_tip = {"role": "system", "content": self.get_system_tip()}
		return [system_tip, *this_user_input]

	def run(self, chat_session: ChatSession, *args) -> Any:
		"""
		即时信息Agent 负责获取当前的日期、时间、天气等即时信息,也能够对日期、时间等内容进行相关计算。

		Args:

		Return:
			返回具体要调用的工具
		"""
		response = self.generate(self.get_input(chat_session, *args), tools=self.tools, prompt=self.get_prompt(chat_session, *args))
		# 提取选择的agent，如果有
		func_calls = self.get_tool_call(super().deal_response(response))

		# print("选择的Agent:", agent_calls["agent"])
		# return agent_calls["agent"])
		return func_calls
