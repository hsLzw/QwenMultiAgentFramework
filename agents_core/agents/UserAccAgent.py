from agents_core.agent_import import *
from agents_core.agents_tools.user_tools.user_account_tool import UserAccountTool
from agents_core.base_agent import BaseAgent
from models_core.chat_session import ChatSession


class UserAccountAgent(BaseAgent):

	def __init__(self,
				 model,
				 name: str = "UserAccountAgent",
				 description="处理用户账号相关内容 包含功能:"
				 ):
		super().__init__(model, name=name, description=description)
		self.initialize_tools()

	def initialize_tools(self):
		# 加入时间工具
		self.tools.extend(UserAccountTool.get_tool_list())
		for i in self.tools:
			self.tools_map[i.name] = i
			self.description += f"{i.description},"

	def get_system_tip(self):
		return '''
		**请深度思考**
		你是一个工具助手，帮助选出正确的工具
		-. 请根据历史对话和本次提供的工具，选出当前最需要的一个工具和参数
		-. 若用户未提供工具所需的必要数据，请思考你是否可以代替用户自行提供
		-. 若用户未提供工具所需的必要数据，并且你不能代替用户提供工具所需的参数，请整理缺失参数的description并必须按照如下示例输出:
		```json
		{"need_args": "需要用户提供的内容描述<参数的description>"}
		```
		-. 若用户提供的内容存在调用工具所需的必要参数，请按照如下指引:
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
