import re

from agents_core.agent_import import *
from agents_core.base_agent import BaseAgent
from models_core.chat_session import ChatSession


class HistorySummarizeAgent(BaseAgent):

	def __init__(self, model, name: str = "HistorySummarizeAgent", description="HistorySummarizeAgent"):
		super().__init__(model, name=name, description=description)

	def get_system_tip(self):
		return '''
		**请深度思考**
		你是一个专业的重点信息提取助手
		请根据所提供的历史对话进行归纳总结，提取出有效的数据，遵循以下原则：
		1、忽略工具调用流程，仅记录成功调用工具时产生的结果(或调用工具出现问题的原因)进行总结，每个工具结果的总结最多100字且不能出现历史对话中不存在的内容。
		2、对每条用户的输入进行总结，每条输入的总结最多不能超过100字且不能出现历史对话中不存在的内容。
		3、除工具结果、用户输入外的其余内容，进行归纳总结，总结出1条数据，不可超过100字且不能出现历史对话中不存在的内容。
		4、针对不同的总结内容，请输出Json数据，严格按照下述模板格式输出:
		```json
		{
			"summarize": [
				{"role": "user", "content": "<用户输入1总结的内容>"},
				{"role": "user", "content": "<用户输入2总结的内容>"},
				{"role": "tool_call", "content": "<工具结果2的总结的内容>"},
				{"role": "tool_call", "content": "<工具结果1的总结的内容>"},
				{"role": "other_summarize", "content": "<其他内容的汇总>"}
			]
		}
		** 请输出json数据 **
		```
		5、输出时禁止修改任何模板格式中的key，只可以增加列表的内容和更改content对应的值
		6、除了针对其他内容的汇总，【用户输入、工具结果】的输出顺序严格按照历史对话的先后顺序写入到summarize中。
		7、严禁出现历史对话中不存在的内容。
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
		print(response)
		return response

