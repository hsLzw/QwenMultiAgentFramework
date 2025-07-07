import re

from agents_core.agent_import import *
from agents_core.base_agent import BaseAgent
from models_core.chat_session import ChatSession


class CoordinatorAgent(BaseAgent):
	"""即时信息Agent 负责获取最新的时间、天气等即时信息。"""

	def __init__(self, model, coordinator_tools, name: str = "CoordinatorAgent", description="时信息Agent 负责获取最新的时间、天气等即时信息"):
		super().__init__(model, name=name, description=description)
		self.tools = coordinator_tools

	def get_system_tip(self, chat_session):
		plan = chat_session.get_plan()
		if plan:
			tips = '''
			**请深度思考**
			你的任务是根据assistant给出的步骤来匹配对应的工具(最近一次assistant的输出)
			1. 所提供的步骤格式如第一步为: Step1- <step description>
			2. 你要知道当前有哪些工具可用，每个工具的作用是什么，工具的作用记录在description
			3. 针对每一个步骤都选出对应的合适的工具,**不要幻想不存在的步骤**。
		  	4. 输出格式为Json。key为coordination_result, coordination_result的值为步骤所对应的工具列表。每个步骤-工具对应表都为一个json，包含step_id为步骤ID,，step_content为步骤描述，matched_agent为使用的工具列表(列表元素为工具名称)。请参照如下示例:
		  	```json
			   {
				 "coordination_result": [
				   {
					 "step_id": 步骤ID<int>,
					 "step_content": "步骤描述<text>",
					 "matched_agent": ["选用的工具名称<text>"]
				   }
				 ]
			   }
			```
			- 若历史对话中存在相关数据，无需调用工具
			- 若所有步骤无可用工具，请返回
			```json
			{"coordination_result": []}
			```
			
			'''
		else:
			tips = '''
			请按照用户要求选择出合适的工具来解决用户的问题
			'''
		return tips

	def get_input(self, chat_session, *args):
		system_ = {"role": "system", "content": self.get_system_tip(chat_session)}
		# 提取最新的用户对话 包含历史对话
		this_user_input = [*chat_session.get_current_input(), {"role": "user", "content": f"请基于以下步骤列表完成协调匹配:\n{chat_session.get_plan_str()}"}]
		return [system_, *this_user_input]

	def run(self, chat_session: ChatSession, *args) -> Any:
		response = self.generate(self.get_input(chat_session, *args), tools=self.tools, prompt=self.get_prompt(chat_session, *args))
		# 提取选择的agent，如果有
		agent_calls = self.get_angets(super().deal_response(response))
		return agent_calls["agent"]

	def get_angets(self, response):
		try:
			# 尝试直接处理
			text = response.strip().replace("```json", "")
			if text.endswith("```"):
				text = text[:-3]
			text = text.strip().replace("\n", "")
			json_data = json.loads(text)
			return {"response": response, "agent": json_data}
		except json.JSONDecodeError as e:
			return {"response": response, "agent": None}





