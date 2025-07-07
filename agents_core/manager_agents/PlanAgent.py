import re

from agents_core.agent_import import *
from agents_core.base_agent import BaseAgent
from models_core.chat_session import ChatSession


class PlanAgent(BaseAgent):
	"""即时信息Agent 负责获取当前的日期时间、天气等即时信息。"""

	def __init__(self, model, tools, name: str = "PlanAgent", description="解析用户诉求，拆分任务"):
		super().__init__(model, name=name, description=description)
		self.tools = tools

	def get_plan(self, response):
		pattern = r'```json(.*?)```'
		match = re.search(pattern, response, re.DOTALL)
		if not match:
			# 尝试直接处理
			text = response.strip().replace("\n", "")
		else:
			text = match.group(1).strip().replace("\n", "")
		try:
			json_data = json.loads(text)
			return {"response": response, "match": text, "plan": json_data}
		except json.JSONDecodeError as e:
			return {"response": response, "match": "", "plan": None}

	def get_system_tip(self):
		return '''
		你是一个专业的任务拆解助手。你的任务是：
		-. 解析用户的核心意图, 识别用户的具体需求
		-. 将用户的需求拆解为一个或多个步骤
		-. 拆分成多个步骤时，注意前后语义的连贯，不要将关键内容丢失
		-. 工具集是不可用的，但你可以根据工具集中的description来拆分任务。
		-. 步骤中不可以出现工具调用, 你的任务是指定步骤，只需要描述步骤即可。
		-. 若可以使用工具解决用户的需求，则必须使用以下JSON格式输出,请不要修改框架,请不要添加额外参数,请不要修改key：
		```json
		{"plan": [{"step": "步骤1", "step_id": 1}, {"step": "步骤2", "step_id": 2}, {"step": "步骤3", "step_id": 3}]}
		```
		-、若你无法根据现有工具集提出有效的解决步骤，请必须按照如下JSON格式输出,即给出空列表,请不要修改框架:
		```json
		{"plan": []}
		```
		-.禁止输出中指定工具的名称和参数。
		-.禁止做无用的思考，只需要拆分用户需求即可，所有工具均不需要参数，不需要你来考虑实际操作需要什么参数。
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
		# 提取计划 如有
		plan_calls = self.get_plan(super().deal_response(response))
		plan = plan_calls["plan"]

		return plan

