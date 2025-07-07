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
		**请深度思考**
		你是一个专业的任务拆解助手。你的任务是：
		1. 仅基于用户的原始需求进行步骤拆分，步骤必须与工具列表中**功能直接相关**的工具对应（无关工具的功能不得纳入步骤）。
		2. 步骤描述只能是用户需求的直接拆解，**绝对禁止添加任何需求中未提及的操作**（如“验证信息”“收集参数”等均属于禁止内容）。
		3. 步骤描述中**完全不能出现任何工具名称、工具功能细节、参数信息**，仅描述需求本身的拆分动作。
		4. 输出格式为JSON，key为plan，值为步骤列表。每个步骤包含“step”（步骤描述）和“step_id”（顺序）。示例：
		```json
		{"plan": [{"step": "步骤1描述", "step_id": 1}, {"step": "步骤2描述", "step_id": 2}]}
		```
		5. 若用户需求仅对应一个工具功能，则步骤列表只能有一个步骤；若需求不对应任何工具功能，plan 值为空列表。
		6. 输出前请逐条检查：
			-. 是否包含无关工具对应的步骤？
			-. 是否提及工具名称 / 参数 / 功能？
			-. 是否添加了需求外的操作？
			-. 格式是否严格符合 JSON 要求？
			-. 若有任何一项不符合，立即修正。
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
		response = self.generate(self.get_input(chat_session, *args), tools=self.tools, temperature=0.5, prompt=self.get_prompt(chat_session, *args))
		# 提取计划 如有
		plan_calls = self.get_plan(super().deal_response(response))
		plan = plan_calls["plan"]

		return plan

