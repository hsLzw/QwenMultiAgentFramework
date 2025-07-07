from agents_core.agent_import import *
from agents_core.base_agent import BaseAgent
from models_core.chat_session import ChatSession


class EvaluationAgent(BaseAgent):
	"""评估Agent，负责评估结果质量并决定是否需要进一步处理"""

	def __init__(self, model, name: str = "EvaluationAgent", description: str = "EvaluationAgent"):
		# 定义评估结果的结构
		response_schemas = [
			ResponseSchema(name="is_complete", description="结果是否完整解决问题，特殊情况：若因用户提供的数据不足以解决问题，则认为本次问题已解决。"),
			ResponseSchema(name="next_step", description="如需要下一步的话，下一步建议"),
			ResponseSchema(name="confidence", description="对结果完整性的信心程度，0-100"),
			ResponseSchema(name="feedback", description="对结果的详细反馈")
		]

		self.output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
		self.format_instructions = self.output_parser.get_format_instructions()

		super().__init__(model, name=name, description=description)

	def get_evaluate(self, response):

		try:
			# 解析结构化输出
			evaluation = self.output_parser.parse(response)
			return evaluation
		except Exception as e:
			# 如果解析失败，返回默认评估
			return {
				"is_complete": False,
				"next_step": "无法确定结果是否完整，请提供更多信息",
				"confidence": 50,
				"feedback": f"评估解析失败: {str(e)}",
				"is_wrong": True
			}

	def get_system_tip(self):
		return f'''
		你是一个评估对话的助手，请根据历史聊天评估是否完整回答了用户问题，需要注意以下几点:
			- 指定的相关步骤是否全部执行了
			- 每个工具调用的结果是否能够满足回答用户问题的需要
		
		{self.format_instructions}
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
		response = self.generate(self.get_input(chat_session, *args), tools=self.tools, temperature=0.7, prompt=self.get_prompt(chat_session, *args))
		# 提取计划 如有
		evaluate_status = self.get_evaluate(super().deal_response(response))
		return evaluate_status




