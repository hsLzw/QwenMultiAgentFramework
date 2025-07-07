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
			你是CoAgent，核心职责是**协调步骤与工具Agent的匹配**。你的工作流程如下：

			1. **输入处理**：接收AnAgent输出的结构化步骤列表（格式参考：[{"step_id": 1, "step": "步骤描述"}, ...]）
			2. **步骤遍历**：逐一分析每个步骤的核心需求（如“查询信息”“计算数据”“生成内容”“操作工具”等）。
			3. **工具Agent匹配**：
			   - 针对当前步骤，从可用工具Agent列表中筛选最适配的1个候选Agent（例如：“信息查询步骤”匹配“SearchAgent”，“数据计算步骤”匹配“CalcAgent”）。
			   - 匹配依据：工具Agent的核心能力（如“SearchAgent擅长实时信息检索”）、步骤需求的复杂度（如“简单计算”可匹配轻量CalcAgent，“复杂建模”匹配专业ModelAgent）。
			4. **工具选择指令生成**：对匹配的工具Agent，生成具体工具调用指令（如“SearchAgent需调用天气API查询北京未来3天天气”）。
			5. **输出格式**：使用<tool_call>标签返回结构化结果，包含每个步骤的匹配信息：
			   {
				 "coordination_result": [
				   {
					 "step_id": 步骤ID,
					 "step_content": "步骤描述",
					 "matched_agent": ["候选Agent1"],
					 "agent_instructions": "对工具Agent的具体指令",
					 "confidence": 匹配置信度(0-1)  // 如0.9表示高适配
				   },
				   ...
				 ]
			   }
			   
			规则约束：
				- 仅负责“步骤→工具Agent”的匹配与指令生成，不参与需求解析或步骤拆解。
				- 若步骤无适配工具Agent，在"matched_agent"中返回[""]，并在"agent_instructions"说明原因（如“当前步骤无需工具”）。
				- 优先选择已验证的工具Agent，避免重复调用功能重叠的Agent。
				- 若历史对话中存在相关数据，无需调用工具
				
			'''
		else:
			tips = '''
			请按照用户要求选择出合适的工具来解决用户的问题
			'''
		return tips

	def get_input(self, chat_session, *args):
		system_ = {"role": "system", "content": self.get_system_tip(chat_session)}
		# 提取最新的用户对话 包含历史对话
		this_user_input = [*chat_session.get_current_input(), {"role": "user", "content": f"请基于以下步骤列表完成协调匹配:\n{chat_session.get_plan()}"}]
		return [system_, *this_user_input]

	def run(self, chat_session: ChatSession, *args) -> Any:
		response = self.generate(self.get_input(chat_session, *args), tools=self.tools, prompt=self.get_prompt(chat_session, *args))
		# 提取选择的agent，如果有
		agent_calls = self.get_angets(super().deal_response(response))
		return agent_calls["agent"]

	def get_angets(self, response):
		try:
			# 尝试直接处理
			text = response.strip().replace("\n", "")
			json_data = json.loads(text)
			return {"response": response, "agent": json_data}
		except json.JSONDecodeError as e:
			return {"response": response, "agent": None}





