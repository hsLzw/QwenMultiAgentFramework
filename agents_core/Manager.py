from agents_core.agent_import import *
from agents_core.manager_agents.CoordinatorAgent import CoordinatorAgent
from agents_core.agents.CurrentAgent import CurrentAgent
from agents_core.manager_agents.EvaluationAgent import EvaluationAgent
from agents_core.manager_agents.HistorySummarizeAgent import HistorySummarizeAgent
from agents_core.manager_agents.OutputAgent import OutputAgent
from agents_core.manager_agents.PlanAgent import PlanAgent
from agents_core.agents.UserAccAgent import UserAccountAgent

from agents_core.base_agent import BaseAgent
from models.QwenModel import QwenFTModel


class MultiAgentSystem:
	"""多Agent协作系统，负责协调不同Agent之间的工作流程"""

	def __init__(self, llm):
		self.llm = llm

		# 初始化各类Agent
		self.current_agent = CurrentAgent(self.llm)
		self.user_account_agent = UserAccountAgent(self.llm)
		# 评估agent
		self.evaluation_agent = EvaluationAgent(self.llm)

		# 注册所有Agent 不包括评估
		self.agents = {
			self.current_agent.name: self.current_agent,
			self.user_account_agent.name: self.user_account_agent,
		}

		# 计划agent
		self.plan_agent = self._create_control_agent(PlanAgent)
		# 初始化协调器Agent
		self.coordinator = self._create_control_agent(CoordinatorAgent)
		# 答复Agent
		self.output_agent = OutputAgent(self.llm)
		# 历史记录归纳总结
		self.summarize_agent = HistorySummarizeAgent(self.llm)


		# 最大评估次数
		self.max_iterations = 3

	def _create_control_agent(self, cls) -> BaseAgent:
		"""创建协调器Agent，负责决定使用哪个专业Agent"""
		coordinator_tools = [
			Tool(
				name=self.current_agent.name,
				func=lambda x: self.current_agent.run(x),
				description=self.current_agent.description
			),
			Tool(
				name=self.user_account_agent.name,
				func=lambda x: self.user_account_agent.run(x),
				description=self.user_account_agent.description
			)
		]
		agent = cls(self.llm, coordinator_tools)
		return agent

	def run_plan(self, chat_session) -> Any:
		# 协调器 协调器的输出不需要加入历史
		agent_call = self.coordinator.run(chat_session)
		# 判断协调器是否存在
		if agent_call:
			# 选择真实的函数
			for step in agent_call["coordination_result"]:
				step_content = step["step_content"]
				matched_agent = step["matched_agent"]
				if matched_agent:
					target_agent = self.agents.get(matched_agent[0], None)
					# 添加中途提示词
					chat_session.add_history({"role": "assistant", "content": f"当前我要从tool_calls中寻找合适的工具，解决【{step_content}】。"})
					# Agent检索tools
					func_calls = target_agent.run(chat_session)
					# 提取tools
					tool_call = func_calls["tool_call"]
					# 如果存在tool_call
					if tool_call:
						# 是否存在此函数 防止模型乱说话
						func = target_agent.tools_map.get(tool_call["name"], None)
						if func:
							if "arguments" in tool_call:
								arguments = tool_call["arguments"].values()
							else:
								arguments = []
							# 调用工具
							result = func.func(*arguments)
							# 添加调用历史
							chat_session.add_history_tool_call("assistant", func_calls["match"])
							# 添加结果历史
							chat_session.add_history_tool_response("assistant", tool_call["name"], result)
					else:
						chat_session.add_history(
						{"role": "assistant", "content": f"解决{step_content}时出现了问题，可能是json解析出错，也可能是需要用户提供信息，具体原因请参考:【{func_calls['response']}】"})
				else:
					chat_session.add_history(
						{"role": "assistant", "content": f"解决{step_content}时出现了问题"})
		else:
			chat_session.add_history({"role": "assistant", "content": "没有相关工具可以被调用，自行考虑如何回复用户..."})

	def process_user_request(self, chat_session) -> str:
		"""处理用户请求的主流程，支持迭代处理"""
		# 计划制定
		plan = self.plan_agent.run(chat_session)
		# 加入计划
		if plan:
			chat_session.create_plan(plan)

		for i in range(self.max_iterations):
			# 执行计划
			self.run_plan(chat_session)
			# 检查是否完成，置信度等
			evaluation_result = self.evaluation_agent.run(chat_session)
			if evaluation_result["is_complete"]:
				break
			else:
				if "is_wrong" in evaluation_result:
					chat_session.add_history({"role": "assistant",
											  "content": f"结果无法进行评估。"})
					break
				chat_session.add_history({"role": "assistant", "content": f"我评估了以下历史对话，并不能解决用户的问题，下一步的建议:{evaluation_result['next_step']}"})

		# 输出内容
		response = self.output_agent.run(chat_session)
		# 总结
		self.summarize_agent.run(chat_session)
		return response

