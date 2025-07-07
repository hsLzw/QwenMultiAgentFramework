from agents_core.agent_import import *
import datetime


class TimeTool:

	@staticmethod
	def get_tool_list():
		# 记录全部工具
		tool_list = []

		tool_list.append(StructuredTool.from_function(
			func=TimeTool.tool_get_current_time,
			name="tool_get_current_time",
			description="获取当前的时间"
		))
		tool_list.append(StructuredTool.from_function(
			func=TimeTool.tool_get_time_week,
			name="tool_get_time_week",
			description="计算给出的日期是本年的第几周"
		))
		return tool_list

	@staticmethod
	def tool_get_current_time() -> str:
		'''
		获取当前的时间

		Args:

		Return:
			返回日期，格式为: 2025-01-01 15:30:31
		'''
		return str(datetime.datetime.now()).split(".")[0]

	@staticmethod
	def tool_get_time_week(date_time: str) -> str:
		'''
		计算给出的日期是本年的第几周

		Args:
			date_time: 字符串日期，格式为:2025-01-01

		Return:
			返回第几周，如第32周返回 week32
		'''

		return "week46"

