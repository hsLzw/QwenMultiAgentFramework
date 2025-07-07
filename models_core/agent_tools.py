import json

from langchain.tools import BaseTool
from langchain_community.utilities import WikipediaAPIWrapper
from typing import Type
import datetime


# 维基百科工具
class WikipediaTool(BaseTool):
	name: str = "wikipedia"  # 添加类型注解
	description: str = "当需要查询百科知识时使用；参数query用来进行查询的关键词(字符串)"  # 添加类型注解

	def _run(self, query: str) -> str:
		try:
			wikipedia = WikipediaAPIWrapper()
			return json.dumps({"role": "function", "content": wikipedia.run(query)})
		except Exception as e:
			return json.dumps({"role": "function", "content": f"查询维基百科时出错: {str(e)}"})

	async def _arun(self, query: str) -> str:
		return self._run(query)


# 时间获取工具
class GetRealTimeTool(BaseTool):
	name: str = "获取当前时间"  # 添加类型注解
	description: str = "当你不知道此时的时间时或用户说你给的时间不对时，可以使用当前工具获取;本工具参数command值必须为字符串a"  # 添加类型注解

	def _run(self, command: str) -> str:
		return json.dumps({"role": "function", "content": str(datetime.datetime.now())})

	async def _arun(self, command: str) -> str:
		return self._run(command)







'''
<tool_call>
{"name": "tool_get_date", "arguments": {}}
</tool_call>
'''



def tool_ger_order_list(date: str) -> list:
	'''
	获取订单列表

	Args:
		date: 要提取的日期，如["2025-01-01", "2025-05-01"]

	Returns:
		指定日期所有的订单列表
	'''
	return ["1111", "2222", "3333"]


def tool_get_date() -> str:
	'''
	获取当前时间

	Args:

	Returns:
		返回当前的时间
	'''
	return str(datetime.datetime.now())


