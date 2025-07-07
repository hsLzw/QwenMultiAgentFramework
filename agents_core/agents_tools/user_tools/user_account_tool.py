from agents_core.agent_import import *


class UserAccountTool:

	@staticmethod
	def get_tool_list():
		# 记录全部工具
		tool_list = []

		tool_list.append(StructuredTool.from_function(
			func=UserAccountTool.tool_register_user_account,
			name="tool_register_user_account",
			description="根据用户提供的信息，帮助用户注册账号"
		))
		return tool_list

	@staticmethod
	def tool_register_user_account(username: str, password: str, token: str) -> dict:
		'''
		根据用户提供的信息，帮助用户注册账号

		Args:
			username: 用户想要注册的账号
			password: 用户想要设定的密码
			token: 开发人员提供的注册令牌, 开发人员同意用户注册账号时会给予注册令牌

		Return
			{
				status: 注册成功或者失败
				register_account: 注册的账号
			}
		'''
		return {
			"status": "success",
			"register_account": username
		}


