import copy
import json
import os.path


# 聊天会话统一管理
class ChatManage:

	def __init__(self, model):
		self.session_map = {}
		self.model = model

	def create_user_session(self, user_id):
		if user_id not in self.session_map:
			self.session_map[user_id] = ChatSession(user_id, self.model)
		return self.session_map[user_id]

	def get_user_session(self, user_id, auto_create=False):
		if user_id not in self.session_map:
			if auto_create:
				return self.create_user_session(user_id)
			return None
		else:
			return self.session_map.get(user_id, None)

	def clear_user_session(self, user_id):
		if user_id not in self.session_map:
			return True
		target_session = self.session_map.get(user_id, None)
		if target_session is None:
			return True
		# 删除全部图像/视频
		for i in target_session.image_path:
			if os.path.exists(i):
				os.remove(i)
		for i in target_session.video_path:
			if os.path.exists(i):
				os.remove(i)
		del target_session
		if user_id in self.session_map:
			self.session_map.pop(user_id)

# 聊天会话
class ChatSession:

	def __init__(self, user_id, model):
		self.model = model
		# 全部历史
		self.all_chat_history = []
		# 纯聊天历史
		self.chat_history = []
		# 本次流程期间全部记录
		self.current_line = []
		# 本次的计划
		self.plan = {}
		# 用户聊天id
		self.user_id = user_id
		# 最长存储历史
		self.max_history_length = 5
		# 多模态
		self.vl_history = []
		self.image_path = []
		self.video_path = []

	def init_current_input(self, input_msg):
		"""初始化本次流程全部数据"""
		self.chat_history = copy.deepcopy(self.all_chat_history)
		self.all_chat_history = []
		self.chat_history.append({"role": "user", "content": input_msg})
		self.plan = {}
		return

	def get_current_input(self, use_history=True):
		return self.chat_history

	def create_plan(self, plan_json):
		self.plan = plan_json
		plan_str = ""
		for i in plan_json["plan"]:
			plan_str += f'''- {i["step"]}\n'''
		self.add_history({"role": "assistant", "content": f"根据用户的需求与现有的功能，我制定了如下步骤:\n{plan_str}"})

	def get_plan(self):
		return self.plan

	def get_plan_str(self):
		plan_str = ""
		for i in self.plan["plan"]:
			plan_str += f'''Step{i["step_id"]}- {i["step"]}\n'''
		return plan_str

	def add_history(self, message):
		self.chat_history.append(message)

	def add_history_tool_call(self, role, match):
		dt = {
			"role": role,
			"content": f"<tool_call>{match}</tool_call>"
		}
		self.add_history(dt)

	def add_history_tool_response(self, role, func_name, result):
		content = json.dumps({
				"name": func_name,
				"result": result
			}, ensure_ascii=False)
		dt = {
			"role": role,
			"content": f"<tool_response>{content}</tool_response>"
		}
		self.add_history(dt)

	def update_all_history(self):
		self.all_chat_history = self.chat_history
		# 将all_chat_history优化

