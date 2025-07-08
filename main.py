from agents_core.Manager import MultiAgentSystem
from models.QwenModel import QwenFTModel, QwenFTModelVLLM
from models_core.chat_session import ChatManage

# 演示主程序
if __name__ == "__main__":
	llm = QwenFTModel("models/Qwen/Qwen3-0___6B")
	# 初始化多Agent系统
	mas = MultiAgentSystem(llm)
	chat_manager = ChatManage(llm)

	# userid
	chat_sid = "asdjcbuadijb"

	# 获取session
	this_chat_session = chat_manager.get_user_session(chat_sid, auto_create=True)

	# 复杂问题示例
	user_input = "什么是量子？"

	# 初始化session
	this_chat_session.init_current_input(user_input)

	# 传入session
	response = mas.process_user_request(this_chat_session)
	print(f"\n最终回复: {response}")






