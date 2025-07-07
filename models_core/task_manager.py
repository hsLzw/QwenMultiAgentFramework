from models.QwenModel import QwenFTModel


class TaskManager:

	def __init__(self, model_path):
		self.model_path = model_path
		self.model = self.init_model()
		self.task_map = {}

	def init_model(self):
		return QwenFTModel(self.model_path)













