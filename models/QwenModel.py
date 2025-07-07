import json
from typing import Optional, List, Mapping, Any
from langchain.llms.base import LLM
import torch
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.outputs import LLMResult
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_community.llms import HuggingFacePipeline
from vllm import LLM as VLLM, SamplingParams
from pydantic import Field
from models_core.agent_tools import tool_ger_order_list, tool_get_date


class QwenFTModelVLLM(LLM):

	vllm_model: VLLM = Field(None, description="VLLM模型实例")
	tokenizer: Any = Field(None, description="分词器")
	model_name: str = Field(..., description="模型路径或名称")
	temperature: float = Field(0.8, description="采样温度")
	top_p: float = Field(0.95, description="top-p采样参数")
	top_k: int = Field(10, description="top-k采样参数")
	max_tokens: int = Field(2048, description="最大生成长度")
	model_device: str = Field("cuda:0", description="模型运行设备")

	def __init__(
			self,
			model_name: str,
			temperature: float = 0.7,
			top_p: float = 0.95,
			top_k: int = 10,
			max_tokens: int = 2048,
			model_device: str = "cuda:0", description="模型运行设备",
			**kwargs
	):
		# 过滤掉未定义的参数
		allowed_params = {
			"temperature": temperature,
			"top_p": top_p,
			"top_k": top_k,
			"max_tokens": max_tokens,
			"model_name": model_name,
			"model_device": model_device
		}
		# 合并允许的参数和其他合法参数
		valid_kwargs = {k: v for k, v in kwargs.items() if k in self.__fields__}
		allowed_params.update(valid_kwargs)
		# 调用Pydantic的初始化方法
		super().__init__(**allowed_params)

		# 提取并设置模型参数
		self.model_name = model_name
		self.temperature = temperature
		self.model_device = model_device
		self.top_p = top_p
		self.top_k = top_k
		self.max_tokens = max_tokens
		self.tokenizer = None
		self.init_model(model_name)

	def init_model(self, model_path):
		self.tokenizer = AutoTokenizer.from_pretrained(model_path)
		# 1. 加载模型和Tokenizer
		self.vllm_model = VLLM(
			model=model_path,  # 模型名（自动从 HF 下载）或本地路径（如 ./Qwen-1_8B-Chat）
			trust_remote_code=True,       # 若模型需自定义代码（如 Qwen），必须开启
			max_model_len=4096,            # 模型上下文长度，按需调整
		)



	@property
	def _llm_type(self) -> str:
		return "qwen-vllm"

	@property
	def _identifying_params(self) -> Mapping[str, Any]:
		return {
			"temperature": self.temperature,
			"top_p": self.top_p,
			"top_k": self.top_k,
			"max_tokens": self.max_tokens,
			"model_name": self.model_name,
			"model_device": self.model_device
		}

	def _call(
			self,
			prompt: str,
			stop: Optional[list[str]] = None,
			run_manager: Optional[CallbackManagerForLLMRun] = None,
			**kwargs: Any,
	) -> str:

		print("实际输入:", prompt)
		formatted_prompt = self.tokenizer.apply_chat_template([{"role": "user", "content": prompt}],
															  tokenize=False,
															  add_generation_prompt=True,
															  enable_thinking=True,
															  )
		print("formatted_prompt: ", formatted_prompt)
		# 2. 创建文本生成 Pipeline
		sampling_params = SamplingParams(
			temperature=self.temperature,  # 随机性，0 为最确定
			top_p=self.top_p,  # 核采样，控制候选词范围
			top_k=self.top_k,  # 核采样，控制候选词范围
			max_tokens=self.max_tokens,  # 生成文本最大长度
			repetition_penalty=1.3   # 重复惩罚
		)

		# 3. 执行推理
		outputs = self.vllm_model.generate(
			[formatted_prompt],
			sampling_params
		)

		# 4. 解析结果
		response = outputs[0].outputs[0].text

		return response

	def _generate(
			self,
			prompts: list[str],
			stop: Optional[list[str]] = None,
			run_manager: Optional[CallbackManagerForLLMRun] = None,
			**kwargs: Any,
	) -> LLMResult:
		return super()._generate(prompts, stop, run_manager, **kwargs)





class QwenFTModel():

	def __init__(
			self,
			model_name: str,
			temperature: float = 0.7,
			top_p: float = 0.95,
			top_k: int = 10,
			max_tokens: int = 2048,
			model_device: str = "cuda:0", description="模型运行设备",
	):
		# 提取并设置模型参数
		self.model_name = model_name
		self.temperature = temperature
		self.model_device = model_device
		self.top_p = top_p
		self.top_k = top_k
		self.max_tokens = max_tokens
		self.tokenizer = None
		self.vllm_model = None
		self.init_model(model_name)

	def init_model(self, model_path):
		self.tokenizer = AutoTokenizer.from_pretrained(model_path)
		# 1. 加载模型和Tokenizer
		self.vllm_model = VLLM(
			model=model_path,  # 模型名（自动从 HF 下载）或本地路径（如 ./Qwen-1_8B-Chat）
			trust_remote_code=True,       # 若模型需自定义代码（如 Qwen），必须开启
			max_model_len=4096,            # 模型上下文长度，按需调整
		)

	def tools_to_json(self, tools):
		"""将StructuredTool列表转换为JSON格式"""
		json_tools = []
		for tool in tools:
			# 提取工具的基本信息
			tool_info = {
				"name": tool.name,
				"description": tool.description,
				"parameters": {
					"type": "object",
					"properties": {},
					"required": []
				}
			}

			# 如果工具的func有参数注释，提取参数信息
			if hasattr(tool.func, "__annotations__"):
				for param_name, param_type in tool.func.__annotations__.items():
					if param_name == "return":
						continue  # 跳过返回值注释

					# 简化的参数类型映射（可根据需要扩展）
					if param_type == str:
						param_schema = {"type": "string"}
					elif param_type == int:
						param_schema = {"type": "integer"}
					elif param_type == float:
						param_schema = {"type": "number"}
					elif param_type == bool:
						param_schema = {"type": "boolean"}
					else:
						param_schema = {"type": "string"}  # 默认视为字符串

					tool_info["parameters"]["properties"][param_name] = param_schema

					# 如果参数没有默认值，视为必需参数
					if param_name not in tool.func.__defaults__ if (hasattr(tool.func, "__defaults__") and tool.func.__defaults__) else []:
						tool_info["parameters"]["required"].append(param_name)

			json_tools.append(tool_info)
		return json_tools

	def generate(
			self,
			user_input: List,
			tools: List,
			temperature: float = 0.7,
			prompt: str = ""
	) -> str:

		if not prompt:
			json_tools = []
			if tools:
				json_tools = self.tools_to_json(tools)
			formatted_prompt = self.tokenizer.apply_chat_template(user_input,
																  tokenize=False,
																  add_generation_prompt=True,
																  enable_thinking=True,
																  tools=json_tools
																  )
		else:
			formatted_prompt = prompt

		# 2. 创建文本生成 Pipeline
		sampling_params = SamplingParams(
			temperature=temperature or self.temperature,  # 随机性，0 为最确定
			top_p=self.top_p,  # 核采样，控制候选词范围
			top_k=self.top_k,  # 核采样，控制候选词范围
			max_tokens=self.max_tokens,  # 生成文本最大长度
			repetition_penalty=1.3   # 重复惩罚
		)
		print("实际模板：", formatted_prompt)
		# 3. 执行推理
		outputs = self.vllm_model.generate(
			[formatted_prompt],
			sampling_params
		)

		# 4. 解析结果
		response = outputs[0].outputs[0].text
		print("实际输出:", response)
		return response


