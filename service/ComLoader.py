"""
@Function:
@Author : ZhangPeiCheng
@Time : 2024/12/6 15:37
"""
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class ComLoader:
    def __init__(self):
        """
        初始化 ComLoader 对象
        """
        self.model = None
        self.tokenizer = None

    def load_model_local(self, model_path):
        """
        加载模型到内存
        """
        global model, tokenizer
        try:
            # 加载量化配置
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype="float16",
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
            logger.info("正在加载模型，请稍候...")
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path, device_map="auto", quantization_config=bnb_config
            ).eval()
            logger.info("模型加载完成！")
        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            raise ValueError(f"模型加载失败: {e}")

    def predict(self, message, max_token):
        """
        根据消息进行模型推理
        """
        if not self.model or not self.tokenizer:
            logger.error("模型或分词器尚未加载！")
            raise ValueError("模型或分词器尚未加载！")
        try:
            inputs = self.tokenizer(message, return_tensors="pt")
            inputs = inputs.to(self.model.device)
            logger.info("start llm interference")
            outputs = self.model.generate(
                **inputs,
                temperature=0.3,
                top_p=0.7,
                do_sample=True,
                num_beams=2,
                early_stopping=True,
                max_new_tokens = max_token,
            )

            result = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            # 对输出的内容进行截断
            if message in result:
                result = result.replace(message, "").strip()
            logger.info("model output: " + result)
            logger.info("推理完成")
            return result
        except Exception as e:
            logger.error(f"推理失败: {e}")
            raise ValueError(f"推理失败: {e}")

    def release(self):
        pass