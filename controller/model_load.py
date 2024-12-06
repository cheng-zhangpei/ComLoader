from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import logging

from service.ComLoader import ComLoader
#
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
app = Flask(__name__)
loader = ComLoader()
@app.route("/load_model_local", methods=["POST"])
def load_model_local():
    """
    接收模型加载请求
    """
    try:
        data = request.json
        model_path = data.get("model_path")
        if not model_path:
            return jsonify({"error": "必须提供模型路径"}), 400

        loader.load_model_local(model_path)
        return jsonify({"status": "模型加载成功"}), 200
    except Exception as e:
        logger.error(f"模型加载请求失败: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/generate", methods=["POST"])
def generate():
    """
    接收预测请求
    """
    try:
        data = request.json
        message = data.get("message")
        # 在生产的时候应该允许用户去配置模型的一些输出特性=> 可以用json的形式来进行表达

        max_token = data.get("max_token", 812)  # 默认最大生成长度 1024
        if not message:
            return jsonify({"error": "必须提供输入消息"}), 400

        result = loader.predict(message, max_token)
        return jsonify({"result": result}), 200
    except ValueError as e:
        logger.error(f"预测请求失败: {e}")
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        logger.error(f"未知错误: {e}")
        return jsonify({"error": "服务器内部错误"}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
