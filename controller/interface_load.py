import logging
from flask import Flask, request, jsonify
from openai import OpenAI

app = Flask(__name__)

# 初始化 OpenAI 客户端
client = OpenAI(api_key="sk-fb30e789666a450a87e027f34100597f", base_url="https://api.deepseek.com")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

@app.route("/generate", methods=["POST"])
def generate():
    """
    接收预测请求并调用 OpenAI API 生成回复
    """
    try:
        # 获取请求数据
        data = request.json
        message = data.get("message")

        # 验证输入消息
        if not message:
            return jsonify({"error": "必须提供输入消息"}), 400

        # 调用 OpenAI API 生成回复
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "You are a expert to assist me to control the k8s cluster"},
                {"role": "user", "content": message},
            ],
            stream=False
        )

        # 提取并返回生成的回复
        result = response.choices[0].message.content
        logger.info(result)
        return jsonify({"result": result}), 200

    except Exception as e:
        # 处理异常情况
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
