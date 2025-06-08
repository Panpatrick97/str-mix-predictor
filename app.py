"""
Flask web service for STR-mix analysis
-------------------------------------
Endpoints
---------
POST /predict_num   -> 返回人数
POST /predict_ratio -> 返回比例
POST /predict       -> 同时返回人数 + 比例
"""

from flask import Flask, request, jsonify, render_template
from predictor import predict as predict_num
from ratio_predictor import predict_ratio

app = Flask(__name__)

# ------------------------------ UI ------------------------------ #
@app.route("/")
def home():
    """Render the simple HTML front‑end."""
    return render_template("index.html")

# ----------------------------- APIs ----------------------------- #
@app.route("/predict_num", methods=["POST"])
def api_predict_num():
    try:
        data = request.get_json(force=True)
        alleles = data.get("alleles", [])
        result = predict_num(alleles)
        return jsonify({"status": "success", "prediction": result})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/predict_ratio", methods=["POST"])
def api_predict_ratio():
    try:
        data = request.get_json(force=True)
        alleles = data.get("alleles", [])
        marker  = data.get("marker", "UNKNOWN")
        result = predict_ratio(alleles, marker=marker)
        return jsonify({"status": "success", "prediction": result})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/predict", methods=["POST"])
def api_predict_both():
    try:
        data = request.get_json(force=True)
        alleles = data.get("alleles", [])
        marker  = data.get("marker", "UNKNOWN")

        num_info   = predict_num(alleles)
        ratio_info = predict_ratio(alleles, marker=marker)

        response = {
            "status": "success",
            "num": num_info["num"],
            **ratio_info
        }
        return jsonify(response)
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


# --------------------------- Run local -------------------------- #
if __name__ == "__main__":
    import os, sys

    # 默认端口 5000，可通过环境变量 PORT 或第一个命令行参数覆盖
    default_port = 5002  # 默认端口改为 5002
    if len(sys.argv) > 1 and sys.argv[1].isdigit():
        default_port = int(sys.argv[1])

    port = int(os.getenv("PORT", default_port))
    app.run(host="0.0.0.0", port=port, debug=True)
