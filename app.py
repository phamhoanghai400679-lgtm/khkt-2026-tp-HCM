from flask import Flask, request, jsonify, render_template
from rust_model import predict
import os

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict_route():
    try:
        file = request.files.get("image")
        if not file:
            return jsonify({"error": "Không có file ảnh"}), 400

        result = predict(file.stream)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
