from flask import Flask, request, jsonify, render_template
from rust_model import predict

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict_route():
    try:
        file = request.files.get("image")
        if not file:
            return jsonify({"error": "Không có ảnh tải lên"})
        result = predict(file)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)
