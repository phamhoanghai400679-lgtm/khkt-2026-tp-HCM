from flask import Flask  # type: ignore

app = Flask(__name__)

# Install Flask from a terminal with: pip install flask

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
