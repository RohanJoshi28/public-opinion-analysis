from flask import Flask, request, render_template
from transformers import AutoTokenizer
from transformers import AutoModel
from transformers import pipeline

app = Flask(__name__)

@app.route('/')
def home():
    return render_template("index.html")

if __name__ == "__main__":
    app.run()
