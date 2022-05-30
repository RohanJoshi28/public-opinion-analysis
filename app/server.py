from flask import Flask, request, render_template
from transformers import AutoTokenizer
from transformers import AutoModel
from transformers import pipeline

app = Flask(__name__)

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model_saved=AutoModel.from_pretrained("RohanJoshi28/twitter_sentiment_analysisv1")
model_name = "RohanJoshi28/twitter_sentiment_analysisv1"

pipe = pipeline(model=model_name, tokenizer=tokenizer)

@app.route("/classify")
def classify_text():
    if request.method=="POST":
        text = request.form.get('text')
        return str(pipe(text)[0]["label"][-1])

@app.route('/')
def home():
    return render_template("index.html")

if __name__ == "__main__":
    app.run()
