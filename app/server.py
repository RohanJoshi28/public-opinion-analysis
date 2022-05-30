from flask import Flask, request, render_template
from transformers import AutoTokenizer
from transformers import AutoModel
from transformers import pipeline
from pytz import timezone
import tweepy
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import datetime
from dotenv import load_dotenv
import os
import io
import random
import base64 
import matplotlib.dates as mdates

load_dotenv()

app = Flask(__name__)

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model_saved=AutoModel.from_pretrained("RohanJoshi28/twitter_sentiment_analysisv1")
model_name = "RohanJoshi28/twitter_sentiment_analysisv1"
bearer_token = os.getenv("BEARER_TOKEN")
# define eastern timezone
eastern = timezone('US/Eastern')

pipe = pipeline(model=model_name, tokenizer=tokenizer)

@app.route("/classify", methods=["GET", "POST"])
def classify_text():
    print(request.method)
    if request.method=="POST":
        text = request.form.get('text')
        sentiment_ratio = get_sentiment_ratio(text)

        dates = []
        for i in range(5, 0, -1):
            dates.append(datetime.datetime.now(eastern)-datetime.timedelta(hours=24*(i)))
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m/%d/%Y'))
        plt.plot(dates, sentiment_ratio)
        plt.xticks(dates)
        plt.gcf().autofmt_xdate()

        plt.xlabel("Dates")
        plt.ylabel("Sentiment ratio (positive:negative)")

        my_stringIObytes = io.BytesIO()
        plt.savefig(my_stringIObytes, format='png')
        my_stringIObytes.seek(0)
        base64_image = base64.b64encode(my_stringIObytes.read()).decode("ascii")
        image_string = "data:image/png;base64," + base64_image
        return render_template("index.html", b64_image=image_string)

 
def get_sentiment_ratio(query):
    client = tweepy.Client(bearer_token=bearer_token)

    positive_ratio = []
    for i in range(5, 0, -1):
        tweets = client.search_recent_tweets(query=query, max_results=20, start_time=datetime.datetime.now(eastern)-datetime.timedelta(hours=24*i), end_time=datetime.datetime.now(eastern)-datetime.timedelta(hours=24*(i-1)+1))
        sentiment = np.array([1,1,1])
        for tweet in tweets:
            for t in tweet:
                if "text" not in t:
                    continue
                label = pipe(t.text)
                label_int = int(label[0]["label"][-1])
                sentiment[label_int]+=1
        positive_ratio.append(sentiment[0]/sentiment[2])

    return positive_ratio

@app.route('/')
def home():
    return render_template("index.html")
