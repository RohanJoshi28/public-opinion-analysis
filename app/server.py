from flask import Flask, request, render_template, flash
from flask_sqlalchemy import SQLAlchemy
import requests

import tweepy
import praw

from transformers import AutoTokenizer
from transformers import AutoModel
from transformers import pipeline

from pytz import timezone
import datetime

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
sns.set_style("darkgrid")

from dotenv import load_dotenv
import os
import io
import base64
from multiprocessing import Process 
import time

load_dotenv()

reddit = praw.Reddit(
    client_id=os.getenv("CLIENT_ID"),
    client_secret=os.getenv("CLIENT_SECRET"),
    password=os.getenv("PASSWORD"),
    user_agent=os.getenv("USER_AGENT"),
    username=os.getenv("USERNAME"),
)

app = Flask(__name__)
app.secret_key = "6310e994bb51a61d81c676ee"
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///site.db"
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

bearer_token = os.getenv("BEARER_TOKEN")
client = tweepy.Client(bearer_token=bearer_token)

api_key = os.getenv("API_KEY")
api_key_secret = os.getenv("API_KEY_SECRET")
access_token = os.getenv("ACCESS_TOKEN")
access_token_secret = os.getenv("ACCESS_TOKEN_SECRET")

auth = tweepy.OAuthHandler(api_key, api_key_secret)
auth.set_access_token(access_token, access_token_secret)

api = tweepy.API(auth)

politicians_list = ["Democratic Party", "Republican Party", "Joe Biden", "Donald Trump", "Nancy Pelosi", "Kamala Harris", "Bernie Sanders", "AOC", "Marco Rubio"]

finance_list = ["GME", "AMC", "GOOGL", "TSLA", "FB", "AMZN", "AAPL", "MSFT"]

entertainment_list = ["Movies", "ESPN", "CNN", "Fox News", "MSNBC", "Disney Plus", "Netflix", "Marvel"]

class Sentiment(db.Model):
    date = db.Column(db.Date, primary_key=True)
    name = db.Column(db.String(32), primary_key=True)
    sentiment_ratio = db.Column(db.Integer)

    def __str__(self):
        return f"{self.name}; {self.date.isoformat()}; {self.sentiment_ratio}"

db.create_all()

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model_saved=AutoModel.from_pretrained("RohanJoshi28/twitter_sentiment_analysisv1")
model_name = "RohanJoshi28/twitter_sentiment_analysisv1"
# define eastern timezone
eastern = timezone('US/Eastern')

pipe = pipeline(model=model_name, tokenizer=tokenizer)

def get_previous_days(num_days):
    dates = []
    for i in range(num_days, 0, -1):
        dates.append(datetime.datetime.now(eastern)-datetime.timedelta(hours=24*(i)))
    return dates

def get_start_and_end_days(today):
    start = today.replace(hour=0, minute=0, second=0, microsecond=0)
    end = start + datetime.timedelta(1)

    return start, end

def save_sentiment_data():
    figures = politicians_list+finance_list+entertainment_list
    past_datetimes = get_previous_days(5)
    for p_datetime in past_datetimes:
        if p_datetime.date() == datetime.datetime.now(eastern).date():
            continue
        for figure in figures:
            entries = Sentiment.query.filter_by(name=figure, date=p_datetime.date())
            if entries.count()==0:
                positive_ratio = get_ratio_given_times(*get_start_and_end_days(p_datetime), figure)
                new_sentiment = Sentiment(date=p_datetime.date(), name=figure, sentiment_ratio=positive_ratio)
                db.session.add(new_sentiment)
                db.session.commit()

def save_sentiment_data_process():
    print("PROCESS HAS STARTED")
    while True:
        save_sentiment_data()
        time.sleep(3600)


@app.route("/classify", methods=["GET", "POST"])
def classify_text():
    if request.method=="POST":
        
        
        text = request.form.get('text')
        num_tweets = int(request.form["myRange"])

        if not text == "":
            
            twitter_sentiment_ratio = get_sentiment_ratio(text, num_tweets, site="twitter")
            reddit_sentiment_ratio = get_sentiment_ratio(text, num_tweets, site="reddit")

            plt.clf()

            dates = get_previous_days(5)
            plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m/%d/%Y'))
            plt.plot(dates, twitter_sentiment_ratio)
            plt.plot(dates, reddit_sentiment_ratio)
            plt.legend(labels=["twitter", "reddit"])
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
        
    # flash('Please enter in a search!')
    return render_template("index.html")

def get_ratio_given_times(start_time, end_time, query, max_results=10, site="twitter"):
    if site=="reddit":
        timestamp1 = int(time.mktime(start_time.timetuple()))
        timestamp2 = int(time.mktime(end_time.timetuple()))
        submissions = requests.get(f"https://api.pushshift.io/reddit/submission/search/?after={timestamp1}&before={timestamp2}&sort_type=score&sort=desc&q={query}").json()
        sentiment = np.array([1,1,1])
        if len(submissions["data"])<max_results:
            max_results = len(submissions["data"])
        for submission in submissions["data"][0:max_results]:
            label = pipe(submission["title"])
            label_int = int(label[0]["label"][-1])
            sentiment[label_int]+=1
            submission_id = submission["id"]
            try:
                num_comments = 0
                reddit_submission = reddit.submission(submission_id)
                for comment in reddit_submission:
                    label = pipe(comment.body)
                    label_int = int(label[0]["label"][-1])
                    sentiment[label_int]+=1
                    num_comments+=1
                    if num_comments>=5:
                        break
            except:
                continue
        return sentiment[2]/sentiment[0]


    else:
        start_id = client.search_recent_tweets(query="hello", start_time=start_time, end_time=start_time + datetime.timedelta(seconds=100))
        start_id = start_id.meta['oldest_id']
        tweets = api.search_tweets(q=query, until=end_time.strftime('%Y-%m-%d'), since_id=start_id, count=max_results, result_type="popular")
        sentiment = np.array([1,1,1])
        #the 0th index of the tweets is where all the tweets are at, the other parts are just metadata
        for tweet in tweets:
            label = pipe(tweet.text)
            label_int = int(label[0]["label"][-1])
            sentiment[label_int]+=1
        return sentiment[2]/sentiment[0]

def get_sentiment_ratio(query, num_tweets=10, site="twitter"):
    positive_ratios = []
    for i in range(5, 0, -1):
        start_time=datetime.datetime.now(eastern)-datetime.timedelta(hours=24*i)
        end_time=datetime.datetime.now(eastern)-datetime.timedelta(hours=24*(i-1)+1)
        positive_ratio = get_ratio_given_times(start_time, end_time, query, num_tweets, site)
        positive_ratios.append(positive_ratio)

    return positive_ratios

@app.route('/')
def home():
    return render_template("home.html")

def get_image_string(date_list, ratio_list):
    plt.clf()

    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m/%d/%Y'))
    sns.lineplot(x=date_list, y=ratio_list)
    plt.xticks([date_list[0], date_list[-1]])
    plt.gcf().autofmt_xdate()

    plt.xlabel("Dates")
    plt.ylabel("Sentiment ratio (positive:negative)")

    my_stringIObytes = io.BytesIO()
    plt.savefig(my_stringIObytes, format='png')
    my_stringIObytes.seek(0)
    base64_image = base64.b64encode(my_stringIObytes.read()).decode("ascii")
    image_string = "data:image/png;base64," + base64_image
    return image_string


@app.route('/search')
def search():
    return render_template("index.html")

@app.route('/about')
def about():
    return render_template("about.html")

@app.route('/finance')
def finance():
    base_64_dict = {}
    for stock in finance_list:
        stock_sentiments = Sentiment.query.filter_by(name=stock)
        date_list = []
        ratio_list = []
        for sentiment in stock_sentiments:
            date_list.append(sentiment.date)
            ratio_list.append(sentiment.sentiment_ratio)
        
        image_string = get_image_string(date_list, ratio_list)
        base_64_dict[stock] = image_string

    
    return render_template("finance.html", base_64_dict=base_64_dict, figure_list=finance_list)

@app.route('/politics')
def politics():
    base_64_dict = {}
    for politician in politicians_list:
        stock_sentiments = Sentiment.query.filter_by(name=politician)
        date_list = []
        ratio_list = []
        for sentiment in stock_sentiments:
            date_list.append(sentiment.date)
            ratio_list.append(sentiment.sentiment_ratio)
        
        image_string = get_image_string(date_list, ratio_list)
        base_64_dict[politician] = image_string

    
    return render_template("politicians.html", base_64_dict=base_64_dict, figure_list=politicians_list)

@app.route('/entertainment')
def entertainment():
    base_64_dict = {}
    for ent in entertainment_list:
        stock_sentiments = Sentiment.query.filter_by(name=ent)
        date_list = []
        ratio_list = []
        for sentiment in stock_sentiments:
            date_list.append(sentiment.date)
            ratio_list.append(sentiment.sentiment_ratio)
        
        image_string = get_image_string(date_list, ratio_list)
        base_64_dict[ent] = image_string

    
    return render_template("entertainment.html", base_64_dict=base_64_dict, figure_list=entertainment_list)
if __name__=="__main__":
    global p
    p = Process(target=save_sentiment_data_process, args=())
    p.start()
    app.run(port=5000, debug=True, use_reloader=False)
    
