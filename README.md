# Public Opinion Analysis

## Purpose 
Use a sentiment analysis model (possibly using huggingface) to create graphs on social media sentiment for different topics like finance, politics, and entertainment. Use APIs like the twitter API (MVP) and later expand to other APIs like the reddit API. Use a database to store past sentiment on topics many days ago.  Allow user search and queries so that users can get sentiment analysis on topics that interest them. 

## Technologies
We are using Flask and SQLAlchemy for the backend, and HTML/CSS/JS for the frontend. Matplotlib/seaborn is used for graphing. We will use huggingface, PyTorch, and kaggle datasets for ML. 

## Setup
First clone the repo like this: 
`git clone https://github.com/RohanJoshi28/public-opinion-analysis`

After that, create a pipenv virtual environment for your requirements:
`pipenv shell`

Then, download the requirements like this:
`pip install -r requirements.txt`

After that, you can run the server in terminal like this:
`python server.py` 

Or alternatively
`$env:FLASK_APP = "server"`

and then 
`flask run`

Create a .env file and make sure to put your own twitter API bearer token in the .env file like this:
`BEARER_TOKEN=blahblahblah`

## Contributors
The contributors to this project are Rohan Joshi, Justin Hwang, and Joshua Kim. 
