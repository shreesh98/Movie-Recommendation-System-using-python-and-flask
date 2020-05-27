from flask import Flask, request, jsonify, render_template
import requests
from bs4 import BeautifulSoup, Comment
import os
import json
import xlrd
from moviee import hybrid
from Cross_platform_movies import get_recommendations
from Cross_platform_tv_shows import get_recommendations_tv

app = Flask(__name__)

@app.route('/', methods = ['GET'])
def get_movies():
    return render_template("index.html")

@app.route('/index.html', methods = ['GET'])
def get_moviesa():
    return render_template("index.html")

@app.route('/about.html', methods = ['GET'])
def get_moviesb():
    return render_template("about.html")

@app.route('/review.html', methods = ['GET'])
def get_moviesc():
    return render_template("review.html")

@app.route('/joinus.html', methods = ['GET'])
def get_moviesd():
    return render_template("joinus.html")

@app.route('/contact.html', methods = ['GET'])
def get_moviese():
    return render_template("contact.html")

@app.route('/', methods = ['POST'])
def devtest():
    platform = request.form.get('platform')
    choice = request.form.get('choice')
    moviename = request.form.get('moviename')
    if platform == "Cross":
        if choice == "Movies":
            data= get_recommendations(moviename).to_dict('records')
            return render_template("review.html", movies=data, p=platform, c=choice)
        else:
            data= get_recommendations_tv(moviename).to_dict('records')
            return render_template("review.html", movies=data, p=platform, c=choice)
    else:
        df= json.loads(hybrid(1,moviename))
        return render_template("review.html", movies=df, p=platform, c=choice)




if __name__ == '__main__':
    app.config["JSON_SORT_KEYS"] = False
    app.run()
