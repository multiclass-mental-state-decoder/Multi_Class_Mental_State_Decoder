from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import os
from preprocess_and_tokenize import preprocess_and_tokenize

# text preprocessing
from nltk import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import re
import nltk
nltk.download('punkt')


# plots and metrics
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

# feature extraction / vectorization
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# classifiers
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline

# save and load a file
import dill
import pickle
# model = pickle.load(open('tfidf_svm.sav', 'rb'))

class MyCustomUnpickler(pickle.Unpickler):
	def find_class(self, module, name):
		if module == "__main__":
			module = "preprocess_and_tokenize"
		return super().find_class(module, name)

# Functions

app = Flask("Text_Emotizer")
@app.route("/", methods = ["GET","POST"])
def home():
	if request.method == "POST":
		content = request.form.get("text")
		if content=="":
			return render_template("index.html")
		# preprocess_and_tokenize(content)
		with open('tfidf_svm.sav', 'rb') as f:
			unpickler = MyCustomUnpickler(f)
			model = unpickler.load()

		# model = pickle.load(open('tfidf_svm.sav', 'rb'))
		sentiment = "The sentiment is: "+model.predict([content])[0]
		return render_template("index.html",answer=sentiment)
	return render_template("index.html")

if __name__ == "__main__":
	#port = int(os.environ.get("PORT",5000)) # uncomment for local host
	port = process.env.PORT  # uncomment for heroku
	app.run(debug=True, host='0.0.0.0', port=port)
