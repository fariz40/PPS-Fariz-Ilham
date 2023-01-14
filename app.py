from flask import Flask, app, render_template, request, redirect, url_for, json, session
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer, CountVectorizer
from imblearn.over_sampling import SMOTE
from sklearn.svm import SVC
from sklearn import svm
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import LabelEncoder

from nltk import word_tokenize
from nltk.corpus import stopwords
from werkzeug.utils import secure_filename
import pandas as pd
import numpy as np
import os
import re
import nltk
import pickle
nltk.download('punkt')

ALLOWED_EXTENSIONS = {'csv','xlsx'}

application = Flask(__name__)
UPLOAD_FOLDER = 'upload'
application.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
application.secret_key = 'Flask'
'''------------------------------Navigasi---------------------------'''
@application.route('/')
def dashboard():
	return render_template('index.html', menu="dashboard")

@application.route('/upload')
def upload_data():
    return render_template('upload.html')



@application.route('/upload',  methods=["POST", "GET"])
def uploadFile():
    if request.method == 'POST':
        # upload file flask
        upload_file = request.files['file']

        # Extracting uploaded data file name
        data_filename = secure_filename(upload_file.filename)

        # flask upload file to database (defined uploaded folder in static path)
        upload_file.save(os.path.join(application.config['UPLOAD_FOLDER'], data_filename))

        # Storing uploaded file path in flask session
        session['uploaded_data_file_path'] = os.path.join(application.config['UPLOAD_FOLDER'], data_filename)

        file = pd.read_excel(session['uploaded_data_file_path'])  
        file.dropna()
        text = file["Text"]
        clean_data = []
        for i in text:
            clean_data.append(cleaning(str(i)))

        clean_text = pd.DataFrame()
        clean_text['text'] = clean_data
        clean_text.to_csv(os.path.join(application.config['UPLOAD_FOLDER'], "cleaning.csv"))

        tokenz = pd.read_csv(os.path.join(application.config['UPLOAD_FOLDER'], "cleaning.csv"))
        text = tokenz["text"]
        data_tokenz = []
        for i in text:
        	data_tokenz.append(tokenizing(str(i)))

        text_tokenz = pd.DataFrame()
        text_tokenz['text'] = data_tokenz
        text_tokenz.to_csv(os.path.join(application.config['UPLOAD_FOLDER'], "token.csv"))

        stopword = pd.read_csv(os.path.join(application.config['UPLOAD_FOLDER'], "token.csv"))
        text = stopword["text"]
        data_stopword = []
        for i in text:
            data_stopword.append(filltering(i))

        text_stopword = pd.DataFrame()
        text_stopword['text'] = data_stopword
        text_stopword.to_csv(os.path.join(application.config['UPLOAD_FOLDER'], "stopword.csv"))

        stemmed = pd.read_csv(os.path.join(application.config['UPLOAD_FOLDER'], "stopword.csv"))
        text = stemmed["text"].head(50)
        data_stemming = []
        for i in text:
        	data_stemming.append(stemming(i))

        txt_stemming = pd.DataFrame()
        txt_stemming['text'] = data_stemming
        txt_stemming.to_csv(os.path.join(application.config['UPLOAD_FOLDER'], "stemming.csv"))

        return redirect(url_for('show_dataset'))

@application.route('/cleaning', methods=["GET"])
def clean():
    data = pd.read_csv(os.path.join(application.config['UPLOAD_FOLDER'], "cleaning.csv"))
    # print(data.info())
    return render_template("cleaning.html", data=data['text'])

@application.route('/tokenizing', methods=["GET"])
def token():
	data = pd.read_csv(os.path.join(application.config['UPLOAD_FOLDER'], "token.csv"), header=0)
	return render_template("tokenizing.html", data=data['text'])

@application.route('/stopword', methods=["GET"])
def stopword():
	data = pd.read_csv(os.path.join(application.config['UPLOAD_FOLDER'], "token.csv"), header=0)
	return render_template("stopword.html", data=data['text'])

@application.route('/stemming', methods=["GET"])
def stemming():
	data = pd.read_csv(os.path.join(application.config['UPLOAD_FOLDER'], "stemming.csv"), header=0)
	return render_template("stemming.html", data=data['text'])

@application.route('/dataset')
def show_dataset():
    try:
        # Retrieving uploaded file path from session
        data_file_path = session.get('uploaded_data_file_path', None)
     
        # read csv file in python flask (reading uploaded csv file from uploaded server location)
        uploaded_df = pd.read_excel(data_file_path, header=0)
        uploaded_df.dropna()
     
        # pandas dataframe to html table flask
        uploaded_df_html = uploaded_df.values
        print(uploaded_df_html)

        return render_template('tes_dataset.html', data=uploaded_df_html)
    except Exception as e:
        return render_template('tes_dataset.html')

def cleaning(text):
    tab = text.replace('\t', ' ').replace('\n', ' ').replace('\\', ' ')
    score = tab.replace('_', '')
    user = re.sub('@[A-Za-z0-9]+', '', score)
    link = re.sub(
    '((https?):((//)|(\\\\))+([\w\d:#@%/;$()~_?\+-=\\\.&](#!)?)*)+', '', user)
    url = re.sub(r'http\S+', '',link)
    punc = re.sub(r'[^\w\s]','',url)
    rt = re.sub(r'RT[\s]+', '', punc)
    no = re.sub('[0-9]+', '', rt)
    slang = re.sub(r'\\n', " ", no)
    reg = re.sub("b'", " ", slang)
    hashtag = re.sub('/#[\w_]+[ \t]*/', '', reg)
    result = hashtag.lower()

    return result

def tokenizing(text):
    text = word_tokenize(text)
    return text

def filltering(text):
	factory = StopWordRemoverFactory()
	stopword = factory.create_stop_word_remover()
	text = tokenizing(stopword.remove(text))
	return text

def stemming(text):
	factory = StemmerFactory()
	stemmer = factory.create_stemmer()
	text = [stemmer.stem(text)]
	return text

@application.route('/training',  methods=["POST", "GET"])
def uploadExcel():
    try:
        if request.method == 'POST':
            # upload file flask
            upload_file = request.files['file']

            # Extracting uploaded data file name
            data_filename = secure_filename(upload_file.filename)

            # flask upload file to database (defined uploaded folder in static path)
            upload_file.save(os.path.join(application.config['UPLOAD_FOLDER'], data_filename))

            # Storing uploaded file path in flask session
            session['uploaded_data_file_path'] = os.path.join(application.config['UPLOAD_FOLDER'], data_filename)

            return redirect(url_for('index'))
        
        # read xlsx file in python flask (reading uploaded xlsx file from uploaded server location)
        uploaded_df = pd.read_excel(os.path.join(application.config['UPLOAD_FOLDER'], "training.xlsx"))
        uploaded_df = pd.DataFrame(uploaded_df)
        print(uploaded_df)
        # pandas dataframe to html table flask
        uploaded_df_html = uploaded_df.values
        p = uploaded_df[uploaded_df['label'] == 'positif']['kelas'].count()
        o = uploaded_df[uploaded_df['label'] == 'netral']['kelas'].count()
        n = uploaded_df[uploaded_df['label'] == 'negatif']['kelas'].count()

        return render_template('upload_excel.html', data=uploaded_df_html, p=p, n=n, o=o)
    except Exception as e:
        return render_template('upload_excel.html')

#membuat file pickle
@application.route('/file_svm', methods=["GET"])
def pickle_function():
    df = pd.read_excel(os.path.join(application.config['UPLOAD_FOLDER'], "data_testing_svm.xlsx"))
    df = pd.DataFrame(df)
    df = df.fillna(' ')

    model = open('klasifikasi_SVM.pickle', 'rb')
    svm_classifier = pickle.load(model)

    data_tweet = df.cleaned_text
    pd.DataFrame(data_tweet)

    x = df.cleaned_text
    y = df.kelas

    vectorizer = TfidfVectorizer(min_df = 5,max_df = 0.8, sublinear_tf = True, use_idf = True)
    train_vectors = vectorizer.fit_transform(x)
    test_vectors = vectorizer.transform(df)

    svc = SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
    classifier_linear = svm.SVC(kernel='linear')
    classifier_linear.fit(train_vectors, y)

    test_vectors = vectorizer.transform(data_tweet)
    prediction_linear = classifier_linear.predict(test_vectors)

     

    result_tweet=[]
    for i in range(len(prediction_linear)):
        if(prediction_linear[i]=="P"):
            sentiment_result='positif'
        elif(prediction_linear[i]=="O"):
            sentiment_result='netral'
        else:
            sentiment_result='negatif'
        #     result_tweet.append({'class':prediction_linear[i], 'result_nbc':sentiment_result})
        result_tweet.append({'cleaned_text':data_tweet[i], 'class':prediction_linear[i] })

    df_tweet = result_tweet
    print(df_tweet)
    print("tes")
    return render_template("SVM.html", data=df_tweet)

#menguji data testing
@application.route('/testing', methods=["GET"])
def test():
    data = pd.read_excel(os.path.join(application.config['UPLOAD_FOLDER'], "data_testing.xlsx"))
    data = pd.DataFrame(data)
    data = data.dropna()

    data.loc[(data['actual'] == 1) & (data['predicted'] == 1), 'keterangan'] = 'TP'
    data.loc[(data['actual'] == 1) & (data['predicted'] == 0), 'keterangan'] = 'FP'
    data.loc[(data['actual'] == 0) & (data['predicted'] == 0), 'keterangan'] = 'TN'
    data.loc[(data['actual'] == 0) & (data['predicted'] == 1), 'keterangan'] = 'FN'

    df = data.values

    TP=data['keterangan'].value_counts()['TP']
    TN=data['keterangan'].value_counts()['TN']
    FP=data['keterangan'].value_counts()['FP']
    FN=data['keterangan'].value_counts()['FN']

    acc = accuracy = (TP+TN)/(TP+TN+FP+FN)
    prec = precission = (TP) / (TP+FP)
    re = recall = (TP) / (TP + FN)
    f_s = F1_Score = 2 * (recall*precission) / (recall + precission)

    return render_template("testing.html", data=df ,acc=acc, prec=prec, re=re, f_s=f_s)

if __name__ == '__main__':
    application.run(debug=True)