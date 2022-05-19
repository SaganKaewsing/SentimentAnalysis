from cmath import nan
from pickle import GLOBAL
from django.shortcuts import render, redirect
from requests import request
from apiclient.discovery import build
import json
from csv import writer,reader
from apiclient.discovery import build
from urllib.request import urlopen
from urllib.parse import urlencode
import os
from django.http import HttpResponse
import pandas as pd
from django.contrib import messages

# Create your views here.
def hello(request):
    return render(request,'main_dashboard.html')

def how(request):
    return render(request,'how.html')

def main(request):
    return render(request,'mains.html')

def analysis(request):
    return render(request,'analysis_text.html')

def build_service():
#You should access to YoutubeApi to obtain the key
    key = "AIzaSyCT8F8vJgHHN0YO1z1taQEn3q5kCpSWmZM"
    YOUTUBE_API_SERVICE_NAME = "youtube"
    YOUTUBE_API_VERSION = "v3"
    return build(YOUTUBE_API_SERVICE_NAME,YOUTUBE_API_VERSION,developerKey=key)


def link(request):
    global videoURL
    videoURL=request.GET['url']
    if "https://youtu.be/" in videoURL and len(videoURL)==28 :

        get_comments()
        ytanalyze()

        col_list = ["Author", "Comment", "Result" ,"Date"]
        fn = pd.read_csv("youtubeoutput.csv", usecols=col_list)[['Author','Comment','Result','Date']]
        
        allcom = list(fn.Result)

        poscom = allcom.count('Positive')
        negcom = allcom.count('Negative')
        neucom = allcom.count('Neutral')

        posper = poscom*100/len(fn)
        negper = negcom*100/len(fn)
        neuper = neucom*100/len(fn)

        alldata=[]

        for a in range (fn.shape[0]):
            temp = fn.iloc[a]
            alldata.append(dict(temp))

        context = {'total_comment':len(fn),
                'loaded_data': alldata,
                'positive_comment': poscom,
                'negative_comment': negcom,
                'neutral_comment': neucom,
                'positive_percent': round(posper,2),
                'negative_percent': round(negper,2),
                'neutral_percent': round(neuper,2),}

        return render(request, "main_dashboard.html", context)
    else:
        messages.info(request,'Input must be Youtube URL.')
        return redirect('/')

    
def ytanalyze():

    df = pd.read_csv('TrainDataset.csv', sep=',', names=['text', 'sentiment','timestamp'], header=0)

    from pythainlp.corpus.common import thai_stopwords
    thai_stopwords = list(thai_stopwords())

    from pythainlp import word_tokenize


    def remove3ConsecutiveDuplicates(string):
        i = 1
        c = 1
        save = ''
        string = str(string)
        string_len = len(string)
        for x in string:
            if x == save:
                c += 1
            elif x != save:
                if c >= 2:
                    c = c + 1
                    string = string.replace(save * c,save)
                c = 0
            if c >= 2 and i == string_len:
                if(x == save):
                    c += 1
                string = string.replace(save * c,save)
                c = 0
            i += 1
            save = x
        return string

    def text_process(text):
        final = "".join(u for u in text if u not in ("?", ".", ";", ":", "!", '"', "ๆ", "ฯ"))
        final = word_tokenize(final)
        final = " ".join(word for word in final)
        final = " ".join(word for word in final.split() 
                     if word.lower not in thai_stopwords)
        return final
    df['text_tokens'] = df['text'].apply(text_process,remove3ConsecutiveDuplicates)

    from sklearn.model_selection import train_test_split
    X = df[['text_tokens']]
    y = df['sentiment']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.00000001, random_state=101)

    from sklearn.feature_extraction.text import CountVectorizer
    cvec = CountVectorizer(analyzer=lambda x:x.split(' '))
    cvec.fit_transform(X_train['text_tokens'])

    train_bow = cvec.transform(X_train['text_tokens'])
    pd.DataFrame(train_bow.toarray(), columns=cvec.get_feature_names_out(),
            index=X_train['text_tokens'])


    from sklearn.linear_model import LogisticRegression
    lr = LogisticRegression()
    lr.fit(train_bow, y_train)


    data = pd.read_csv(r'youtubecomments.csv' ,header=None)
    data.to_csv(r'youtubecomments.csv', header=["Source","Date","Author","Comment"], index=False)

    data = pd.read_csv(r'youtubecomments.csv')

    date = [data.Date]

    def splitdate(t):
        t = str(t)
        a = t.split("T")
        b = a[0].split("-")
        val = b[2]+"-"+b[1]+"-"+b[0]
        return (val)

    d=0
    new_date=[]
    for x in date:
        new_date_string = str(x)
        if pd.isna(new_date_string):
            break
        new_date_string = splitdate(new_date_string)
        new_date.append(new_date_string)
        d+1

    data.Date = new_date
    data.to_csv('youtubecomments.csv', index=False)


    list_of_csv = [list(row) for row in data.values]
    flat_list = [item for sublist in list_of_csv for item in sublist]
    comment = flat_list[3::4]

    comment = [str(r).replace('\r', ' ') for r in comment]
    data.Comment=comment
    comment = data['Comment'].apply(remove3ConsecutiveDuplicates)
    data.Comment=comment
    data.to_csv('youtubecomments.csv', index=False)

    i=0
    consec=[]
    for asd in comment:
        new_string = asd
        new_string = remove3ConsecutiveDuplicates(new_string)
        consec.append(new_string)
        i+1

    k=0
    fn_predictions = []

    for dsa in consec:
        new_strings = dsa
        yt_tokens = text_process(new_strings)
        yt_bow = cvec.transform(pd.Series([yt_tokens]))
        yt_predictions = lr.predict(yt_bow)
        if (yt_predictions == ['pos']) : fn_predictions.append("Positive")
        elif (yt_predictions == ['neg']) : fn_predictions.append("Negative")
        elif (yt_predictions == ['neu']) : fn_predictions.append("Neutral")
        k+1


    with open('youtubecomments.csv', 'r', encoding="utf8") as fi:
        lines = [[j.strip() for j in line.strip().split(',')] \
            for line in fi.readlines()]

    col = ['Result'] + fn_predictions

    new_lines = [line + [str(col[j])] for j, line in enumerate(lines)]

    with open('youtubeoutput.csv', 'w', encoding="utf8") as fo:
        for line in new_lines:
            fo.write(','.join(line) + '\n')
    
    

def get_comments(part='snippet', 
                    maxResults=100, 
                    textFormat='plainText',
                    order='time', #Youtube URL
                    csv_filename="data"):
        videoId = videoURL.replace("https://youtu.be/","")
        
        #3 create empty lists to store desired information
        comments = []
        # build our service from path/to/apikey
        service = build_service()
        
        #4 make an API call using our service
        response = service.commentThreads().list(part=part, maxResults=maxResults, textFormat=textFormat, order=order, videoId=videoId).execute()

        f = open("youtubecomments.csv", "w")
        f.truncate()
        f.close()          

        while response: # this loop will continue to run until you max out your quota
                    
            for item in response['items']:
                #4 index item for desired data features
                comment1 = item['snippet']['topLevelComment']['snippet']
                comment = comment1['textDisplay'].replace('\n', '')
                author = comment1['authorDisplayName']
                date = comment1['publishedAt']
                source = comment1['videoId']

                #4 append to lists
                comments.append(comment)
            
                #7 write line by line
                with open('youtubecomments.csv','a+',encoding='utf-8-sig') as f:
                    # write the data in csv file with colums(source, date, author, text of comment)
                    csv_writer = writer(f)
                    csv_writer.writerow([source,date,author,comment])


                #8 check for nextPageToken, and if it exists, set response equal to the JSON response
            if 'nextPageToken' in response:
                response = service.commentThreads().list(
                    part=part,
                    maxResults=maxResults,
                    textFormat=textFormat,
                    order=order,
                    videoId=videoId,
                    pageToken=response['nextPageToken']
                ).execute()
            else:
                break

        #9 return our data of interest
        return {
            'Comments': comments,
           
        }


def txt(request):
    
    df = pd.read_csv('TrainDataset.csv', sep=',', names=['text', 'sentiment', 'timestamp'], header=0)

    from pythainlp.corpus.common import thai_stopwords
    thai_stopwords = list(thai_stopwords())

    from pythainlp import word_tokenize


    def remove3ConsecutiveDuplicates(string):
        i = 1
        c = 1
        save = ''
        string = str(string)
        string_len = len(string)
        for x in string:
            if x == save:
                c += 1
            elif x != save:
                if c >= 2:
                    c = c + 1
                    string = string.replace(save * c,save)
                c = 0
            if c >= 2 and i == string_len:
                if(x == save):
                    c += 1
                string = string.replace(save * c,save)
                c = 0
            i += 1
            save = x
        return string


    def text_process(text):
        final = "".join(u for u in text if u not in ("?", ".", ";", ":", "!", '"', "ๆ", "ฯ"))
        final = word_tokenize(final)
        final = " ".join(word for word in final)
        final = " ".join(word for word in final.split() 
                     if word.lower not in thai_stopwords)
        return final
    df['text_tokens'] = df['text'].apply(text_process,remove3ConsecutiveDuplicates)

    from sklearn.model_selection import train_test_split
    X = df[['text_tokens']]
    y = df['sentiment']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.00000001, random_state=101)

    from sklearn.feature_extraction.text import CountVectorizer
    cvec = CountVectorizer(analyzer=lambda x:x.split(' '))
    cvec.fit_transform(X_train['text_tokens'])

    train_bow = cvec.transform(X_train['text_tokens'])
    pd.DataFrame(train_bow.toarray(), columns=cvec.get_feature_names_out(),
            index=X_train['text_tokens'])


    from sklearn.linear_model import LogisticRegression
    lr = LogisticRegression()
    lr.fit(train_bow, y_train)


    global analyzetxt
    analyzetxt=request.GET['txt']
 
    my_tokens = text_process(analyzetxt)
    my_bow = cvec.transform(pd.Series([my_tokens]))
    my_predictions = lr.predict(my_bow)
    if (my_predictions == ['pos']) : my_predictions = "Positive"
    elif (my_predictions == ['neg']) : my_predictions = "Negative"
    elif (my_predictions == ['neu']) : my_predictions = "Neutral"

    contxt = {'text_analyze':my_predictions}

    return render(request, "analysis_text.html", contxt)


def linkpos(request):
    
    col_list = ["Author", "Comment", "Result" ,"Date"]
    fn = pd.read_csv("youtubeoutput.csv", usecols=col_list)[['Author','Comment','Result','Date']]

    fn_pos = fn [(fn ["Result"] == "Positive")]

    posdata = []
    for a in range (fn_pos.shape[0]):
        temp = fn_pos.iloc[a]
        posdata.append(dict(temp))

    allcom = list(fn.Result)

    poscom = allcom.count('Positive')
    negcom = allcom.count('Negative')
    neucom = allcom.count('Neutral')

    posper = poscom*100/len(fn)
    negper = negcom*100/len(fn)
    neuper = neucom*100/len(fn)

    context = {'total_comment':len(fn),
                'loaded_data': posdata,
                'positive_comment': poscom,
                'negative_comment': negcom,
                'neutral_comment': neucom,
                'positive_percent': round(posper,2),
                'negative_percent': round(negper,2),
                'neutral_percent': round(neuper,2),}

    return render(request, "main_dashboard.html", context)


def linkneg(request):
    
    col_list = ["Author", "Comment", "Result" ,"Date"]
    fn = pd.read_csv("youtubeoutput.csv", usecols=col_list)[['Author','Comment','Result','Date']]

    fn_neg = fn [(fn ["Result"] == "Negative")]

    negdata = []
    for a in range (fn_neg.shape[0]):
        temp = fn_neg.iloc[a]
        negdata.append(dict(temp))

    allcom = list(fn.Result)

    poscom = allcom.count('Positive')
    negcom = allcom.count('Negative')
    neucom = allcom.count('Neutral')

    posper = poscom*100/len(fn)
    negper = negcom*100/len(fn)
    neuper = neucom*100/len(fn)

    context = {'total_comment':len(fn),
                'loaded_data': negdata,
                'positive_comment': poscom,
                'negative_comment': negcom,
                'neutral_comment': neucom,
                'positive_percent': round(posper,2),
                'negative_percent': round(negper,2),
                'neutral_percent': round(neuper,2),}


    return render(request, "main_dashboard.html", context)


def linkneu(request):
    
    col_list = ["Author", "Comment", "Result" ,"Date"]
    fn = pd.read_csv("youtubeoutput.csv", usecols=col_list)[['Author','Comment','Result','Date']]

    fn_neu = fn [(fn ["Result"] == "Neutral")]

    neudata = []
    for a in range (fn_neu.shape[0]):
        temp = fn_neu.iloc[a]
        neudata.append(dict(temp))

    allcom = list(fn.Result)

    poscom = allcom.count('Positive')
    negcom = allcom.count('Negative')
    neucom = allcom.count('Neutral')

    posper = poscom*100/len(fn)
    negper = negcom*100/len(fn)
    neuper = neucom*100/len(fn)

    context = {'total_comment':len(fn),
                'loaded_data': neudata,
                'positive_comment': poscom,
                'negative_comment': negcom,
                'neutral_comment': neucom,
                'positive_percent': round(posper,2),
                'negative_percent': round(negper,2),
                'neutral_percent': round(neuper,2),}

    return render(request, "main_dashboard.html", context)

def linktotal(request):
    
    col_list = ["Author", "Comment", "Result" ,"Date"]
    fn = pd.read_csv("youtubeoutput.csv", usecols=col_list)[['Author','Comment','Result','Date']]

    alldata=[]

    for a in range (fn.shape[0]):
        temp = fn.iloc[a]
        alldata.append(dict(temp))

    allcom = list(fn.Result)

    poscom = allcom.count('Positive')
    negcom = allcom.count('Negative')
    neucom = allcom.count('Neutral')

    posper = poscom*100/len(fn)
    negper = negcom*100/len(fn)
    neuper = neucom*100/len(fn)

    context = {'total_comment':len(fn),
                'loaded_data': alldata,
                'positive_comment': poscom,
                'negative_comment': negcom,
                'neutral_comment': neucom,
                'positive_percent': round(posper,2),
                'negative_percent': round(negper,2),
                'neutral_percent': round(neuper,2),}

    return render(request, "main_dashboard.html", context)


import mimetypes

def download_file(request):
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    filename = 'youtubeoutput.csv'
    filepath = './youtubeoutput.csv'
    path = open(filepath, 'r', encoding="utf8")
    mime_type, _ = mimetypes.guess_type(filepath)
    response = HttpResponse(path, content_type=mime_type)
    response['Content-Disposition'] = "attachment; filename=%s" % filename
    return response
