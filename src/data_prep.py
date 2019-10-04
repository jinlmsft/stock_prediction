"""
    This module is to prepare training data set for model development, 
    and inputs for stock prediction model. 

    The functions include loading earnings surprise data, 
    scrape earnings call transcript from seekingAlpha.com, 
    extract the tones from the text of earnings call transcript, 
    and compute ground truth stock price/volatility change.  

"""
import os
import pandas as pd
import numpy as np
import re
import Load_MasterDictionary as LM
MASTER_DICTIONARY_FILE = r'LoughranMcDonald_MasterDictionary_2014.csv'
lm_dictionary = LM.load_masterdictionary(MASTER_DICTIONARY_FILE, True)
import urllib.request
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
import string
import pandas_datareader as pdr


ticker_list = {"Apple":"AAPL","Amazon":"AMZN","Twitter":"TWTR","Microsoft":"MSFT","IBM":"IBM",
                "Facebook":"FB","Ebay":"EBAY","Google":"GOOG","Oracle":"ORCL","Intel":"INTC"}

# Load earnings surprise data
# Input: a string of a company's name
# Output: a data frame includes time stamp and a company's earning surprise. 
def load_eps(company_name):
    path = os.getcwd()
    path = path[:-3]
    path = path + 'data/'
    eps_data = pd.read_csv(path + company_name + '_eps.csv',sep = '\t',names = ['time','eps'])
    eps_data['time'] = pd.to_datetime(eps_data['time'])
    eps_data['eps'] = eps_data['eps'].str.replace('%','').apply(float)
    return eps_data


# Scrape the earnings call transcript from the web and return a html file 
user_agent = 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/71.0.3578.98 Safari/537.36'
def download_url(url):
    
    headers = {
        "User-Agent": user_agent,
        "referrer": 'https://google.com',
        "accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8"
    }
    req = urllib.request.Request(
        url, 
        data=None, 
        headers=headers
    )

    f = urllib.request.urlopen(req)

    return f.read().decode('utf-8')
    
# Parse the scraped html file 
# and return strings of time stamp, company name and full text of earnings call transcript
def get_one_earnings_call(url):
    html = download_url(url)
    soup_obj = BeautifulSoup(html, 'html.parser')
    #get the title
    title_obj = soup_obj.find_all("h1")
    title = title_obj[0].get_text()
    #get company name
    company_name = title.split()[0]
    #get time stamp string
    time_str = soup_obj.find("div",{"class":"a-info clearfix"}).find("time")['content']
    #get full text
    paras = soup_obj.find_all("p")
    full_text = "\n".join([p.text for p in paras])
    return time_str,company_name,title,full_text
    

# Split one transcript text into introduction and Q&A, and compute the tones for both parts
# Input: a string of full text of one earnings call transcript
# Outputs: introduction tone, Q&A tone, abnormal tone which is the difference btw the tones of introduction and Q&A 
def parse_one_call_transcript(doc):
    doc = doc.upper()
    div = re.findall('QUESTION.{0,2}AND.{0,2}ANSWER.{0,10}\n|QUESTION.{0,2}&.{0,2}ANSWER.{0,10}\n',doc)
    if len(div) == 0:
        div = re.findall('QUESTION.*AND.ANSWER.{0,10}OPERATOR',doc)
    if len(div) == 0:
        div = re.findall('Q&A.{0,8}',doc)
    
    sections = doc.split(div[0])
    intro = sections[0]
    qna = sections[1]
    #compute linguistic characteristics using L and McDonauld dictionary 
    odata_intro = feature_extraction(intro)
    odata_qna = feature_extraction(qna)
    #tone = positive sentiment - negative sentiment
    tone_intro = odata_intro[3] - odata_intro[4]
    tone_qna = odata_qna[3] - odata_qna[4]
    #compute abnormal tone which is difference btw introduction tone and q&a tone
    tone_ab = tone_intro - tone_qna
    
    return tone_intro,tone_qna, tone_ab

# Extract linguistic characteristics from a string of text using Loughran and McDonald Dictionary
# Input: a string of text
# Outputs: a list of numbers for linguistic features (e.g.percentage of negative, positive words)
def feature_extraction(doc):
    
    vdictionary = {}
    _odata = [0] * 17
    total_syllables = 0
    word_length = 0
    
    doc = doc.upper()
    tokens = re.findall('\w+', doc)  # Note that \w+ splits hyphenated words
    for token in tokens:
        if not token.isdigit() and len(token) > 1 and token in lm_dictionary:
            _odata[2] += 1  # word count
            word_length += len(token)
            if token not in vdictionary:
                vdictionary[token] = 1
            if lm_dictionary[token].positive: _odata[3] += 1
            if lm_dictionary[token].negative: _odata[4] += 1
            if lm_dictionary[token].uncertainty: _odata[5] += 1
            if lm_dictionary[token].litigious: _odata[6] += 1
            if lm_dictionary[token].weak_modal: _odata[7] += 1
            if lm_dictionary[token].moderate_modal: _odata[8] += 1
            if lm_dictionary[token].strong_modal: _odata[9] += 1
            if lm_dictionary[token].constraining: _odata[10] += 1
            total_syllables += lm_dictionary[token].syllables

    _odata[11] = len(re.findall('[A-Z]', doc))
    _odata[12] = len(re.findall('[0-9]', doc))
    # drop punctuation within numbers for number count
    doc = re.sub('(?!=[0-9])(\.|,)(?=[0-9])', '', doc)
    doc = doc.translate(str.maketrans(string.punctuation, " " * len(string.punctuation)))
    _odata[13] = len(re.findall(r'\b[-+\(]?[$€£]?[-+(]?\d+\)?\b', doc))
    _odata[14] = total_syllables / _odata[2]
    _odata[15] = word_length / _odata[2]
    _odata[16] = len(vdictionary) #number of unqiue words
    
    # Convert counts to %
    for i in range(3, 10 + 1):
        _odata[i] = (_odata[i] / _odata[2]) * 100
    # Vocabulary
        
    return _odata


# Compute ground truth percentage changes of stock price and volatility
# Inputs: time stamp, company ticker, number of days after and before the call
# Outputs: price and volatility changes
def compute_stock_prop(call_time,ticker,time_window):
    start_time = call_time - timedelta(days = time_window[0] + 5)#add extra days to account for weekend 
    end_time = call_time + timedelta(days = time_window[1] + 5) 
    stock_data = pdr.get_data_yahoo(ticker,start = start_time,end = end_time).reset_index()
    if len(stock_data) < 6:
        return None
    stock_data['Date'] = pd.to_datetime(stock_data['Date'])
    
    #split into before and after earnings call
    stock_before = stock_data[stock_data['Date'] <= call_time]['Adj Close']
    stock_after = stock_data[stock_data['Date'] > call_time]['Adj Close']
    stock_before = stock_before[-time_window[0]:]
    stock_after = stock_after[0:time_window[1]]
    
    #compute stock price and volatitiy change
    price_change  = (stock_after.mean() - stock_before.mean())/stock_before.mean()
    vol_before = stock_before.std()*np.sqrt(len(stock_before))
    vol_after = stock_after.std()*np.sqrt(len(stock_after))
    vol_change = (vol_after - vol_before)/vol_before
    
    return price_change,vol_change