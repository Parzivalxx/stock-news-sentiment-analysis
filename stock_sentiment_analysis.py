from urllib.request import urlopen, Request
from bs4 import BeautifulSoup
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pandas as pd
import matplotlib.pyplot as plt
import time
import math

finviz_url = "https://finviz.com/quote.ashx?t="

news_tables = {}
tickers = []
print("Welcome to the stock sentiment analysis program")
print("You may analyse up to 4 different stocks below, enter 1 when done")
time.sleep(0.5)

i = 0
while i < 4:
    ticker = input(f"Please enter stock {i+1}: ")
    if ticker == "1":
        if len(tickers) == 0:
            print("Require at least 1 stock for analysis")
            continue
        break
    if ticker in tickers:
        print("Error: that stock was already entered, please try again")
        continue
    tickers.append(ticker)
    i += 1

print("Proceeding to analyse...")
time.sleep(1)

for ticker in tickers:
    print(f"Analysing {ticker}...")
    url = finviz_url + ticker
    req = Request(url=url, headers={'user-agent': 'my-app'})
    response = urlopen(req)
    html = BeautifulSoup(response, features='html.parser')
    news_table = html.find(id='news-table')
    news_tables[ticker] = news_table

data = []

for ticker, news_table in news_tables.items():
    for row in news_table.findAll("tr"):
        title = row.a.text
        dateinfo = row.td.text.split()
        if len(dateinfo) == 1:
            time = dateinfo[0]
        else:
            date = dateinfo[0]
            time = dateinfo[1]
        data.append([ticker,date,time,title])

df = pd.DataFrame(data, columns = ["ticker", "date", "time", "title"])

vader = SentimentIntensityAnalyzer()

getscorefunc = lambda title: vader.polarity_scores(title)["compound"]

df['compound'] = df['title'].apply(getscorefunc)
df['date'] = pd.to_datetime(df["date"]).dt.date

mean_df = df.groupby(['ticker', 'date']).mean()

findings_dates = {}
findings_sentiment = {}
tker = ""
i = 0

for finding in mean_df.index.values:
    ticker = finding[0]
    date = finding[1]
    if ticker == tker:
        findings_dates[ticker].append(date)
        findings_sentiment[ticker].append(mean_df.compound[i])
    else:
        tker = ticker
        findings_dates[ticker] = [date]
        findings_sentiment[ticker] = [mean_df.compound[i]]
    i += 1

fig, axs = plt.subplots(2,2)
fig.suptitle("Sentiment analysis on stock news")
fig.tight_layout()

i = 0
j = 0
for ticker in tickers:
    if i == 2:
        i = 0
        j += 1
    axs[j, i].plot(findings_dates[ticker], findings_sentiment[ticker])
    axs[j, i].set_title(ticker)
    axs[j, i].set(xlabel = "Dates", ylabel = "Compound score")
    axs[j, i].xaxis.set_tick_params(labelsize = 6)
    #for tick in axs[j, i].get_xticklabels():
    #    tick.set_rotation(45)
    i += 1

plt.show()