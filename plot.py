import plotly.plotly as py
from plotly.graph_objs import *
import json
import urllib2
from pprint import pprint

# response = urllib2.urlopen('https://blockchain.info/charts/market-price?showDataPoints=false&timespan=all&show_header=true&daysAverageString=1&scale=0&format=json&address=')
data = json.load(response)

py.sign_in('pico', 'hpr9mlm9ze')

x_data = data['values'][:]
for val in range(len(x_data)):
  x_data[val] = x_data[val]['x']
y_data = data['values'][:]
for val in range(len(y_data)):
  y_data[val] = y_data[val]['y']

print x_data
# print "JASLDJKASLKDJAS"
# print y_data

trace0 = Scatter(
    x=x_data,
    y=y_data
)

data = Data([trace0]) #, trace1])
plot_url = py.plot(data, filename='basic-line')