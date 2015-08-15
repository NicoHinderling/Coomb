import requests
import json
import csv
from time import sleep
from datetime import datetime

currencies = ['btc', 'banx', 'clam', 'pro']

while True:
    for currency in currencies:
        url = "http://coinmarketcap-nexuist.rhcloud.com/api/{}/price".format(currency)
        r = requests.get(url)
        

        current_rate = currency + ", " + str(r.json()["usd"]) + " usd \n"
        fd = open(currency + '.csv','a')
        fd.write(str(datetime.utcnow()) + ", " + current_rate)
        fd.close()
        
    sleep(300)
