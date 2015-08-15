from blockchain import exchangerates

ticker = exchangerates.get_ticker()
#print the 15 min price for every currency
for k in ticker:
    print k, ticker[k].p15min