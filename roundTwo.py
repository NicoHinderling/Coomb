from pprint import pprint
import csv

def find_2nd(string, substring):
   return string.find(substring, string.find(substring) + 1)

def buySellExperiment(currency):
    values = []
    first95 = []
    with open(currency + '.csv', 'rb') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        init = 0
        length = 0
        for row in spamreader:
            s = ' '.join(row)
            r = s.index(',')
            m = find_2nd(s, ',')
            values.append([s[:r],s[m+2:-5]])
            if length < 95:
                first95.append(s[m+2:-5])
            length+=1

    found = False
    for n in values:
        if(n[1] < min(first95)):
            print "Buy up Bitcoin! The time is " + n[0]
            found = True
        if(n[1] > max(first95)):
            print "Time to sell! The time is " + n[0]

    if found == False:
        print "No deal!"

buySellExperiment('btc')