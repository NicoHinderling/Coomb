from pprint import pprint
import csv

def find_2nd(string, substring):
   return string.find(substring, string.find(substring) + 1)

values = []
with open('btc.csv', 'rb') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
    for row in spamreader:
        s = ' '.join(row)
        m = find_2nd(s, ',')
        values.append(s[m+2:-5])
pprint(values)