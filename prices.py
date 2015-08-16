import csv

def find_2nd(string, substring):
   return string.find(substring, string.find(substring) + 1)

with open('btc.csv', 'rb') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
    counter = 1
    for row in spamreader:
        s = ' '.join(row)
        m = find_2nd(s, ',')
        if counter % 2 == 1:
            print ('open_price', s[m+2:-5])
        else:
            print ('close_price', s[m+2:-5])
        counter+=1