from pprint import pprint
import csv

def find_2nd(string, substring):
   return string.find(substring, string.find(substring) + 1)

def buySellExperiment(currency):
    values_1 = []
    values_2 = []

    with open(currency + '.csv', 'rb') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        init = 0
        length = 0
        for row in spamreader:
            s = ' '.join(row)
            r = s.index(',')
            m = find_2nd(s, ',')
            if length % 2 == 0:
                values_1.append(float(s[m+2:-5]))
            else:
                values_2.append(float(s[m+2:-5]))
            length += 1
    
    return values_1, values_2

print(len(buySellExperiment('pro')[1]))
print(buySellExperiment('pro')[1])