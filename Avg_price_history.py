import urllib2
import csv
import locale
locale.setlocale(locale.LC_ALL, '')

btc_response = urllib2.urlopen('https://api.bitcoinaverage.com/history/USD/per_day_all_time_history.csv')
csv_f = csv.reader(btc_response)

# assigning the average column from api call to an array
year_row = 0.0
counter = 0
for row in csv_f:
    if not row[0].find('2012'):
        year_row += float(row[3])
        counter += 1

print year_row
print counter
print year_row / counter
