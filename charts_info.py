import json
import urllib2
from time import sleep
from pprint import pprint

response = urllib2.urlopen('https://blockchain.info/charts/market-price?showDataPoints=false&timespan=all&show_header=true&daysAverageString=1&scale=0&format=json&address=')
data = json.load(response)
# for x in data['values']:
#     print x
#     sleep(1)

nums = 0
# for vals in data['values']:
#     nums += long(vals['y'])

# print nums

eight_weeks_count = 0
for i in nums[:-57:-1]:
    eight_weeks_count += i

print 'The past 56 data points (8 weeks) is a total of: ' + str(eight_weeks_count)

# one_week_count = 0
# for ints in nums[:-8:-1]:
#     one_week_count += ints
# print 'The past 7 data points (1 week) is a total of: ' + str(one_week_count)
