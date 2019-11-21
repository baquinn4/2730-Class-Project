import tweepy
import keys
import csv
import logging
import time
import sys

## This script will create the dataset for my model
## Takes a username(as command line arg), creates a list of all of the users followers and then creates
## 		an individual .csv file for each friend and user filled with tweets, retweets and unusable tweets
##		are excluded from dataset.

#handles some exceptions and provides a manual
def man():
	print("\ncreateDataSet.py takes only one argument, a super user from which twitter followers will be added to a list")
	print("\ncreateDataSet.py will then create individual .csv files for each follower and store their tweets each with a unique id")
	quit()

if len(sys.argv) > 2:
	man()

if sys.argv[1] == "man":
	man()

#handles rate limit 
logging.basicConfig()
logger = logging.getLogger('tweepy.binder')
print(logger.info)

auth = tweepy.OAuthHandler(keys.CONSUMER_KEY,
	keys.CONSUMER_SECRET)
auth.set_access_token(keys.ACCESS_TOKEN,
	keys.ACCESS_TOKEN_SECRET)
api = tweepy.API(auth,wait_on_rate_limit=True,wait_on_rate_limit_notify=True,compression=True)

superUser = sys.argv[1]
#gets tweets of user and creates csv file for user
def getTweets(api,username):

	page = 1
	deadend = False
	tweet_counter = 1

	with open(username + '.csv','wb') as csvfile:
		filewriter = csv.writer(csvfile,delimiter=',',quotechar= '|', quoting=csv.QUOTE_MINIMAL)
		#ID will be an integer typecasted to a string to represent ID # will be passed by get_user_tweets.py
		filewriter.writerow(["ID","Tweet"])
		
	
		while page < 15:
			
			tweets = api.user_timeline(username,page = page)

			for tweet in tweets:
				tweetString = str(unicode(tweet.text.encode("utf-8"), errors='ignore'))
				if not tweet.retweeted and 'RT @' not in tweetString:
					filewriter.writerow([str(tweet_counter),tweetString])
					tweet_counter = tweet_counter + 1
					print("record added to: %s.csv" %username)

			if not deadend:
				page = page + 1
	print("15 pages done, file closed")
			
#returns list of friends of "super" user, list is iterated over, each element in list is passed
#	to getTweets()
def getFriends(api):

	friendslist = []
	print("Getting friends from super user: " + superUser)
	c = tweepy.Cursor(api.friends, screen_name=superUser).items()
	
	
	for user in c:
		if len(friendslist) < 10:
			friendslist.append(str(user.screen_name))
			print("friendslist appended: " + str(user.screen_name))
			print(len(friendslist))
		else:
			break
		time.sleep(1)

	return friendslist

fl = getFriends(api)
print(len(fl))
for i in range(len(fl)):
	getTweets(api,fl[i])

print("Done")



