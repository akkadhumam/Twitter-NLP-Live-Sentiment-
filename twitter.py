from tweepy import Stream
from tweepy import OAuthHandler
from tweepy.streaming import StreamListener
import time
import json
import sent_mod as s





#consumer key, consumer secret, access token, access secret.
#Need to get your own from developer.Twitter
ckey=""
csecret=""
atoken=""
asecret=""

#Keyword to analyze pertaining tweets
string = ""

class listener(StreamListener):

    def on_data(self, data):

        all_data = json.loads(data)
        #Relevant Data

        tweet = all_data['text']
        #user = all_data['user']['name']
        #Analysis
        sentiment,confidence= s.sentiment(tweet)

        #Display
        print(tweet,sentiment,confidence)

        #If we're confident of our assesment,
        #write to file to plot later
        if confidence*100 >= 66:
            output = open('twitter-out.txt','a')
            output.write(sentiment)
            output.write('\n')
            output.close()




        print(tweet)

        return True

    def on_error(self, status):
        print(status)


auth = OAuthHandler(ckey, csecret)
auth.set_access_token(atoken, asecret)

twitterStream = Stream(auth, listener())
twitterStream.filter(track=[string])
