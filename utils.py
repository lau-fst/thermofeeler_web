import re
from nltk.tokenize import word_tokenize

def preproc_func(tweet):
            '''Does the preprocessing of the tweets'''

            # stopwords: remove articles, prepositions, conjunctions etc
            stopwords=['a','te','tu','tua','tuas','tém','um','uma','você','vocês','vos','à','às','ao','aos',
                'aquela','aquelas','aquele','aqueles','aquilo','as','até','com','como','da','das','de',
                'dela','delas','dele','deles','depois','do','dos','e','ela','elas','ele','eles','em',
                'entre','essa','essas','esse','esses','esta','eu','foi','fomos','for','fora','foram',
                'forem','formos','fosse','fossem','fui','fôramos','fôssemos', 'isso','isto','já','lhe',
                'lhes','me','mesmo','meu','meus','minha','minhas','muito','na','nas','no','nos','nossa',
                'nossas','nosso','nossos','num','numa','nós','o','os','para','pela','pelas','pelo','pelos',
                'por','qual','quando','que','quem','se','seja','sejam','sejamos','sem','serei','seremos',
                'seria','seriam','será','serão','seríamos','seu','seus','somos','sou','sua','suas','são',
                'só','também','ah','q','g','oh','eh','vc','tbm','também','tambem','voceh','você','voce','rt',
                'é','n','não','nao','pro','pra','tá','ta']

            tweet = tweet.lower() # lowercase

            tweet=re.sub('https?://[A-Za-z0-9./]+','',tweet) # remove links que começam com https?://
            tweet=re.sub('https://[A-Za-z0-9./]+','',tweet) # remove links que começam com https://
            tweet=re.sub('http://[A-Za-z0-9./]+','',tweet) # remove links que começam com http://

            tweet = re.sub(r'@[A-Za-z0-9_]+','',tweet) # remove @mentions
            tweet = re.sub(r'#','',tweet) # remove #hashtags

            tweet = re.sub(r'[^\w\s]','',tweet) # remove remove punctuation
            tweet = re.sub(r'[0-9]','',tweet) # remove numbers

            word_tokens=word_tokenize(tweet) # tokenize

            filtered_tweet = [w for w in word_tokens if not w in stopwords] # remove stopwords

            return filtered_tweet
