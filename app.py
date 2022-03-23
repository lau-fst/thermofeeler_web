import streamlit as st
import requests
import matplotlib.pyplot as plt
import re
from wordcloud import WordCloud
from nltk.tokenize import word_tokenize

st.set_page_config(page_title="ThermoFeeler", page_icon="🌡",
        layout="centered",
        initial_sidebar_state="auto")

title = """<p style="font-family:'Tangerine'; color:Red; font-size:42px;">ThermoFeeler</p>"""
st.markdown(title, unsafe_allow_html=True)

st.markdown("""Enter a twitter query""")
query= st.text_input('Example : Apple', 'Apple')

with st.spinner('Wait for it...'):
    url = f'https://thermofeeler-6hn6fqkota-uc.a.run.app/predict_query?query={query}&max_results=10'
    response = requests.get(url).json()
st.success('Done!')

col1, col2, col3, col4 = st.columns(4)
col1.write(f"Total number of tweets retrieved : {response[-1]['total']}")
col2.write(f"Total number of negative tweets  : {response[-1]['negative total']}")
col3.write(f"Total number of neutral tweets : {response[-1]['neutral total']}")
col4.write(f"Total number of positive tweets : {response[-1]['positive total']}")


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
          'só','também','ah','q','g','oh','eh','vc','tbm','também','tambem','voceh','você','voce','rt']

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

preproc_tweets=[]
for tweet in response[0][0]:
    preproc_tweets.append(preproc_func(tweet))

lista_palavras=[]
for tweet in preproc_tweets:
    for word in tweet:
        lista_palavras.append(word)
string=' '.join(lista_palavras)


fig, ax = plt.subplots(figsize=(20,6))
ax.pie([response[-1]['negative total'],response[-1]['positive total'],response[-1]['neutral total']],
        explode=[0.05,0.05,0.05],
        labels=['Negativo','Positivo','Neutro'],
        colors=['darkred','lightgreen','lightgray'],
        autopct='%1.1f%%',
        textprops={'fontsize': 14})

ax.set_title('Os 10 tweets mais recentes estão distribuídos da seguinte forma:',
             size=16,pad=20,loc='center')
st.pyplot(fig)

fig, ax = plt.subplots(figsize=(20,6))
word_cloud = WordCloud(background_color = 'white')
word_cloud.generate(string)
plt.imshow(word_cloud, interpolation='bilinear')
plt.title('As palavras que mais aparecem nesses tweets são:',
         size=20,pad=40,loc='center')
plt.axis("off")

st.pyplot(fig)
