import streamlit as st
import requests
import matplotlib.pyplot as plt
import re
from wordcloud import WordCloud
from nltk.tokenize import word_tokenize
import pandas as pd
import numpy as np
import seaborn as sns


st.set_page_config(page_title="ThermoFeeler", page_icon="🌡",
        layout="wide",
        initial_sidebar_state="auto")

title = """<p style="font-family:'Tangerine'; color:Red; font-size:42px;">ThermoFeeler</p>"""

st.markdown(title, unsafe_allow_html=True)

query=None


st.sidebar.markdown("""
    # Sobre
    Este aplicativo web foi desenvolvido em dez dias como projeto final do bootcamp
    de Data Science do Le Wagon.

    Após inserido o texto no campo de pesquisa, realizamos um _query_ na API do Twitter.
    O resultado é enviado à nossa API hospedada no Google Cloud Platform,
    a qual possui uma Rede Neural que realiza o preprocessamento dos tweets e prevê
    o sentimento.
    """)

st.sidebar.markdown(f"""# Equipe""")

sidecol1, sidecol2,sidecol3 = st.sidebar.columns(3)
sidecol1.image("https://avatars.githubusercontent.com/u/98327733?s=400&u=fa490bc2b388515f06385e3b968551f8055696bf&v=4")
sidecol2.write('\n')
sidecol2.write('[Guilherme Chacon](https://github.com/GSChacon)')
sidecol2.write('\n')
sidecol2.write('\n')
sidecol1.image("https://avatars.githubusercontent.com/u/80108511?v=4")
sidecol2.write('[Haroldo Oliveira](https://github.com/haroldo-oliveira)')
sidecol2.write('\n')
sidecol2.write('\n')
sidecol2.write('\n')
sidecol1.image("https://avatars.githubusercontent.com/u/98071615?v=4")
sidecol2.write('[Lauranne Fossat](https://github.com/lau-fst)')


st.markdown("""Realize uma pesquisa no Twitter em português:""")
query_in= st.text_input('Exemplo: Covid')

if query_in != "" :
    query_words = query_in.split(' ')

    if len(query_words) ==1:
        query = query_in

    if len(query_words) == 2:
        st.markdown('Como você prefere buscar?')
        col_1,col_2, = st.columns(2)
        possibility_1 = (f'"{query_words[0]} {query_words[1]}"')
        if col_1.button(possibility_1):
            query = (possibility_1)
        possibility_2 = (f'"{query_words[0]}" E "{query_words[1]}"')
        if col_2.button(possibility_2):
            query = query_in

    if len(query_words) == 3:
        st.markdown('Como você prefere buscar?')
        col_1,col_2,col_3, col_4= st.columns(4)
        possibility_1 = (f'"{query_words[0]} {query_words[1]} {query_words[2]}"')
        if col_1.button(possibility_1):
            query = possibility_1
        possibility_2 = (f'"{query_words[0]} {query_words[1]}" E "{query_words[2]}"')
        if col_2.button(possibility_2):
            query = (f'"{query_words[0]} {query_words[1]}" {query_words[2]}')
        possibility_3 = (f'"{query_words[0]}" E "{query_words[1]} {query_words[2]}"')
        if col_3.button(possibility_3):
            query = (f'{query_words[0]} "{query_words[1]} {query_words[2]}"')
        possibility_4 = (f'"{query_words[0]}" E "{query_words[1]}" E "{query_words[2]}"')
        if col_4.button(possibility_4):
            query = (f'{query_words[0]} {query_words[1]} {query_words[2]}')

    if len(query_words) > 3:
        query = query_in

    if query != None :
        with st.spinner('Buscando os tweets...'):
            url = f'https://thermofeeler-6hn6fqkota-uc.a.run.app/predict_query?query={query}&max_results=10'
            response= requests.get(url)

        if response.status_code != 200:
            st.error("A busca inserida é inválida, desatualizada ou inexistente")
        else :
            response = response.json()
            col1, col2, col3, col4 = st.columns(4)
            col1.write(f"Total de tweets obitidos: {response[-1]['total']}")
            col2.write(f"Total de tweets negativos: {response[-1]['negative total']}")
            col3.write(f"Total de tweets neutros: {response[-1]['neutral total']}")
            col4.write(f"Total de tweets positivos: {response[-1]['positive total']}")

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

            ax.set_title(f"Os {response[-1]['total']} tweets mais recentes estão distribuídos da seguinte forma:",
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

            st.write("Para acessar a análise da semana passada inteira, pressione o botão abaixo")
            if st.button("Aqui"):
                url = f"https://thermofeeler-6hn6fqkota-uc.a.run.app/predict_week?{query}=apple&max_results=20"
                response_week = requests.get(url)

                df = pd.DataFrame(response_week[0][2],response_week[1]).reset_index()
                df['date'] = df[0].dt.strftime("%d/%m/%Y")
                df = df.drop(columns=[0])

                colors = ["#20B2AA","#FF4040","#FFD700","#00CD00","#FF9912", "#FF1493"]
                fig, ax = plt.subplots(figsize=(20,3))
                for i,date,color in zip(range(7), df.date.unique(),colors):
                    plt.subplot(1,6,i+1)
                    sentiment_day = df[df['date'] == date]['index']
                    sns.histplot(sentiment_day, color=color, kde=True)
                    plt.ylabel('')
                    plt.xlabel(date)
                    plt.yticks([0,5,10,15,20])
                    plt.xticks([-1,0,1])

                st.pyplot(fig)
