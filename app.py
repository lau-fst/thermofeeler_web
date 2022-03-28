import streamlit as st
import requests
import matplotlib.pyplot as plt
import re
from wordcloud import WordCloud
from nltk.tokenize import word_tokenize
import pandas as pd
import numpy as np
import seaborn as sns
import plotly.figure_factory as ff
import plotly.graph_objects as go


st.set_page_config(page_title="ThermoFeeler", page_icon="🌡",
        layout="wide",
        initial_sidebar_state="auto")

title = """<p style="font-family:'Tangerine'; color:darkred; font-size:42px;">ThermoFeeler</p>"""

st.markdown(title, unsafe_allow_html=True)
query = None

st.sidebar.markdown("""
    # Sobre
    Este aplicativo web foi desenvolvido em dez dias como projeto final do bootcamp
    de Data Science do Le Wagon.

    Após inserido o texto no campo de pesquisa, realizamos um _query_ na API do Twitter,
    selecionando apenas tweets em português. O resultado é enviado à nossa API
    hospedada no Google Cloud Platform, a qual possui uma Rede Neural que realiza
    o preprocessamento dos tweets e prevê o sentimento.
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
query_in= st.text_input('Insira abaixo a sua pesquisa')

if query_in != '':
    query_words = query_in.split(' ')

    if len(query_words) == 1:
        query = query_in

    if len(query_words) == 2:
        st.markdown('Como você prefere buscar?')
        col_1,col_2, = st.columns(2)
        possibility_1 = (f'"{query_words[0]} {query_words[1]}"')
        if col_1.button(possibility_1):
            query = (possibility_1)
        possibility_2 = (f'"{query_words[0]}" e "{query_words[1]}"')
        if col_2.button(possibility_2):
            query = query_in

    if len(query_words) == 3:
        st.markdown('Como você prefere buscar?')
        col_1,col_2,col_3, col_4= st.columns(4)
        possibility_1 = (f'"{query_words[0]} {query_words[1]} {query_words[2]}"')
        if col_1.button(possibility_1):
            query = possibility_1
        possibility_2 = (f'"{query_words[0]} {query_words[1]}" e "{query_words[2]}"')
        if col_2.button(possibility_2):
            query = (f'"{query_words[0]} {query_words[1]}" {query_words[2]}')
        possibility_3 = (f'"{query_words[0]}" e "{query_words[1]} {query_words[2]}"')
        if col_3.button(possibility_3):
            query = (f'{query_words[0]} "{query_words[1]} {query_words[2]}"')
        possibility_4 = (f'"{query_words[0]}" e "{query_words[1]}" e "{query_words[2]}"')
        if col_4.button(possibility_4):
            query = query_in

    if len(query_words) > 3:
        st.markdown("Sua consulta excede 3 palavras, se você deseja continuar, consulte a documentação do twitter api para realizar sua busca exatamente como você deseja.")
        query = query_in

    if query != None :

        with st.spinner('Buscando os tweets...'):
            url = f'https://thermofeeler-6hn6fqkota-uc.a.run.app/predict_query?query={query}&max_results=100'
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
                    'só','também','ah','q','g','oh','eh','vc','tbm','também','tambem','voceh','você','voce','rt',
                    'é','n','não','nao','pro','pra','tá','ta','p']

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

            st.markdown('\n')
            st.markdown('\n')

            fig, (ax1,ax2) = plt.subplots(1,2,figsize=(20,6))
            fig.suptitle(f"Os {response[-1]['total']} tweets mais recentes estão distribuídos da seguinte forma:",
                         size=20,y=1.08)
            plt.tight_layout(pad=60)

            ax1.set_title(f"Distribuição de sentimentos",loc='center',size=16,pad=15)

            ax1.pie([response[-1]['negative total'],response[-1]['positive total'],response[-1]['neutral total']],
                    explode=[0.05,0.05,0.05],
                    labels=['Negativo','Positivo','Neutro'],
                    colors=['#E13B17','limegreen','lightgray'],
                    autopct='%1.1f%%',
                    textprops={'fontsize':14})


            ax2.set_title(f"Dispositivos mais utilizados",loc='center',size=16,pad=15)

            source_list=[entry.replace('Twitter','').replace('for','').strip() for entry in response[0][3]]
            count_list=[]
            for entry in source_list:
                count_list.append(1)
            source_df=pd.DataFrame(pd.Series(count_list,index=source_list).groupby(level=0).count(),columns=['count'])

            source_df['count']=source_df['count'].apply(lambda x: None if x < 3 else int(x))
            source_df.dropna(inplace=True)

            if (len(source_list)-source_df.sum()[0]) != 0:
                other_df=pd.DataFrame([len(source_list)-source_df.sum()[0]],columns=['count'],index=['Outros'])
                source_df=pd.concat([source_df,other_df])

            explode_list=[]
            for i in range(source_df.shape[0]):
                explode_list.append(0.05)

            ax2.pie(source_df['count'],
                    explode=explode_list,
                    labels=source_df.index,
                    autopct='%1.1f%%',
                    textprops={'fontsize':14})

            my_circle=plt.Circle((0,0), 0.82, color='white')
            ax2.add_patch(my_circle)
            st.pyplot(fig)

            fig, ax = plt.subplots(figsize=(10,3))
            def black_color_func(word, font_size, position,orientation,random_state=None, **kwargs):
                return("hsl(0,100%, 1%)")

            # set width and height to higher quality, 3000 x 2000
            wordcloud = WordCloud(background_color="white",colormap="Blues",
                                  width=600,height=200).generate(string)

            # set the word color to black
            wordcloud.recolor(color_func = black_color_func)

            # plot the wordcloud
            plt.imshow(wordcloud,interpolation='lanczos')

            # remove plot axes
            plt.axis("off")
            plt.margins(x=0, y=0)
            st.pyplot(fig)

            st.write("Para acessar a análise da semana passada inteira, pressione o botão abaixo")

            if st.button("Aqui"):
                max_results = 10
                query_2 = query
                url_2 = f"https://thermofeeler-6hn6fqkota-uc.a.run.app/predict_week?query={query_2}&max_results={max_results}"
                tweets_week, predict_list = requests.get(url_2).json()

                for index, value in enumerate(predict_list):
                    if value == 0:
                        predict_list[index] = -1
                    elif value == 1:
                        predict_list[index] = 0
                    elif value == 2:
                        predict_list[index] = 1

                df = pd.DataFrame(tweets_week[2],predict_list).reset_index()
                df['date'] = pd.to_datetime(df[0], format='%Y-%m-%d').dt.strftime("%d/%m/%Y")
                df = df.drop(columns=[0])

                if max_results == 10 :
                    y_ticks = [0,5,10]
                elif max_results == 20 :
                    y_ticks = [0,5,10,15,20]

                colors = ["#ed6a5a","#eb6565",'#c24747',"#da3030","#ff2a00","#FF0101"]
                fig, ax = plt.subplots(figsize=(20,3))
                for i,date,color in zip(range(7), df.date.unique(),colors):
                    plt.subplot(1,6,i+1)
                    sentiment_day = df[df['date'] == date]['index']
                    sns.histplot(sentiment_day, color=color, binwidth=0.4)
                    plt.ylabel('')
                    plt.xlabel(date)
                    plt.yticks(y_ticks)
                    plt.xticks([-1,0,1])
                    plt.xlim(-1.2, 1.2)

                st.pyplot(fig)
