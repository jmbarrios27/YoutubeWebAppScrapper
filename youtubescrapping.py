# LIBRERIAS DE YOUTUBE
from datetime import date
from sys import path
from nltk import data
from nltk.featstruct import retract_bindings
from numpy.core.fromnumeric import size
from pandas.core.frame import DataFrame
from selenium import webdriver
from webdriver_manager.firefox import GeckoDriverManager
from bs4 import BeautifulSoup
import time
import pandas as pd
import os
import warnings
import re
import string

# Procesamiento Natural de Lenguaje y Análisis de sentimiento.
from sentiment_analysis_spanish import sentiment_analysis
from wordcloud import WordCloud
from nltk.corpus import stopwords
from textblob import TextBlob
import gensim
import itertools,collections
import nltk
import base64
import xlsxwriter
from sklearn.feature_extraction.text import TfidfVectorizer,  CountVectorizer
from sklearn.cluster import MiniBatchKMeans
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV,KFold, StratifiedKFold

# VISUALIZACIÓN DE DATOS
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import seaborn as sns
import plotly.figure_factory as ff
import plotly.express as px
import plotly.graph_objs as go

# WEB APP
import streamlit as st
st.set_option('deprecation.showPyplotGlobalUse', False)
import urllib.request as url
from io import BytesIO
from PIL import Image
import urllib.request as url
import numpy as np
import requests as r
from streamlit_player import st_player
import sys 


st.cache(suppress_st_warning=True)
start_time = time.time()
print('*************************************************************************')
print('INICIANDO EJECUCIÓN')

# Carpeta de descarga
def get_download_path():
    """Returns the default downloads path for linux or windows"""
    if os.name == 'nt':
        import winreg
        sub_key = r'SOFTWARE\Microsoft\Windows\CurrentVersion\Explorer\Shell Folders'
        downloads_guid = '{374DE290-123F-4565-9164-39C4925E467B}'
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, sub_key) as key:
            location = winreg.QueryValueEx(key, downloads_guid)[0]
        return location
    else:
        return os.path.join(os.path.expanduser('~'), 'downloads')

# Guardar carpeta de descarga.
carpeta_descarga = get_download_path()

# LIMPIEZA DE FECHA DE PUBLICACIÓN
def limpieza_fecha_comentario(text):
    text = re.sub('hace', '', text)
    text = re.sub('(editado)', '', text)
    return text


# funcion para eliminar spanish stopwords
def removeStopwords( texto):
    blob = TextBlob(texto).words
    outputlist = [word for word in blob if word not in stopwords.words('spanish')]
    return(' '.join(word for word in outputlist))


# Limpieza de textos
def text_clean(text):
    # Remover rt
    text = re.sub('RT@', ' ', text)
    text = re.sub('RT @', ' ', text)
    # Remover Menciones
    text = re.sub("@\S+", " ", text)
    # remover URLs
    text = re.sub("https*\S+", " ", text)
    # Remover Hash
    text = re.sub("#\S+", " ", text)
    # Remover Signos de puntuación
    text = re.sub("\'\w+", '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), ' ', text)
    # Remover numeros.
    text = re.sub('[0-9]+', '', text)
    # Remover doble espacios
    text = re.sub('\s{2,}', " ", text)
    # Remover caracteres faltantes
    text = re.sub('¡', " ", text)
    text = re.sub("¿", " ", text)
    # Trasnformar todo a texto y a miniscula.
    text = str(text).capitalize()
    return text


# Funcion para elimiar emojis

def remove_emoji(text):
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               u"\U00002500-\U00002BEF"  # chinese char
                               u"\U00002702-\U000027B0"
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               u"\U0001f926-\U0001f937"
                               u"\U00010000-\U0010ffff"
                               u"\u2640-\u2642"
                               u"\u2600-\u2B55"
                               u"\u200d"
                               u"\u23cf"
                               u"\u23e9"
                               u"\u231a"
                               u"\ufe0f"  # dingbats
                               u"\u3030"
                               u"\u261d"  # index pointing up
                               u"\u2705"  # check mark button
                               "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)

# Extrayendo coeficiente de Polaridad
def spanish_sentiment(data):
    sentiment = sentiment_analysis.SentimentAnalysisSpanish()
    return sentiment.sentiment(data)


# Extrayendo sentimiento
def sentimiento(Polaridad):
    if Polaridad <= 0.2:
        return 'NEGATIVO'
    elif Polaridad >= 0.21 and Polaridad <=0.29:
        return 'NEUTRAL'
    else:
        return 'POSITIVO'


# Crear botón de descarga de archivos.
def to_excel(df):
    output = BytesIO()
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    df.to_excel(writer, index=False, sheet_name='Sheet1')
    workbook = writer.book
    worksheet = writer.sheets['Sheet1']
    format1 = workbook.add_format({'num_format': '0.00'}) 
    worksheet.set_column('A:A', None, format1)  
    writer.save()
    processed_data = output.getvalue()
    return processed_data


def wordCloudGenerator(data, background_color):
        try:
            allwords = ' '.join([fk for fk in data.TEXTO_STOPWORD])
            wordcloud_todo = WordCloud(width=600, height=300, random_state=22, max_font_size=119, background_color= background_color).generate(allwords)
            fig_todo = plt.figure(figsize=(12,10))
            plt.imshow(wordcloud_todo, interpolation='bilinear')
            plt.axis('off')
            st.pyplot(fig_todo, use_column_width=True)
        
        except:
            st.write('No existen comentarios con este tipo de sentimiento')


def conteo_palabras(dataframe, color):
    try:
        # CONTEO DE TERMINOS
        #Convertiremos a X Varias listas dentro de listas, para luego ser tokenizadas y stemmizadas (llevads a su raiz)
        cleanWordsList_es = []
        stop_words_es = set(nltk.corpus.stopwords.words("spanish"))
        tokenizer_es = nltk.tokenize.RegexpTokenizer(r'\w+')
        for par_es in dataframe["COMENTARIO"].values:
            tmp_es = []
            sentences_es = nltk.sent_tokenize(par_es)
        for sent_es in sentences_es:
            sent_es = sent_es.lower()
            tokens_es = tokenizer_es.tokenize(sent_es)
            filtered_words_es = [w_es.strip() for w_es in tokens_es if w_es not in stop_words_es and len(w_es) > 1]
            tmp_es.extend(filtered_words_es)
        cleanWordsList_es.append(tmp_es)
        # Vamos a ver las 50 palabras más utilizadas en este dataset con tweets en español
        all_words_counter_es = list(itertools.chain(*cleanWordsList_es))
        # Create counter
        commonWordCount_es = collections.Counter(all_words_counter_es)
        #Creando DataFrame para luego graficar las 50 palabras que más se utilizarón
        final_word_count_es = pd.DataFrame(commonWordCount_es.most_common(50),
                                    columns=['palabras', 'conteo'])

        # -----------------------------------
        # -------------------------------------------
        fig_conteo_palabras, ax = plt.subplots(figsize=(12, 12))

        # final_word_count_es
        #Plot para ver conteo de palabras más utilizadas en textos en español
        final_word_count_es.sort_values(by='conteo').plot.barh(x='palabras',
                            y='conteo',
                            ax=ax,
                            color=color)

        ax.set_title("CONTEO DE TERMINOS")
        plt.figure(figsize=(40,40))
        st.pyplot(fig_conteo_palabras)

    except UnboundLocalError:
        st.write('No Existen Comentarios Suficientes para realizar este Gráfico')
    

def termFrequencyVocab(data,tipo_sentimiento):
    try:
        #Fitting TF-IDF
        tfidf = TfidfVectorizer()
        train_tv = tfidf.fit_transform(data['TEXTO_STOPWORD'])
        vocab = tfidf.get_feature_names()
        st.write(f'Vocabulario de Palabaras para textos en datos {tipo_sentimiento}')
        st.write(vocab)
    except ValueError:
            st.write('No existen Comentarios')
    except AttributeError:
            st.write('No existen comenatarios')
    

# Extrayendo sentimiento
def getSentiment(polaridad):
    try:
        if polaridad <= 0.2:
            return 0
        else:
            return 1
    except ValueError:
            st.write('No existen Comentarios')
    except AttributeError:
            st.write('No existen comenatarios')


def text_dist(data):
    try:
        # DISTRIBUCION DE TEXTOS BASADO EN SUBJETIVIDAD
        st.write('Distrubución de textos:')
        plt.figure(figsize=(12,10))
        sns.barplot(data=data, x="AUTOR", y='CARACTERES', palette='bright')
        plt.xticks(rotation=90)
        plt.show()
        st.pyplot()
    except ValueError:
            st.write('Insuficiente Cantidad de Palabras para generar Gráfico de Distribución de textos ')
    except AttributeError:
            st.write('Insuficiente Cantidad de Palabras para generar Gráfico de Distribución de textos ')



def get_top_text_ngrams(corpus, n, g, dataframe):
    vec = CountVectorizer(ngram_range=(g, g)).fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq = sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:n]


def plot_text_ngrams(df):
    try:
        plt.figure(figsize = (16,9))
        most_common_uni = get_top_text_ngrams(df.TEXTO_STOPWORD,20,1, dataframe=df)
        most_common_uni = dict(most_common_uni)
        sns.barplot(x=list(most_common_uni.values()),y=list(most_common_uni.keys()), palette='bright')
        plt.xticks(rotation=90)
        plt.show()
        st.pyplot()
    except ValueError:
        st.write('No Existen Datos Con este sentimiento - No se Desplegara este Gráfico')


def popular_users(data):
    try:
        # DISTRIBUCION DE TEXTOS BASADO EN SUBJETIVIDAD
        st.write('Usuarios con que más comentan')
        plt.figure(figsize=(12,10))
        sns.countplot(data=data, y="AUTOR", palette='bright')
        plt.xticks(rotation=90)
        plt.show()
        st.pyplot()
    except ValueError:
            st.write('Insuficiente Cantidad de Palabras para generar Gráfico de Distribución de textos ')
    except AttributeError:
            st.write('Insuficiente Cantidad de Palabras para generar Gráfico de Distribución de textos ')
############################# STREAMLIT ###############################3
image = Image.open('youtubelogo.jpg')
st.title("YOUTUBE SCRAPPER - TELERED")
st.markdown("<h3 style='text-align: center; color: magenta;'>CIENCIA DE DATOS</h3>", unsafe_allow_html=True)
st.image(image, caption='Logo YouTube')

# Add a selectbox to the sidebar:
add_selectbox = st.sidebar.selectbox(
    'Método de Scrapping',
    ('Link de Video','Archivo Excel')
)


# -----------------------------------------------------------------------
def input_url():
    try:
        url = st.text_input("Escriba el Link de YouTube:")
        count = 0
        while True:
            if url.startswith('https://www.youtube.com/watch?v=') == False:
                st.markdown("<h4 style='text-align: center; color: red;'>Este link no pertenece a un video de YouTube :(</h4>", unsafe_allow_html=True)
                break
            else:
                url.startswith('https://www.youtube.com/watch?v=')
                st.markdown("<h4 style='text-align: center; color: green;'>¡Link con Formato correcto! ¡Muy Bien Teler!</h4>", unsafe_allow_html=True)
                st.write('\N{grinning face with smiling eyes}')
                st.balloons()
                break
    except OSError:
        st.write('Colocar url con formato correcto')
    return url

url = input_url()

#--------------------- Barra de Progreso
st.markdown("""
<style>
.stProgress .st-bo {
    background-color: green;
}
</style>
""", unsafe_allow_html=True)
my_bar = st.progress(0)
for percent_complete in range(100):
    time.sleep(0.1)
    my_bar.progress(percent_complete + 1)

st.write('Link del video extraido: ',url)
st_player(url)

st.markdown("<h4 style='text-align: center; color: green;'>Cargando Todos los datos...</h4>", unsafe_allow_html=True)
def ScrapComment(url):
    option = webdriver.FirefoxOptions()
    option.add_argument("--headless")
    driver = webdriver.Firefox(executable_path=GeckoDriverManager().install(), options=option)
    driver.get(url)
    prev_h = 0
    while True:
        height = driver.execute_script("""
                function getActualHeight() {
                    return Math.max(
                        Math.max(document.body.scrollHeight, document.documentElement.scrollHeight),
                        Math.max(document.body.offsetHeight, document.documentElement.offsetHeight),
                        Math.max(document.body.clientHeight, document.documentElement.clientHeight)
                    );
                }
                return getActualHeight();
            """)
        driver.execute_script(f"window.scrollTo({prev_h},{prev_h + 200})")
        # fix the time sleep value according to your network connection
        time.sleep(1)
        prev_h +=200  
        if prev_h >= height:
            break
    soup = BeautifulSoup(driver.page_source, 'html.parser')
    driver.quit()

    print(f'INICIANDO EXTRACCIÓN DE LA URL: {url}')
    # TITULO DE VIDEO.
    title_text_div = soup.select_one('#container h1')
    title = title_text_div and title_text_div.text

    # COMENTARIO
    comment_div = soup.select("#content #content-text")
    comment_list = [x.text for x in comment_div]

    # AUTOR
    author_div = soup.select("#body #author-text")
    author_list = [y.text for y in author_div]

    # extract list
    vote_div = soup.select('#toolbar #vote-count-middle')
    vote_list = [z.text for z in vote_div]
    

    # Post owner.
    owner_div = soup.select_one('#text-container a.yt-simple-endpoint.style-scope.yt-formatted-string')
    owner_list = owner_div and owner_div.text

    # date of the post.
    date_div = soup.select_one('#info #info-strings')
    date_list = date_div and date_div.text

    # date of the post.
    comment_date_div = soup.select('#header-author a.yt-simple-endpoint.style-scope.yt-formatted-string')
    comment_date_list = [cm.text for cm in comment_date_div]
                                                                                

    # fecha de scrapping
    today = date.today()

    dataframe = pd.DataFrame(
    {'TITULO': title,
     'COMENTARIO': comment_list,
     'AUTOR': author_list,
     'LIKE': vote_list,
     'PUBLICADO_POR': owner_list,
     'FECHA_POST': date_list,
     'FECHA_COMENTARIO': comment_date_list
    })

    dataframe['LIKE'] = dataframe['LIKE'].astype('int')
    print(type(dataframe['LIKE']))
    # ELIMIANNDO ESPACIONES ANTES Y DESPUES DE LOS TEXTOS
    dataframe['TITULO'] = dataframe['TITULO'].str.strip()
    dataframe['PUBLICADO_POR'] = dataframe['PUBLICADO_POR'].str.strip()
    dataframe['FECHA_POST'] = dataframe['FECHA_POST'].str.strip()

    # LIMPIENZA FECHA DE COMENTARIO
    try:
        dataframe['FECHA_COMENTARIO'] = dataframe['FECHA_COMENTARIO'].apply(limpieza_fecha_comentario)
        dataframe['FECHA_COMENTARIO'] = dataframe['FECHA_COMENTARIO'].str.strip()
    except AttributeError: 
        st.write('EL Post no Cuenta con comentarios, por favor intentar con otro.')

    dataframe['FECHA_EXTRACCION'] = today

    try:
        dataframe['COMENTARIO'] = dataframe['COMENTARIO'].apply(text_clean)
        dataframe['COMENTARIO'] = dataframe['COMENTARIO'].apply(remove_emoji)
        dataframe['COMENTARIO'] = dataframe['COMENTARIO'].str.strip()
        dataframe['CARACTERES'] = dataframe['COMENTARIO'].apply(lambda x: len(x))
        dataframe['CARACTERES'] = dataframe['CARACTERES'].astype('int')

        dataframe['TEXTO_STOPWORD'] = dataframe['COMENTARIO']
        dataframe['TEXTO_STOPWORD'] = dataframe['TEXTO_STOPWORD'].apply(removeStopwords)
        dataframe['TEXTO_STOPWORD'] = dataframe['TEXTO_STOPWORD'].str.strip()
    except AttributeError:
        st.write('EL Post no Cuenta con comentarios, por favor intentar con otro.')

    try:
        # LIMPIANDO AUTOR
        dataframe['AUTOR'] = dataframe['AUTOR'].apply(text_clean)
        dataframe['AUTOR'] = dataframe['AUTOR'].apply(remove_emoji)
        dataframe['AUTOR'] = dataframe['AUTOR'].str.lstrip()


        # ELIMINANDO COMENTARIOS EN BLANCO
        dataframe['LEN'] = dataframe['COMENTARIO'].str.len()
        dataframe = dataframe[dataframe['LEN']>=1]
        dataframe.drop(columns=['LEN'], inplace=True)
    except AttributeError:
        st.write('EL Post no Cuenta con comentarios, por favor intentar con otro.')

    # exportación a excel
    datestring = date.today().strftime('%Y-%m-%d')
    clean_url = re.sub(r'https://www.youtube.com/watch', '', url)
    clean_url = clean_url[3:]

    # llave video
    dataframe['LLAVE_VIDEO'] = clean_url
    cols = list(dataframe.columns)
    cols = [cols[-1]] + cols[:-1]
    dataframe = dataframe[cols]

    # POLARIDAD Y SENTIMIENTO
    try:
        dataframe['POLARIDAD'] = dataframe['COMENTARIO'].apply(spanish_sentiment)
        dataframe['POLARIDAD'] = dataframe['POLARIDAD'].astype('float')
        dataframe['SENTIMIENTO'] = dataframe['POLARIDAD'].apply(sentimiento)
       

        positivo = dataframe[dataframe['SENTIMIENTO']=='POSITIVO']
        negativo = dataframe[dataframe['SENTIMIENTO']=='NEGATIVO']
        neutral = dataframe[dataframe['SENTIMIENTO']=='NEUTRAL']
    except AttributeError:
        st.write('EL Post no Cuenta con comentarios, por favor intentar con otro.')

    try:
        dataframe['NUMERIC_SENTIMENT'] = dataframe['POLARIDAD'].apply(getSentiment)
    except AttributeError:
        st.write('EL Post no Cuenta con comentarios, por favor intentar con otro.')

    dataframe.to_excel('D:\\ComentarioYoutube\\{0}'.format('comentarios_youtube' +'_'+clean_url +'_'+datestring + '.xlsx'), index=False)
    file_size = os.stat('D:\\ComentarioYoutube\\{0}'.format('comentarios_youtube' +'_'+clean_url +'_'+datestring + '.xlsx'))
    print("Size of file :", file_size.st_size, "bytes")
    return dataframe, positivo, neutral, negativo


# Extrayendo datos segregados por sentimiento y transofrmandolas a variables de entorno.
DATA, POSITIVO, NEUTRAL, NEGATIVO = ScrapComment(url=url)


def streamlitWebAPP(dataframe, positivo, negativo, neutral):
    st.write('')
    st.markdown("<h2 style='text-align: center; color: red;'>TABLA DE COMENTARIOS</h1>", unsafe_allow_html=True)
    try:
        # Titulo de video
        titulo_video = dataframe['TITULO']
        titulo_video = pd.DataFrame(titulo_video)
        titulo_video = titulo_video.iloc[0]['TITULO']
        st.subheader('TITULO DEL VIDEO')
        st.write(titulo_video)
    except IndexError:
        st.write('EL Post no Cuenta con comentarios, por favor intentar con otro.')
    # ----------------------------------------------------------------------------------------------------------------------------
    # Tabla de todos los datos
    try:
        data_table = dataframe.drop(columns=['LLAVE_VIDEO', 'TITULO'])
        st.dataframe(data_table.head(15),10000,10000)
        st.write('Número de Comentarios extraidos: ', len(data_table))
    except ValueError:
            st.write('No existen Comentarios')
    except AttributeError:
            st.write('No existen comenatarios')

    # Tabla de todos los datos positivos
    try:
        positivo_table = positivo.drop(columns=['LLAVE_VIDEO', 'TITULO'])
        st.dataframe(positivo_table.head(15),10000,10000)
        st.write('Número de Comentarios positivos extraidos: ', len(positivo_table))
    except ValueError:
            st.write('No existen Comentarios positivos')
    except AttributeError:
            st.write('No existen comenatarios positivos')

      # Tabla de todos los datos negativos
    try:
        negativo_table = negativo.drop(columns=['LLAVE_VIDEO', 'TITULO'])
        st.dataframe(negativo_table.head(15),10000,10000)
        st.write('Número de Comentarios negativo extraidos: ', len(negativo_table))
    except ValueError:
            st.write('No existen Comentarios negativos')
    except AttributeError:
            st.write('No existen comenatarios negativos')

       # Tabla de todos los datos neutrakes
    try:
        neutral_table = neutral.drop(columns=['LLAVE_VIDEO', 'TITULO'])
        st.dataframe(neutral_table.head(15),10000,10000)
        st.write('Número de Comentarios neutrales extraidos: ', len(neutral_table))
    except ValueError:
            st.write('No existen Comentarios neutrales')
    except AttributeError:
            st.write('No existen comenatarios neutrales')
    
    # ----------------------------------------------------------------------------------------------------------------------------
    try:
        st.write('Comentario con Más Likes')
        st.write(max(dataframe.LIKE))
    except ValueError:
            st.write('Ningún comentario tienen Like')
    except AttributeError:
            st.write('Ningún Comentario tiene  Like')

    # ----------------------------------------------------------------------------------------------------------------------------
    # Habilitar botón para descargar el archivo
    df_xlsx = to_excel(dataframe)
    # exportación a excel
    datestring = date.today().strftime('%Y-%m-%d')
    clean_url = re.sub(r'https://www.youtube.com/watch', '', url)
    clean_url = clean_url[3:]
    st.download_button(label='📥 DESCARGAR ARCHIVO COMPLETO',
                                data=df_xlsx ,
                                file_name= 'comentarios_youtube' +'_'+clean_url +'_'+datestring + '.xlsx')
    # ----------------------------------------------------------------------------------------------------------------------------
    # # show statistics on the data
    st.write(dataframe.describe())
        
    # ----------------------------------------------------------------------------------------------------------------------------
    st.markdown("<h2 style='text-align: center; color: red;'>Comentarios que contienen 'Me Gusta'</h1>", unsafe_allow_html=True)
    #Bar Chart
    st.bar_chart(data=dataframe['LIKE'])
    # ----------------------------------------------------------------------------------------------------------------------------
    # Likes por comentario
    like_barplot = sns.barplot(x="AUTOR", y="LIKE", data=data_table, palette='bright')
    plt.title('LIKES', color='black')
    plt.xlabel('Usuario Youtube')
    plt.xticks(rotation=90)
    st.pyplot()

    # ----------------------------------------------------------------------------------------------------------------------------
    st.markdown("<h2 style='text-align: center; color: red;'>Algoritmos para Análisis de Sentimientos en los Datos</h1>", unsafe_allow_html=True)
    st.subheader('Distribución de sentimientos entre los comentarios extraidos')
    color_sentimiento = ['red', 'green', 'gray']
    sns.countplot(x='SENTIMIENTO',data=data_table)
    plt.title('Sentimientos de Comentarios', color='black', size=18)
    st.pyplot()
    # ----------------------------------------------------------------------------------------------------------------------------
    st.subheader('Usuarios con más Comentarios')
    st.write('Usuarios con Para todos los Comentarios')
    popular_users(data=data_table)
    st.write('Usuarios con Para todos los Comentarios Positivos')
    popular_users(data=positivo_table)
    st.write('Usuarios con Para todos los Comentarios Negativos')
    popular_users(data=negativo_table)
    st.write('Usuarios con Para todos los Comentarios Neutrales')
    popular_users(data=neutral_table)
    # ----------------------------------------------------------------------------------------------------------------------------
    st.markdown("<h2 style='text-align: center; color: red;'>Word Cloud- Nube de Palabras</h1>", unsafe_allow_html=True)
    st.subheader('Palabras que más se repiten en los comentarios')
    # Generador de WordCloud
    wordCloudGenerator(data=data_table, background_color='navy')
    wordCloudGenerator(data=positivo_table, background_color='darkgreen')
    wordCloudGenerator(data=negativo_table, background_color='darkred')
    wordCloudGenerator(data=neutral_table, background_color='gray')
    # ----------------------------------------------------------------------------------------------------------------------------
    st.markdown("<h2 style='text-align: center; color: red;'>Algoritmo N-Gram</h1>", unsafe_allow_html=True)
    st.subheader('Términos Más relevantes extraidos de los textos en los comentarios')
    st.write('Terminos Más importantes ségun Algoritmo N-GRAM - Todos los Comentarios')
    plot_text_ngrams(df=data_table)
    st.write('Terminos Más importantes ségun Algoritmo N-GRAM - Comentarios Positivos')
    plot_text_ngrams(df=positivo_table)
    st.write('Terminos Más importantes ségun Algoritmo N-GRAM - Comentarios Negativos')
    plot_text_ngrams(df=negativo_table)
    st.write('Terminos Más importantes ségun Algoritmo N-GRAM - Comentarios Neutros')
    plot_text_ngrams(df=neutral_table)
    # -----------------------------------------------------------------------------------------------------------
    st.markdown("<h2 style='text-align: center; color: red;'>Algoritmo TF-IDF</h1>", unsafe_allow_html=True)
    st.subheader('Vocabulario de Terminos Extraidos')
    termFrequencyVocab(data=data_table, tipo_sentimiento='totales')
    termFrequencyVocab(data=positivo_table, tipo_sentimiento='positivo')
    termFrequencyVocab(data=negativo, tipo_sentimiento='negativo')
    termFrequencyVocab(data=neutral, tipo_sentimiento='neutrales')
    # -----------------------------------------------------------------------------------------------------------
    st.subheader('Distribución de Caracteres por Cada autor de Comentario')
    st.write('Distribución de Para Todos los comentarios')
    text_dist(data=data_table)
    st.write('Distribución de Para Todos los comentarios Positivos')
    text_dist(data=positivo_table)
    st.write('Distribución de Para Todos los comentarios Negativos')
    text_dist(data=negativo)
    st.write('Distribución de Para Todos los comentarios Neutrales')
    text_dist(data=neutral)
    #-------------------------------------------------------------------------------------------------------------------

    st.success('¡Análisis Finalizado con exito Teler!')
    st.balloons()

if __name__ == "__main__":
    ScrapComment(url=url)
    streamlitWebAPP(dataframe=DATA, positivo=POSITIVO, negativo=NEGATIVO, neutral=NEUTRAL)

############### TIEMPO DE EJECUCIÓN TOTAL DEL PROGRAMA ######################

print('')
end_time = time.time()
print(f'TIEMPO DE EJECUCIÓN TOTAL DEL PROGRAMA: {end_time - start_time} segundos.')
