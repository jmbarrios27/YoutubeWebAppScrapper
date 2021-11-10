# LIBRERIAS DE YOUTUBE
from datetime import date
from sys import path
from selenium import webdriver
from webdriver_manager.firefox import GeckoDriverManager
from bs4 import BeautifulSoup
import time
import pandas as pd
import os
import warnings
import re
import string
# ALGORITMO DE CLASIFICACIÓN DE SENTIMIENTO
from sentiment_analysis_spanish import sentiment_analysis
from wordcloud import WordCloud
from nltk.corpus import stopwords
from textblob import TextBlob
import gensim
import itertools,collections
import nltk

# WEB APP
import streamlit as st
import urllib.request as url


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


# PIDE EL link
#urls = input('INGRESE EL LINK DEL VIDEO QUE DESEA EXTRAER COMENTARIOS:')

def ScrapComment(url):
    # so we can see the output side by side
    st.set_page_config(layout="wide")

# i made these just to hold the test_input box so
# the rest of the output can match up below and we can
# compare easier
    col1,col2 = st.columns(2)
    with col2:
        url = st.text_input('URL link to scrape')

        st.write('the link:')
        st.write(url)

    st.write('the link:')
    st.write(url)
    url = url
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

    # ELIMIANNDO ESPACIONES ANTES Y DESPUES DE LOS TEXTOS
    dataframe['TITULO'] = dataframe['TITULO'].str.strip()
    dataframe['PUBLICADO_POR'] = dataframe['PUBLICADO_POR'].str.strip()
    dataframe['FECHA_POST'] = dataframe['FECHA_POST'].str.strip()

    # LIMPIENZA FECHA DE COMENTARIO
    dataframe['FECHA_COMENTARIO'] = dataframe['FECHA_COMENTARIO'].apply(limpieza_fecha_comentario)
    dataframe['FECHA_COMENTARIO'] = dataframe['FECHA_COMENTARIO'].str.strip()

    dataframe['FECHA_EXTRACCION'] = today

    dataframe['COMENTARIO'] = dataframe['COMENTARIO'].apply(text_clean)
    dataframe['COMENTARIO'] = dataframe['COMENTARIO'].apply(remove_emoji)
    dataframe['COMENTARIO'] = dataframe['COMENTARIO'].str.strip()

    dataframe['TEXTO_STOPWORD'] = dataframe['COMENTARIO']
    dataframe['TEXTO_STOPWORD'] = dataframe['TEXTO_STOPWORD'].apply(removeStopwords)
    dataframe['TEXTO_STOPWORD'] = dataframe['TEXTO_STOPWORD'].str.strip()

    # LIMPIANDO AUTOR
    dataframe['AUTOR'] = dataframe['AUTOR'].apply(text_clean)
    dataframe['AUTOR'] = dataframe['AUTOR'].apply(remove_emoji)
    dataframe['AUTOR'] = dataframe['AUTOR'].str.lstrip()


    # ELIMINANDO COMENTARIOS EN BLANCO
    dataframe['LEN'] = dataframe['COMENTARIO'].str.len()
    dataframe = dataframe[dataframe['LEN']>=1]
    dataframe.drop(columns=['LEN'], inplace=True)

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
    dataframe['POLARIDAD'] = dataframe['COMENTARIO'].apply(spanish_sentiment)
    dataframe['SENTIMIENTO'] = dataframe['POLARIDAD'].apply(sentimiento)

    positivo = dataframe[dataframe['SENTIMIENTO']=='POSITIVO']
    negativo = dataframe[dataframe['SENTIMIENTO']=='NEGATIVO']
    neutral = dataframe[dataframe['SENTIMIENTO']=='NEUTRAL']

    dataframe.to_excel('D:\\ComentarioYoutube\\{0}'.format('comentarios_youtube' +'_'+clean_url +'_'+datestring + '.xlsx'), index=False)
    file_size = os.stat('D:\\ComentarioYoutube\\{0}'.format('comentarios_youtube' +'_'+clean_url +'_'+datestring + '.xlsx'))
    print("Size of file :", file_size.st_size, "bytes")
    return dataframe, positivo, neutral, negativo, url


# Extrayendo datos segregados por sentimiento y transofrmandolas a variables de entorno.
DATA, POSITIVO, NEUTRAL, NEGATIVO ,urls= ScrapComment()

##############################################################c###############################################################################

def streamlitWebAPP(dataframe):
    st.dataframe(dataframe,1000,1000)

if __name__ == "__main__":
    ScrapComment(url=urls)
    streamlitWebAPP(dataframe=DATA)

############### TIEMPO DE EJECUCIÓN TOTAL DEL PROGRAMA ######################
print('')
end_time = time.time()
print(f'TIEMPO DE EJECUCIÓN TOTAL DEL PROGRAMA: {end_time - start_time} segundos.')