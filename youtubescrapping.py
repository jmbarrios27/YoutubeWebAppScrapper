# LIBRERIAS DE YOUTUBE
from datetime import date
from sys import path
from nltk import data
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

# VISUALIZACIÓN DE DATOS
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import seaborn as sns
import plotly.figure_factory as ff
import plotly.express as px
import plotly.graph_objs as go

# WEB APP
import streamlit as st
import urllib.request as url
from io import BytesIO
from PIL import Image
import urllib.request as url
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


def conteo_palabras(dataframe):
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

    fig_conteo_palabras, ax = plt.subplots(figsize=(12, 12))

    #Plot para ver conteo de palabras más utilizadas en textos en español
    final_word_count_es.sort_values(by='conteo').plot.barh(x='palabras',
                        y='conteo',
                        ax=ax,
                        color="#33A1FF")

    ax.set_title("CONTEO DE TERMINOS")
    plt.figure(figsize=(40,40))
    st.pyplot(fig_conteo_palabras)


############################# STREAMLIT ###############################3

st.title("YOUTUBE SCRAPPER - TELERED")
# Input
url = input('Ingrese url:')


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
    dataframe['POLARIDAD'] = dataframe['POLARIDAD'].astype('float')
    dataframe['SENTIMIENTO'] = dataframe['POLARIDAD'].apply(sentimiento)

    positivo = dataframe[dataframe['SENTIMIENTO']=='POSITIVO']
    negativo = dataframe[dataframe['SENTIMIENTO']=='NEGATIVO']
    neutral = dataframe[dataframe['SENTIMIENTO']=='NEUTRAL']

    dataframe.to_excel('D:\\ComentarioYoutube\\{0}'.format('comentarios_youtube' +'_'+clean_url +'_'+datestring + '.xlsx'), index=False)
    file_size = os.stat('D:\\ComentarioYoutube\\{0}'.format('comentarios_youtube' +'_'+clean_url +'_'+datestring + '.xlsx'))
    print("Size of file :", file_size.st_size, "bytes")
    return dataframe, positivo, neutral, negativo


# Extrayendo datos segregados por sentimiento y transofrmandolas a variables de entorno.
DATA, POSITIVO, NEUTRAL, NEGATIVO = ScrapComment(url=url)


def streamlitWebAPP(dataframe, positivo, negativo):
    # Tabla
    data_table = dataframe.drop(columns=['LLAVE_VIDEO', 'TITULO'])
    st.dataframe(data_table,10000,10000)
    
    # ----------------------------------------------------------------------------------------------------------------------------
    # Habilitar botón para descargar el archivo
    df_xlsx = to_excel(dataframe)
    # exportación a excel
    datestring = date.today().strftime('%Y-%m-%d')
    clean_url = re.sub(r'https://www.youtube.com/watch', '', url)
    clean_url = clean_url[3:]
    st.download_button(label='📥 DESCARGAR ARCHIVO',
                                data=df_xlsx ,
                                file_name= 'comentarios_youtube' +'_'+clean_url +'_'+datestring + '.xlsx')
    # ----------------------------------------------------------------------------------------------------------------------------
    # # show statistics on the data
    st.write(dataframe.describe())
        
    # ----------------------------------------------------------------------------------------------------------------------------
    st.subheader('NÚMERO DE LIKES DE COMENTARIOS')
    #Bar Chart
    st.bar_chart(data=dataframe['LIKE'])
    # ----------------------------------------------------------------------------------------------------------------------------
   
    # ----------------------------------------------------------------------------------------------------------------------------
    # Wordcloud global 
    allwords = ' '.join([fk for fk in dataframe.TEXTO_STOPWORD])
    wordcloud = WordCloud(width=600, height=300, random_state=22, max_font_size=119, background_color='white').generate(allwords)
    fig = plt.figure(figsize=(12,10))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')

     # Wordcloud positivo
    allwords = ' '.join([fk for fk in positivo.TEXTO_STOPWORD])
    wordcloud = WordCloud(width=600, height=300, random_state=22, max_font_size=119, background_color='green').generate(allwords)
    fig_positivo = plt.figure(figsize=(12,10))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')

     # Wordcloud negativo
    allwords = ' '.join([fk for fk in negativo.TEXTO_STOPWORD])
    wordcloud = WordCloud(width=600, height=300, random_state=22, max_font_size=119, background_color='red').generate(allwords)
    fig_negativo = plt.figure(figsize=(12,10))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')

    # Por columna
    col1, col2, col3 = st.columns(3)
    with col1:
        st.header("NUBE 1")
        st.pyplot(fig, use_column_width=True)
    with col2:
        st.header("NUBE 2")
        st.pyplot(fig_positivo, use_column_width=True)
    with col3:
        st.header("NUBE 3")
        st.pyplot(fig_negativo, use_column_width=True)

    # ----------------------------------------------------------------------------------------------------------------------------
    # conteo de palabras
    st.subheader('NÚMERO DE PALABRAS PARA TODOS LOS COMENTARIOS')
    conteo_palabras(dataframe=dataframe)
    st.subheader('NÚMERO DE PALABRAS PARA TODOS LOS COMENTARIOS POSITIVOS')
    conteo_palabras(dataframe=positivo)
    st.subheader('NÚMERO DE PALABRAS PARA TODOS LOS COMENTARIOS NEGATIVOS')
    conteo_palabras(dataframe=negativo)
    # -----------------------------------------------------------------------------------------------------------

    
if __name__ == "__main__":
    ScrapComment(url=url)
    streamlitWebAPP(dataframe=DATA, positivo=POSITIVO, negativo=NEGATIVO)

############### TIEMPO DE EJECUCIÓN TOTAL DEL PROGRAMA ######################
print('')
end_time = time.time()
print(f'TIEMPO DE EJECUCIÓN TOTAL DEL PROGRAMA: {end_time - start_time} segundos.')
