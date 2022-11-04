############################################################## All Packages 
import requests     # allows to send HTTP requests
from bs4 import BeautifulSoup     # for pulling data out of HTML and XML files
import re           #regular expression package
#from wordcloud import WordCloud
#import matplotlib.pyplot as plt     #for plotting..To save the wordcloud into a file, matplotlib can also be installed
import nltk          #natural language tool kit package for natural language processing
#from nltk.corpus import stopwords   
#from sklearn.feature_extraction.text import TfidfVectorizer
#from wordcloud import STOPWORDS
#from sklearn.feature_extraction.text import CountVectorizer    # Using count vectoriser to view the frequency of bigrams
#conda install -c conda-forge wordcloud
#from pprint import pprint
#import urllib.request
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')    
#pip install textblob
#from textblob import TextBlob
from nltk.corpus import wordnet                   #for parts of speech
import pandas as pd
nltk.download('omw-1.4')               #Resource omw-1.4 not found. That's why i downloaded it.
nltk.download('averaged_perceptron_tagger')      #Resource averaged_perceptron_tagger not found.
#import csv

###################################################################
#NOTE: Run stopwords, Positive words, Negative words  only once..... in the next time if needed
###############################################################################################
###############################################################################################
###############################################################################################
#########Stop words
with open("C:/Users/yamini/Desktop/Text Analysis/Stopwords/StopWords_Auditor.txt","r") as sw:
    stop_words1 = sw.read()
type(stop_words1)                #stopwords are not in list format

#function to convert stopwords into list
def stop(i):
    print(i)
    x=i.split()
    return x

with open("C:/Users/yamini/Desktop/Text Analysis/Stopwords/StopWords_Auditor.txt","r") as sw:
    stop_words1 = sw.read()    #taking stopwords from the text file
stop_words1=stop(stop_words1)
print(stop_words1)
type(stop_words1)

with open("C:/Users/yamini/Desktop/Text Analysis/Stopwords/StopWords_Currencies.txt","r") as sw:
    stop_words2 = sw.read()    #taking stopwords from the text file
stop_words2=stop(stop_words2)
print(stop_words2)
type(stop_words2)

with open("C:/Users/yamini/Desktop/Text Analysis/Stopwords/StopWords_DatesandNumbers.txt","r") as sw:
    stop_words3 = sw.read()    #taking stopwords from the text file
stop_words3=stop(stop_words3)
print(stop_words3)
type(stop_words3)

with open("C:/Users/yamini/Desktop/Text Analysis/Stopwords/StopWords_Generic.txt","r") as sw:
    stop_words4 = sw.read()    #taking stopwords from the text file
stop_words4=stop(stop_words4)
print(stop_words4)
type(stop_words4)

with open("C:/Users/yamini/Desktop/Text Analysis/Stopwords/StopWords_GenericLong.txt","r") as sw:
    stop_words5 = sw.read()    #taking stopwords from the text file
stop_words5=stop(stop_words5)
print(stop_words5)
type(stop_words5)

with open("C:/Users/yamini/Desktop/Text Analysis/Stopwords/StopWords_Geographic.txt","r") as sw:
    stop_words6 = sw.read()    #taking stopwords from the text file
stop_words6=stop(stop_words6)
print(stop_words6)
type(stop_words6)

with open("C:/Users/yamini/Desktop/Text Analysis/Stopwords/StopWords_Names.txt","r") as sw:
    stop_words7 = sw.read()    #taking stopwords from the text file
stop_words7=stop(stop_words7)
print(stop_words7)
type(stop_words7)

stop_words=stop_words1 + stop_words2 + stop_words3 + stop_words4 + stop_words5 + stop_words6 + stop_words7
len(stop_words1 + stop_words2 + stop_words3 + stop_words4 + stop_words5 + stop_words6 + stop_words7)
len(stop_words)

######## Positive words
with open("C:/Users/yamini/Desktop/Text Analysis/positive-words.txt","r") as pos:
  poswords = pos.read().split("\n") 
poswords

####### Negative words
with open("C:/Users/yamini/Desktop/Text Analysis/negative-words.txt","r") as neg:
  negwords = neg.read().split("\n")
negwords


###############################################################################################
###############################################################################################
###############################################################################################


data=pd.read_excel("C:/Users/yamini/Desktop/Text Analysis/Input.xlsx") 
data
type(data)
data.head()
data.columns
links_url=data["URL"]
links_url
type(links_url)


def parse_product(url):
    headers = {
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64; rv:60.0) Gecko/20100101 Firefox/60.0"
    }    
    page = requests.get(url, headers=headers)
    soup = BeautifulSoup(page.content, 'html.parser')
    arti_title=soup.find_all('title')
    arti_text=soup.find_all('p')
    
    try:
        title=soup.find_all('title')   
    except AttributeError as err:
        title= 'None'
     
    try:
        paragraph=soup.find_all('p')
    except AttributeError as err:
        title= "None"
###############################################################################################
###############################################################################################
###############################################################################################
    #Converting article text into string data type
    data1 = []
    for x in arti_text:
        data1.append(str(x))
    type(data1)
    
    data2=''.join(data1)
    type(data2)                                 #take data2
    
    #Converting article title into string data type
    data3 = []
    for x in arti_title:
        data3.append(str(x))
    type(data3)
    
    data4=''.join(data3)
    type(data4)                                 #take data4
    
    #removing html tags
    regex = re.compile(r'<[^>]+>')
    def remove_html(string):
        return regex.sub('', string)
    
    data5=remove_html(data2)
    data5                                      #take data5
    data6=remove_html(data4)
    data6                                       #take data6       
    
    #joining 2 strings(article text + article title)
    data7=data6+data5
    data7      

    ######################################################## Data cleaning 
    #removing special characters from a string
    data8=re.sub(r"[^a-zA-Z0-9 ]", "", data7)
    data8
    
    #converting into lowercase
    data9=data8.lower()
    data9

    ######################################################## Lemmatization
    # Init the Wordnet Lemmatizer
    lemmatizer = WordNetLemmatizer()
    #print(lemmatizer.lemmatize("bats"))
    #print(data10)
    
    # Lemmatize list of words and join
    #lemmatized_output = ' '.join([lemmatizer.lemmatize(w) for w in data10])
    #print(lemmatized_output)           #still not getting the correct output.
    
    #parts of speech tagging
    #print(nltk.pos_tag(['feet']))
    
    def get_wordnet_pos(word):
        """Map POS tag to first character lemmatize() accepts"""
        tag = nltk.pos_tag([word])[0][1][0].upper()
        tag_dict = {"J": wordnet.ADJ,
                    "N": wordnet.NOUN,
                    "V": wordnet.VERB,
                    "R": wordnet.ADV}
        return tag_dict.get(tag, wordnet.NOUN)
    #word = 'said'
    #print(lemmatizer.lemmatize(word, get_wordnet_pos(word)))
    
    data10=[lemmatizer.lemmatize(w, get_wordnet_pos(w)) for w in nltk.word_tokenize(data9)]
    data10

    ####################################################### stopword removal
    data11 = [w for w in data10 if not w in stop_words]
    len(data10)              #1750
    total_words_after_cleaning=len(data11)              #1003      nearly 750 stopwords are removed
    total_words_after_cleaning
    data12 = " ".join(data11)   #converting into string to do tokenization
    data12
    
    ########################################################### tokenization     (total number of words)
    data13 = word_tokenize(data12)
    type(data13)                #990
    len(data13)
    
    ###############################################################################################
    ######################################## Positive Score #######################################
    df = pd.DataFrame(data13)      #created dataframe
    print(df)
    df.columns = ['preprocess_txt']           # created column
    print(df)
    #df['total_len'] = df['preprocess_txt'].map(lambda x: len(x))     #number of letters
    num_pos = [w for w in df["preprocess_txt"] if w in poswords]
    num_pos
    total_pos_score=len(num_pos)                             #73
    total_pos_score
    
    ######################################## Negative Score ###############################
    num_neg = [w for w in df["preprocess_txt"] if w in negwords]
    num_neg
    total_neg_score=len(num_neg)                             #33
    total_neg_score
    
    ######################################## Polarity Score ################################
    #Polarity_Score = (pos_score – neg_score)/ ((pos_score + neg_score) + 0.000001)
    pos_score=len(num_pos)
    neg_score=len(num_neg)
    a=pos_score + neg_score
    s=pos_score - neg_score
    
    Polarity_Score=s/(a+0.000001)
    Polarity_Score                          #0.377358487006052
    
    ####################################### Subjectivity Score ##############################
    #Subjectivity Score = (Positive Score + Negative Score)/ ((Total Words after cleaning) + 0.000001)
    Subjectivity_Score=a/(total_words_after_cleaning+0.000001)
    Subjectivity_Score                      #0.10707070696255484
    
    ##################################### Average Sentence Length ################################
    #Average Sentence Length = the number of words / the number of sentences
    #sentences = data7.split(".") #split the text into a list of sentences.
    #len(sentences)
    #words = data7.split(" ") #split the input text into a list of separate words
    #len(words)
    def avg_sentence_len(text):
      sentences = text.split(".") #split the text into a list of sentences.
      words = text.split(" ") #split the input text into a list of separate words
      if(sentences[len(sentences)-1]==""): #if the last value in sentences is an empty string
        average_sentence_length = len(words) / len(sentences)-1
      else:
        average_sentence_length = len(words) / len(sentences)
      return average_sentence_length 
      
    Average_Sentence_Length = avg_sentence_len(data7) 
    Average_Sentence_Length                   #21.74025974025974
    
    ################################### Total Number of Sentences ###############################
    sentences = data7.split(".") #split the text into a list of sentences.
    total_sen=len(sentences)
    total_sen
    
    ################################## Complex Word Count  ##############################
    def count_syllables(word):    
            a= re.findall('(?!e$)[aeiouy]+', word, re.I) + re.findall('^[^aeiouy]*e$', word, re.I)
            return a
    cwc=count_syllables(data12)
    cwc
    type(cwc)
    
    def my_func(sentence):
        retList = []
        for x in sentence:
            if len(x) >= 3:
                retList.append(x)
        return retList
    
    complex_word_count=my_func(cwc)
    complex_word_count
    num_complex_words=len(complex_word_count)
    num_complex_words                                             #7
    
    ################################## Word Count ##############################
    #did stopword removal, punchuation, lemmatization with parts of speech tagging
    data13
    number_of_words=len(data13)                         #990
    number_of_words

    ################################## Average Number of Words Per Sentence   ##############################
    #Average Number of Words Per Sentence = the total number of words / the total number of sentences
    Avg_num_words_per_sentence= number_of_words/total_sen
    Avg_num_words_per_sentence                            #12.857142857142858
    
    ################################## Percentage of Complex words  ##############################
    #Percentage of Complex words = (the number of complex words / the number of words)*100
    Percentage_of_Complex_words = (num_complex_words/number_of_words)*100
    Percentage_of_Complex_words                #0.7070707070707071.........7%
    
    ################################## Fog Index   ##############################
    #Fog Index=0.4 * (Average Sentence Length + Percentage of Complex words)
    Fog_Index=0.4 *(Average_Sentence_Length+Percentage_of_Complex_words)
    Fog_Index                                             #8.97893217893218
    
    ################################## SYLLABLE PER WORD  ##############################
    #for this we have to take lemmatized with pos tagging data to avoid 'ed', 'es'
    VOWEL_RUNS = re.compile("[aeiouy]+", flags=re.I)
    EXCEPTIONS = re.compile(
        # fixes trailing e issues:
        # smite, scared
        "[^aeiou]e[sd]?$|"
        # fixes adverbs:
        # nicely
        + "[^e]ely$",
        flags=re.I
    )
    ADDITIONAL = re.compile(
        # fixes incorrect subtractions from exceptions:
        # smile, scarred, raises, fated
        "[^aeioulr][lr]e[sd]?$|[csgz]es$|[td]ed$|"
        # fixes miscellaneous issues:
        # flying, piano, video, prism, fire, evaluate
        + ".y[aeiou]|ia(?!n$)|eo|ism$|[^aeiou]ire$|[^gq]ua",
        flags=re.I
    )
    
    def count_syllables(word):
        vowel_runs = len(VOWEL_RUNS.findall(word))
        exceptions = len(EXCEPTIONS.findall(word))
        additional = len(ADDITIONAL.findall(word))
        return max(1, vowel_runs - exceptions + additional)
    
    SYLLABLE_PER_WORD=count_syllables(data12)                                        #2608
    SYLLABLE_PER_WORD
    ##average_syllables_per_word = total_syllables / total_words
    
    ################################## AVG WORD LENGTH  ##############################
    #Sum of the total number of characters in each word/Total number of words
    total_num_char=sum(len(i) for i in data13)
    total_num_char
    
    avg_word_length= total_num_char  / number_of_words
    avg_word_length                         #7.2555555555555555
    
    ################################## PERSONAL PRONOUNS  ##############################
    #I am taking data8 as i still not done lemmatization,lowercase, stopwords...So that US also present in the data.
    data8
    #“I,” “we,” “my,” “ours,” and “us”. 
    count_I = len(re.findall(r'(?<!\S)'+ "I" + r'(?!\S)', data8))
    count_I
    count_we = len(re.findall(r'(?<!\S)'+ "we" + r'(?!\S)', data8))
    count_we
    count_my = len(re.findall(r'(?<!\S)'+ "my" + r'(?!\S)', data8))
    count_my
    count_ours = len(re.findall(r'(?<!\S)'+ "ours" + r'(?!\S)', data8))
    count_ours
    count_us = len(re.findall(r'(?<!\S)'+ "us" + r'(?!\S)', data8))
    count_us           #i didnt mention re.IGNORECASE..So that it wont take country name "US".
    
    Personal_pronouns=count_I + count_we + count_my + count_ours + count_us
    Personal_pronouns

########################################################################################### 
###########################################################################################
###########################################################################################   
    product={
        'title' : arti_title,
        'paragraph' : arti_text,
        'POSITIVE SCORE' : total_pos_score,
        'NEGATIVE SCORE' : total_neg_score, 
        'POLARITY SCORE' : Polarity_Score,
        'SUBJECTIVITY SCORE' : Subjectivity_Score,
        'AVG SENTENCE LENGTH' : Average_Sentence_Length,
        'PERCENTAGE OF COMPLEX WORDS' : Percentage_of_Complex_words,
        'FOG INDEX': Fog_Index,
        'AVG NUMBER OF WORDS PER SENTENCE': Avg_num_words_per_sentence,
        'COMPLEX WORD COUNT' : num_complex_words,
        'WORD COUNT' : number_of_words,
        'SYLLABLE PER WORD' : SYLLABLE_PER_WORD,
        'PERSONAL PRONOUNS' : Personal_pronouns,
        'AVG WORD LENGTH' : avg_word_length
        }
    return product
    
    
def main():
    results=[]
    urls=links_url
    for url in urls:
            results.append(parse_product(url))             
    df=pd.DataFrame(results)
    #print(df.head(5))
    print(df.columns)
    print(len(df.index))
    print(df)
    df.to_csv('textanalysis.csv')
               
    #final output
    data2=pd.read_csv("C:/Users/yamini/Desktop/Text Analysis/textanalysis.csv")
    data2                
    data2.columns
    
    data1=pd.read_excel("C:/Users/yamini/Desktop/Text Analysis/Input.xlsx")
    data1.head(5)
    
    result=pd.concat([data1,data2],axis=1)
    result.columns          #For the 3 links there is empty list the page of those links are empty and getting 404 error.
    # So it doesnot scrape any data as it doesnot contain title or paragraph. And gave the variable details
    #according to that
    
    #Instead of droping,I am taking column names
    result1= result[['URL_ID', 'URL','POSITIVE SCORE', 'NEGATIVE SCORE', 'POLARITY SCORE',
           'SUBJECTIVITY SCORE', 'AVG SENTENCE LENGTH',
           'PERCENTAGE OF COMPLEX WORDS', 'FOG INDEX',
           'AVG NUMBER OF WORDS PER SENTENCE', 'COMPLEX WORD COUNT', 'WORD COUNT',
           'SYLLABLE PER WORD', 'PERSONAL PRONOUNS', 'AVG WORD LENGTH']]
    result1.columns 
    result1.to_csv('final_Output .csv')

    #Getting url id, links,title, paragraph in seperate csv file..
    all_text_data=result[['URL_ID', 'URL','title', 'paragraph']]
    all_text_data.to_csv("url_links_text.csv")    

if __name__ == '__main__':
    main()



    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    