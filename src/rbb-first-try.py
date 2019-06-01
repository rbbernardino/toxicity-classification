#%% Change working directory from the workspace root to the ipynb file location. Turn this addition off with the DataScience.changeDirOnImportExport setting
# ms-python.python added
import os
try:
	os.chdir(os.path.join(os.getcwd(), 'src'))
	print(os.getcwd())
except:
	pass

#%%
import numpy as np
import pandas as pd 
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import matplotlib.pyplot as plt
import random
import re, os
import nltk
from nltk.corpus import stopwords
import warnings
warnings.simplefilter("ignore", UserWarning)
print(os.listdir("../input"))
get_ipython().run_line_magic('matplotlib', 'inline')


#%%
text_all = pd.read_csv("../input/train.csv")
test_data = pd.read_csv("../input/test.csv")


#%%
def clean_str(string):
    # split "he'll" and punctuation
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    
    # remove repeated spaces
    string = re.sub(r"\s{2,}", " ", string)

    # remove html tags, numbers, ' and _
    cleanr = re.compile('<.*?>')
    string = re.sub(r'\d+', '', string)
    string = re.sub(cleanr, '', string)
    string = re.sub("'", '', string)
    string = string.replace('_', '')

    # fix words like "finallllly" and "awwwwwesome"
    pttrn_repchar = re.compile(r"(.)\1{2,}")
    string = pttrn_repchar.sub(r"\1\1", string)
    
    # TODO fix common spelling errors

    # Stop words
    #stop_words = set(stopwords.words('english'))
    #word_list = text_to_word_sequence(string)
    #no_stop_words = [w for w in word_list if not w in stop_words]
    #no_stop_words = " ".join(no_stop_words)
    #string = no_stop_words

    return string.strip().lower()


#%%
clean_str('asdf\n  sdaf日本語sadf ')


#%%
def print_full(x):
    pd.set_option('display.max_rows', len(x))
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 2000)
    pd.set_option('display.float_format', '{:20,.2f}'.format)
    pd.set_option('display.max_colwidth', -1)
    display(x)
    pd.reset_option('display.max_rows')
    pd.reset_option('display.max_columns')
    pd.reset_option('display.width')
    pd.reset_option('display.float_format')
    pd.reset_option('display.max_colwidth')

def hasNonASCII(s):
    clean_str(s)
    try:
        s.encode(encoding='utf-8').decode('ascii')
    except UnicodeDecodeError:
        return False
    else:
        return True

def countNonASCII(s):
    if hasNonASCII(s):
        space_split = s.split(' ')
        non_ascii_count = 0
        for item in space_split:
            if(not hasNonASCII(item)):
                non_ascii_count += 1
        return non_ascii_count
    else:
        return 0


#%%
test_data.head(10)

#%% [markdown]
# ### Check train data language

#%%
# print_full(text_all.sample(10).filter(['comment_text']))
text_all['ASCII'] = text_all['comment_text'].apply(hasNonASCII)
non_ascii_rows = text_all[~text_all['ASCII']]
print("non-ASCII characters:", len(non_ascii_rows), "samples")


#%%
text_all['nASCII_count'] = (non_ascii_rows['comment_text']
                            .apply(countNonASCII))


#%%
display(non_ascii_rows
        .sort_values(by=['nASCII_count'], ascending=False)
        .head(10))


#%%
nascii_tox_q = '(toxic + severe_toxic + obscene + threat + insult + identity_hate)>0'
tox_nonascii = (non_ascii_rows
                .sort_values(by=['nASCII_count'], ascending=False)
                .query(nascii_tox_q))
print("Toxic with non-ascii chars:", len(tox_nonascii))
print_full(tox_nonascii
           .filter(['comment_text', 'nASCII_count'])
           .head(100)
           .sample(10)) 

#%% [markdown]
# ## Some problems found
# - **17215** samples with non ASCII characters

#%%
indexes = [
    123420, # [text]
    109029, 46638, # very long / very short
    47648, # emojis
    117817, # "do-do-do-do-do" pattern
    144121, 87185, # some foreing words
    126, # bunch of weird symbol
    72146, # speratated by '•' char
    10359, # translated text with untranslated sentences (starts with "Translated text")
    147587, # almost no english, contains 'Sry for no English'
    24515, # extremelly offensive with bunch of non-ascii characters (possibly to full AI systems)
]

#%% [markdown]
# ### Check Test data language

#%%
# print_full(test_data.sample(10).filter(['comment_text']))
test_data['ASCII'] = test_data['comment_text'].apply(hasNonASCII)
test_non_ascii_rows = test_data[~test_data['ASCII']]
print("non-ASCII characters:", len(test_non_ascii_rows), "samples")


#%%
test_data['nASCII_count'] = (test_data['comment_text']
                            .apply(countNonASCII))


#%%
test_non_ascii_rows = test_data[test_data['ASCII']]
print_full(test_non_ascii_rows
        .sort_values(by=['nASCII_count'], ascending=False)
        .head(10))

#%% [markdown]
#  # Texto sem Pré-processamento

#%%
text_all['comment_text'][0]

#%% [markdown]
# # Texto limpo

#%%
text_all['comment_text'] = text_all['comment_text'].apply(lambda x: clean_str(x))


#%%
text_all['comment_text'][0]


#%%
text_all.columns
labels = text_all.columns[2:]


#%%
labels

#%% [markdown]
# # Nuvem de palavras para cada label

#%%

list_words = []
for label  in labels:
    text= ""
    for comment, li in zip(text_all['comment_text'], text_all[label]):
            if li == 1:
                text += " "+comment
    print(label)
    wordcloud = WordCloud(max_font_size=100, max_words=1000000, background_color="white").generate(text)
    plt.figure()
    plt.imshow(wordcloud,interpolation="bilinear")
    plt.axis("off")
    plt.show()
    list_words.append(wordcloud.words_)
                

#%% [markdown]
# # Gráfico com as palavras que mais aparecem para cada Label

#%%
i = 0;
for label in labels:

    words = list(list_words[i].keys())
    frequencia = list(list_words[i].values())

    print(label)
    get_colors = lambda n: list(map(lambda i: "#" + "%06x" % random.randint(0, 0xFFFFFF),range(30)))
    plt.figure(figsize=(15,10))
    plt.bar(words[:30], frequencia[:30], color=get_colors(30))

    plt.xticks(rotation=50)
    plt.xlabel("Palavras")
    plt.ylabel("Frequência")
    plt.show()
    i += 1


