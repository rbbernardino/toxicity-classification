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

#%% [markdown]
# # 1. Prepare the data


#%%
train_df = pd.read_csv("../input/train.csv")
test_df = pd.read_csv("../input/test.csv")

#%% [markdown]
# ## String cleaning utilities

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

    # remove punctuation
    string = re.sub(r"[…–!—“”\"#$%&’()*+,-./:;<=>?@[\]^_`{|}~]+", "", string)

    # Emojis pattern
    emoji_pattern = re.compile("["
                u"\U0001F600-\U0001F64F"  # emoticons
                u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                u"\U0001F680-\U0001F6FF"  # transport & map symbols
                u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                u"\U00002702-\U000027B0"u"\U000024C2-\U0001F251"
                u"\U0001f926-\U0001f937"u'\U00010000-\U0010ffff'u"\u200d"
                u"\u2640-\u2642"u"\u2600-\u2B55"u"\u23cf"u"\u23e9"u"\u231a"
                u"\u3030"u"\ufe0f"
    "]+", flags=re.UNICODE)
    string = emoji_pattern.sub(u'', string)

    return string.strip().lower()

# test the cleaning functions
test_str = 'as ❤☮☺m☯ 😀 df\n \\ /"12() sdaات کنشی سازf日本語sadf เบอร์10!! ส้มสวย 01แฝดของ08'
print(clean_str(test_str))

#%% [markdown]
# ## Printing and Searching Utilities

#%%
def display_full(x):
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
        return True
    else:
        return False

def countNonASCII(s):
    if hasNonASCII(s):
        space_split = s.split(' ')
        non_ascii_count = 0
        for item in space_split:
            if(hasNonASCII(item)):
                non_ascii_count += 1
        return non_ascii_count
    else:
        return 0

#%% [markdown]
# ## Check train data language

#%%
# train_df = train_df.sample(500).copy()
train_df['comment_text'] = train_df['comment_text'].apply(clean_str)
train_df['nASCII'] = train_df['comment_text'].apply(hasNonASCII)
non_ascii_rows = train_df[train_df['nASCII']]
print("non-ASCII characters:", len(non_ascii_rows), "samples")


#%%
train_df['nASCII_count'] = (train_df['comment_text']
                                .apply(countNonASCII))


#%%
non_ascii_rows = train_df[train_df['nASCII']] # recreate df to include new column
display_full(non_ascii_rows
        .sort_values(by=['nASCII_count'], ascending=False)
        .head(10))


#%%
nascii_tox_q = '(toxic + severe_toxic + obscene + threat + insult + identity_hate)>0'
tox_nonascii = (non_ascii_rows
                .query(nascii_tox_q)
                .sort_values(by=['nASCII_count'], ascending=False))
print("Toxic with non-ascii chars:", len(tox_nonascii))
display_full(tox_nonascii
           .filter(['comment_text', 'nASCII_count'])
           .head(100)
           .sample(10)) 

#%% [markdown]
# ## Some problems found
# - **8477** samples with non ASCII characters
# - **275** samples tagged as toxic with non ASCII characters

#%%
bad_train_indexes = [
    123420, # [text]
    109029, 46638, # very long / very short
    47648, # emojis
    117817, # "do-do-do-do-do" pattern
    144121, 87185, # some foreing words
    126, # bunch of weird symbol
    72146, # speratated by '•' char
    10359, # translated text with untranslated sentences (starts with "Translated text")
    147587, # almost no english, contains 'Sry for no English'
    24515, # extremelly offensive with bunch of non-ascii characters (possibly to fool AI systems)
]

#%% [markdown]
# ------------
# ### Check Test data non ASCII (wraps all above steps)

#%%
test_df['nASCII'] = test_df['comment_text'].apply(hasNonASCII)
test_nASCII_df = test_df[test_df['nASCII']].copy()
print("Test set non-ASCII characters:", len(test_nASCII_df), "samples")
####
test_nASCII_df['nASCII_count'] = (test_nASCII_df['comment_text']
                            .apply(countNonASCII))
display_full(test_nASCII_df
        .sort_values(by=['nASCII_count'], ascending=False)
        .head(100)
        .sample(3))
####

#%%
bad_test_idexes = [
    46301, # mostly german
    98150, # 100% arabic
    49341, # no content, only repeated string
    62122, # Tâmil ?
]

#%% [markdown]
#  # Visualize the words in the comments

#%%
train_df.columns
labels = train_df.columns[2:]
labels

#%% [markdown]
# # Nuvem de palavras para cada label

#%%
list_words = []
for label  in labels:
    text= ""
    for comment, li in zip(train_df['comment_text'], train_df[label]):
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
i = 0
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


