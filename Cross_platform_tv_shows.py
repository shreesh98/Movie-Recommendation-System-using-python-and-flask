#!/usr/bin/env python
# coding: utf-8

# In[200]:


import pandas as pd
import numpy as np
from ast import literal_eval
from wordcloud import WordCloud
import string
from string import punctuation
# import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('white')
# get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
from string import punctuation
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


# In[267]:


df1=pd.read_csv('Cross_platform_tv_shows.csv')


# In[269]:


df1.head()
# df1.drop('Unnamed: 0',axis=1)


# In[270]:


df=df1


# In[271]:


df.sample(15)
#df.drop('index',axis=1)


# In[272]:


from nltk.stem import SnowballStemmer
stemmer_s = SnowballStemmer("english")


# In[273]:


stop_nltk = stopwords.words("english")


# In[274]:


stop_updated = stop_nltk + list(punctuation) + ["tv","shows","british","american","new zealand","taiwanese","chinese","argentenian","malaysian","korean","indian"]


# In[275]:


def  clean_txt(genres):
    tokens = word_tokenize(genres.lower())
    stemmed = [stemmer_s.stem(term) for term in tokens
               if term not in stop_updated and len(term) > 2]
    res= " ".join(stemmed)
    res1= res.replace("show ","")
    return res1


# In[276]:


df['clean_genre'] = df.Genres.apply(clean_txt)
#df.head()


# In[78]:


#df3.to_csv('Amazon Prime TV.csv')


# In[277]:


genres_combined = " ".join(df.clean_genre.values)
#genres_combined


# In[278]:


word_cloud = WordCloud(width=800,height=800,background_color='white').generate_from_text(genres_combined)


# In[279]:


# plt.figure(figsize=[8,8])
# plt.imshow(word_cloud)


# In[280]:


all_terms = []
fdist = {}
all_terms = genres_combined.split(" ")
for word in all_terms:
    fdist[word] = fdist.get(word,0) + 1


# In[281]:


freq = {"words":list(fdist.keys()),"freq":list(fdist.values())}
df_dist = pd.DataFrame(freq)


# In[282]:


# get_ipython().run_line_magic('matplotlib', 'inline')
df_dist.sort_values(ascending=False, by="freq").head(25).plot.bar(x= "words", y= "freq",figsize=(20,10))


# In[227]:


#df.head()


# In[283]:


df = df.reset_index()
titles = df[['Name','Platform']]
indices = pd.Series(df.index, index=df['Name'])


# In[284]:


indices


# In[285]:


def get_recommendations_tv(title):
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:31]
    movie_indices = [i[0] for i in sim_scores]
    return titles.iloc[movie_indices]


# In[286]:


count = CountVectorizer(analyzer='word',ngram_range=(1, 2),min_df=0, stop_words='english')
count_matrix = count.fit_transform(df['clean_genre'])


# In[288]:


count_matrix.shape


# In[287]:


cosine_sim = cosine_similarity(count_matrix, count_matrix)


# In[289]:


cosine_sim[0]


# In[292]:


# get_recommendations('Red Oaks – Season 2 [Ultra HD]')


# In[ ]:
