#Import libraries
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import ast
#%matplotlib inline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

#Import datasets
movies = pd.read_csv("https://raw.githubusercontent.com/4GeeksAcademy/k-nearest-neighbors-project-tutorial/main/tmdb_5000_movies.csv")
credits = pd.read_csv("../data/raw/tmdb_5000_credits.csv")

#Merge datasets
movies_df = movies.merge(credits, on='title')

#Drop irrelevant data
movies_df = movies_df[['movie_id','title','overview','genres','keywords','cast','crew']]

#Drop NAs
movies_df.dropna(inplace = True)

#Define functions for data transformation
def convert(obj):
    L = []
    for i in ast.literal_eval(obj):
        L.append(i['name'])
    return L

def convert3(obj):
    L = []
    count = 0
    for i in ast.literal_eval(obj):
        if count < 3:
            L.append(i['name'])
        count +=1  
    return L

def fetch_director(obj):
    L = []
    for i in ast.literal_eval(obj):
        if i['job'] == 'Director':
            L.append(i['name'])
            break
    return L

def collapse(L):
    L1 = []
    for i in L:
        L1.append(i.replace(" ",""))
    return L1

#Apply functions
movies_df['genres'] = movies_df['genres'].apply(convert)
movies_df['keywords'] = movies_df['keywords'].apply(convert)
movies_df['cast'] = movies_df['cast'].apply(convert3)
movies_df['crew'] = movies_df['crew'].apply(fetch_director)
movies_df['overview'] = movies_df['overview'].apply(lambda x : x.split())
movies_df['cast'] = movies_df['cast'].apply(collapse)
movies_df['crew'] = movies_df['crew'].apply(collapse)
movies_df['genres'] = movies_df['genres'].apply(collapse)
movies_df['keywords'] = movies_df['keywords'].apply(collapse)

#Reducing dataset by combining cols
movies_df['tags'] = movies_df['overview']+movies_df['genres']+movies_df['keywords']+movies_df['cast']+movies_df['crew']
new_df = movies_df[['movie_id','title','tags']]
new_df['tags'] = new_df['tags'].apply(lambda x :" ".join(x))

#Creating text vectorizer
cv = CountVectorizer(max_features=5000 ,stop_words='english')
vectors = cv.fit_transform(new_df['tags']).toarray()
similarity = cosine_similarity(vectors)
sorted(list(enumerate(similarity[0])),reverse =True , key = lambda x:x[1])[1:6]

#Creating recommendation function
def recommend(movie):
    movie_index = new_df[new_df['title'] == movie].index[0] ##fetching the movie index
    distances = similarity[movie_index]
    movie_list = sorted(list(enumerate( distances)),reverse =True , key = lambda x:x[1])[1:6]
    
    for i in movie_list:
        print(new_df.iloc[i[0]].title)
