import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.feature_extraction.text import TfidfVectorizer

anime_df = pd.read_csv('anime_with_synopsis.csv')
#rating_df = pd.read_csv('rating.csv')

anime_df = anime_df.rename({'sypnopsis': 'Synopsis'}, axis=1)
anime_df = anime_df.rename({'MAL_id': 'anime_id'}, axis=1)

#rating_df = rating_df[(rating_df["rating"] != -1)]

anime_copy = anime_df.copy()
anime_copy = anime_copy[['Name', 'Genres', 'Synopsis']]
anime_copy['tagline'] = anime_copy['Synopsis'] + anime_copy['Genres']

tfidf = TfidfVectorizer(stop_words='english')
anime_copy['tagline'] = anime_copy['tagline'].fillna('')
tfidf_matrix = tfidf.fit_transform(anime_copy['tagline'])

from sklearn.metrics.pairwise import linear_kernel
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

def get_recommendations(Name, cosine_sim=cosine_sim):
    idx = anime_copy[anime_copy['Name'] == Name].index[0]
    sim_scores = sorted(list(enumerate(cosine_sim[idx])), reverse=True, key=lambda x: x[1])[1:11]
    anime_indices = [i[0] for i in sim_scores]
    return anime_df[['Name', 'Synopsis']].iloc[anime_indices]

def Table(df):
    fig=go.Figure(go.Table(columnorder = [1,2,3],
          columnwidth = [100,100],
            header=dict(values=['Name','Synopsis'],
                        line_color='black',font=dict(color='white',size= 19),height=40,
                        fill_color='red',
                        align=['center','center']),
                cells=dict(values=[df.Name,df.Synopsis],
                       fill_color='#ffdac4',line_color='grey',
                           font=dict(color='black', family="Lato", size=16),
                       align='left')))
    fig.update_layout(height=500, title ={'text': "Top 10 Anime Recommendations", 'font': {'size': 22}})
    return st.plotly_chart(fig,use_container_width=True)
anime_list = anime_df['Name'].values

st.title(':red[Anime] Recommendation System')
selected_anime = st.selectbox(
    "Type or select an anime from the dropdown",
    anime_list
)

if st.button('Show Recommendation'):
    recommended_anime_names = get_recommendations(selected_anime)
    #list_of_recommended_anime = recommended_anime_names.to_list()
   # st.write(recommended_anime_names[['title', 'description']])
    Table(recommended_anime_names)
    
st.write('  '
         )
st.write(' ')
