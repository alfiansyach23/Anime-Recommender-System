import streamlit as st
import pandas as pd
import numpy as np
import re
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import os

# data pre-processing
project_dir = os.path.dirname(os.path.abspath(__file__))

anime_movies_scrapping_file_path = os.path.join(project_dir, 'Anime_Recommender_System_Scrapping.csv')
anime_movies_scrapping = pd.read_csv(anime_movies_scrapping_file_path)

anime_movies_file_path = os.path.join(project_dir, 'Anime_Recommender_System.csv')
anime_movies = pd.read_csv(anime_movies_file_path)

df = anime_movies_scrapping[['Rank', 'Title', 'Rating', 'Image_URL', 'Episodes', 'Dates', 'Members']].merge(
     anime_movies[['title', 'genres', 'studios', 'producers', 'synopsis']],
     left_on='Title', right_on='title', how='left')

df.drop(columns='title', inplace=True)

df.rename(columns={'genres': 'Genres',
                   'studios': 'Studios',
                   'producers': 'Producers',
                   'synopsis': 'Synopsis'},
                   inplace=True)

df.replace(r'^\s*$', np.nan, regex=True, inplace=True)
df.dropna(inplace=True)

def remove_punctuation(df, column_name):
    df[column_name] = df[column_name].str.replace(r"[^\w\s]", " ", regex=True)
    df[column_name] = df[column_name].str.replace(r"\s+", " ", regex=True)

remove_punctuation(df, 'Title')
remove_punctuation(df, 'Synopsis')

def remove_square_brackets_and_quotation_marks(df, column_name):
    df[column_name] = df[column_name].str.strip("[]").str.strip("'")
    df[column_name] = df[column_name].apply(lambda x: ' '.join(re.findall(r'\b\w+\b', x)))

remove_square_brackets_and_quotation_marks(df, 'Genres')
remove_square_brackets_and_quotation_marks(df, 'Studios')
remove_square_brackets_and_quotation_marks(df, 'Producers')

def process_episodes(episodes):
    match = re.match(r'(\w+) \((\d+) eps\)', episodes)
    if match:
        return match.group(1), int(match.group(2))
    else:
        return None, None

df['Type'], df['Episodes'] = zip(*df['Episodes'].map(process_episodes))
df['Episodes'] = df['Episodes'].replace(r'[\(\)eps]+', '', regex=True)
df['Episodes'] = df['Episodes'].fillna('Unknown')

def process_dates(date_range):
    if '-' in date_range:
        start_date, end_date = map(str.strip, date_range.split('-'))
        end_date = 'Present' if end_date == '' else end_date
    else:
        start_date, end_date = date_range, 'Present'
    return start_date, end_date

df[['StartDate', 'EndDate']] = df['Dates'].apply(process_dates).apply(pd.Series)
df = df.drop(columns='Dates')

def determine_status(end_date):
    return 'Currently Airing' if end_date == 'Present' else 'Finished Airing'

df['Status'] = df['EndDate'].apply(determine_status)

col_order = ['Rank', 'Title', 'Rating', 'Type', 'Episodes', 'StartDate', 'EndDate', 'Status', 'Genres', 'Studios', 'Producers', 'Synopsis', 'Members']
df = df[col_order]

df.replace(r'^\s*$', np.nan, regex=True, inplace=True)
df.dropna(inplace=True)

# Content-based filtering
df['Combined'] = df['Genres'] + ' ' + df['Studios'] + ' ' + df['Producers']
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(df['Combined'])
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
df['Content_Based_Score'] = cosine_sim.diagonal()

# Popularity-based filtering
scaler = MinMaxScaler()
df[['Rank_Normalized', 'Rating_Normalized', 'Members_Normalized']] = scaler.fit_transform(df[['Rank', 'Rating', 'Members']])
weights = {'Rank_Normalized': 0.2, 'Rating_Normalized': 0.5, 'Members_Normalized': 0.3}
df['Popularity_Score'] = df.apply(lambda row: sum(row[col] * weights[col] for col in weights), axis=1)

# Combine content-based and popularity-based scores
content_based_weight = 0.7
popularity_based_weight = 0.3
df['Final_Score'] = content_based_weight * df['Content_Based_Score'] + popularity_based_weight * df['Popularity_Score']
df_sorted_final = df.sort_values(by='Final_Score', ascending=False)

st.title('Anime Recommender System')

anime_input = st.text_input('Enter Anime Title:')

def search(title, df_sorted_final, cosine_sim):
    try:
        idx = df_sorted_final[df_sorted_final['Title'].str.contains(title, case=False)].index[0]
    except IndexError:
        st.warning(f"No matching anime found for the input '{title}'.")
        return None

    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    top_similar_indices = [i[0] for i in sim_scores[1:6]]

    recommended_anime = df_sorted_final[['Title', 'Final_Score']].iloc[top_similar_indices]
    return recommended_anime

if st.button('Search'):
    recommended_anime = search(anime_input, df_sorted_final, cosine_sim)
    if recommended_anime is not None:
        # Merge recommended_anime with the original DataFrame df based on the "Title" column
        merged_df = pd.merge(recommended_anime, df[['Title', 'Rating']], on='Title', how='left')

        # Sort merged_df based on the "Final_Score" column in descending order
        merged_df = merged_df.sort_values(by='Final_Score', ascending=False)

        st.markdown('<div style="background-color:#7B66FF;color:#ffffff;padding:10px;border-radius:5px;"><b>Anime Recommendation for You :</b></div>', unsafe_allow_html=True)
        st.write('')

        for i, row in enumerate(merged_df.itertuples(), 1):
            # Use "Rating" column for displaying star rating
            rating = row.Rating if 'Rating' in row._fields else 'N/A'  # Handle the case where 'Rating' is not available
            st.markdown(f'<div style="background-color:#F0F0F0;color:#000000;padding:10px;border-radius:5px;margin-bottom:8px;">{i}. {row.Title} (⭐ {rating:.2f})</div>', unsafe_allow_html=True)
