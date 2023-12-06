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

col_order = ['Rank', 'Title', 'Rating', 'Image_URL', 'Type', 'Episodes', 'StartDate', 'EndDate', 'Status', 'Genres', 'Studios', 'Producers', 'Synopsis', 'Members']
df = df[col_order]

df.replace(r'^\s*$', np.nan, regex=True, inplace=True)
df.dropna(inplace=True)
df.reset_index(inplace=True)

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

def get_image_url(title, df):
    matching_row = df[df['Title'].str.contains(title, case=False)]
    if not matching_row.empty and not pd.isnull(matching_row.iloc[0]['Image_URL']):
        return matching_row.iloc[0]['Image_URL']
    else:
        return None

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

    recommended_anime = df_sorted_final[['Title', 'Genres', 'Synopsis', 'Final_Score', 'Image_URL']].iloc[top_similar_indices]
    recommended_anime['Image_URL'] = recommended_anime['Title'].apply(lambda x: get_image_url(x, df_sorted_final))
    return recommended_anime

def truncate_text(text, max_chars):
    return (text[:max_chars] + '...') if len(text) > max_chars else text

if st.button('Search'):
    matching_anime_titles = df_sorted_final[df_sorted_final['Title'].str.contains(anime_input, case=False)]['Title'].tolist()

    if matching_anime_titles:
        st.markdown('<div style="background-color:#7B66FF;color:#ffffff;padding:10px;border-radius:5px;"><b>List of Suitable Anime Titles :</b></div>', unsafe_allow_html=True)
        st.write('')

        for anime_title in matching_anime_titles[:5]:
            col1, col2 = st.columns([1, 3])

            anime_data = df_sorted_final[df_sorted_final['Title'] == anime_title].iloc[0]

            image_url = get_image_url(anime_data['Title'], df_sorted_final)
            if image_url:
                col1.image(image_url, use_column_width=True, output_format='png')
            else:
                placeholder_image = "https://via.placeholder.com/150x228"
                col1.image(placeholder_image, use_column_width=True, output_format='png')

            genres_with_commas = ', '.join(anime_data['Genres'].split())
            truncated_synopsis = truncate_text(anime_data['Synopsis'], max_chars=250)
            
            col2.write(f"**Title:** {anime_data['Title']}")
            col2.markdown(f"**Rating:** ⭐ {anime_data['Final_Score']:.2f}\n\n**Genres:** {genres_with_commas}\n\n**Synopsis:** {truncated_synopsis}")

    recommended_anime = search(anime_input, df_sorted_final, cosine_sim)

    if recommended_anime is not None:
        st.markdown('<div style="background-color:#7B66FF;color:#ffffff;padding:10px;border-radius:5px;"><b>Anime recommendations you might be looking for :</b></div>', unsafe_allow_html=True)
        st.write('')

        for i, row in enumerate(recommended_anime.itertuples(), 1):
            rating = row.Final_Score
            col1, col2 = st.columns([1, 3])

            image_url = get_image_url(row.Title, df_sorted_final)
            if image_url:
                col1.image(image_url, use_column_width=True, output_format='png')
            else:
                placeholder_image = "https://via.placeholder.com/150x228"
                col1.image(placeholder_image, use_column_width=True, output_format='png')

            genres_with_commas = ', '.join(row.Genres.split())
            truncated_synopsis = truncate_text(row.Synopsis, max_chars=250)
            col2.markdown(f"**Title:** {row.Title}\n\n**Rating:** ⭐ {rating:.2f}\n\n**Genres:** {genres_with_commas}\n\n**Synopsis:** {truncated_synopsis}...")

def truncate_text(text, max_chars):
    return (text[:max_chars] + '...') if len(text) > max_chars else text
