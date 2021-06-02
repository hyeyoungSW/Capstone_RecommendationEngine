import csv
from os import error
from pandas.core.frame import DataFrame
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd
import json

def setGenreSimilarity(genre):
    transformer = CountVectorizer()
    genres_vector = transformer.fit_transform(genre)
    similarity = cosine_similarity(genres_vector, genres_vector)
    similarity = similarity.argsort()
    similarity = similarity[:, ::-1]
    return similarity

def setDescriptionSimilarity(description):
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(description)
    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
    return cosine_sim

class Movie:
    def __init__(self):
        self.content_df = pd.read_csv("new_final_movie_upload_data_encoding.csv")
        # self.content_df['etc'] = self.content_df['etc'].str.split(
        #     '\n').str[1].str.replace(" ", "").str.replace(',', " ")
        self.genre_similarity = setGenreSimilarity(self.content_df['etc'])
        self.description_similarity = setDescriptionSimilarity(self.content_df['description'])
        self.en_feature = ['idx', 'title_en', 'etc', 'description_en']
        self.kr_feature = ['idx', 'title', 'etc', 'description']

    
    def recommendByUserEmotion(self, init_emotion, goal_emotion):
        try:
            user_init = pd.DataFrame([init_emotion])
            user_goal = pd.DataFrame([goal_emotion])

            # find the userinitmood -> usergoalmood vector
            user_vector = user_goal - user_init

            # find the userinitmood -> reviewmood vector
            user_vector_list = user_vector.values.tolist()

            review_emotionDF = pd.DataFrame(data=self.content_df, columns=['review_sadness', 'review_joy', 'review_fear', 'review_disgust', 'review_anger'])
            review_vector = pd.DataFrame(user_vector_list*(int)(review_emotionDF.size/5), columns=['review_sadness', 'review_joy', 'review_fear', 'review_disgust', 'review_anger'])
            review_vector = review_emotionDF - review_vector
            cosine_sim = linear_kernel(user_vector, review_vector)
            print(cosine_sim[0])
            print ("\n---------\n")
            sim_scores = list(enumerate(cosine_sim[0]))
            print(sim_scores)
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
            movie_indices = [i[0] for i in sim_scores[0:2]]
            closest_items = self.content_df.iloc[movie_indices]
            #print(closest_items[self.kr_feature])
            return closest_items[self.kr_feature]
            #closest_items = {self.content_df.iloc[sim_scores[1][0]], self.content_df.iloc[sim_scores[2][0]]}
        except:
            return error

    def recommendByUserSentence(self, goal_sentence):
        try:
            vector_new = self.content_df['description'].copy().append(
                pd.Series([goal_sentence]), ignore_index=True)
            tfidf = TfidfVectorizer(stop_words='english')
            tfidf_matrix = tfidf.fit_transform(vector_new)

            cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
            idx = vector_new.size-1
            # print(cosine_sim.shape)
            sim_scores = list(enumerate(cosine_sim[idx]))
            # print(sim_scores)
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

            movie_indices = [i[0] for i in sim_scores[1:3]]
            closest_items = self.content_df.iloc[movie_indices]
            #items = self.content_df.iloc[sim_scores[1][0]]
            return closest_items[self.kr_feature]
        except:
            return error

    def recommendByGenre(self, title, top):
        try:
            search_df = self.content_df[self.content_df['title'] == title]
            search_df_index = search_df.index.values
            similarity_index = self.genre_similarity[search_df_index, :int(top)].reshape(-1)
            similarity_index = similarity_index[similarity_index != search_df_index]
            result = self.content_df.iloc[similarity_index][:int(top)]

            return result[self.kr_feature]
        except:
            return error

    def recommendByDescription(self, title, top):
        try:
            indices = pd.Series(self.content_df.index, index=self.content_df['title'])
            idx = indices[title]
            sim_scores = list(enumerate(self.description_similarity[idx]))
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
            sim_scores = sim_scores[1:1+int(top)]
            movie_indices = [i[0] for i in sim_scores]
            result = self.content_df.iloc[movie_indices]

            return result[self.kr_feature]
        except:
            return error

    def recommendByItemContent(self, items, top):
        try:
            result = pd.DataFrame(columns = self.kr_feature)

            for title in items:
                # recommend by description
                recommended_list = pd.concat([self.recommendByDescription(title, int(top)/2), self.recommendByGenre(title, int(top)/2)], ignore_index=True)
                result = pd.concat([result, recommended_list[self.kr_feature]], ignore_index=True)

            result = result.drop_duplicates(subset=['idx'])
            return {"type":"movie", "total_count" : len(result), "items" : result.to_json(orient='records')}
        except:
            return error

    def getItemByIndex(self, idx_list):
        try:            
            result = pd.DataFrame(columns = self.en_feature)

            for i in idx_list:
                idx = int(i)
                item = self.content_df.loc[self.content_df['idx'] == idx]
                result = pd.concat([result, item[self.en_feature]], ignore_index=True)
            return result.to_json(orient='records')
        except:
            return error