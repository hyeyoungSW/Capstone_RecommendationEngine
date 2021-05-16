import csv
from os import error
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd
import json


def setDescriptionSimilarity(description):
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(description)
    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
    return cosine_sim

class Book:
    def __init__(self):
        self.content_df = pd.read_csv("final_book_upload_data.csv")
        self.description_similarity = setDescriptionSimilarity(self.content_df['content_description'])
    
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
            sim_scores = list(enumerate(cosine_sim[0]))
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
            movie_indices = [i[0] for i in sim_scores[0:2]]
            closest_items = self.content_df.iloc[movie_indices]
            return closest_items
            #closest_items = {self.content_df.iloc[sim_scores[1][0]], self.content_df.iloc[sim_scores[2][0]]}
        except:
            return error

    def recommendByUserSentence(self, goal_sentence):
        try:
            vector_new = self.content_df['content_description'].copy().append(pd.Series([goal_sentence]), ignore_index=True)
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
            return closest_items
        except:
            return error

    def recommendByDescription(self, title, top):
        indices = pd.Series(self.content_df.index, index=self.content_df['content_booktitle'])
        idx = indices[title]
        sim_scores = list(enumerate(self.description_similarity[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:11]
        movie_indices = [i[0] for i in sim_scores]

        result = self.content_df.iloc[movie_indices]

        return result

    def recommendByItemContent(self, items, top):
        try:
            result_description={}
            for title in items:
            # recommend by description
                result_description[title] = {'recommend_way' : 'description', 'items' : self.recommendByDescription(title, 10).to_json(orient='records')}
            
            # (optional) sort them by like/dislike counts
            return result_description
        except:
            return error