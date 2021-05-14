import csv
from os import error
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd
import json

def setSimilarity(genre_list):
    transformer = CountVectorizer()
    genres_vector = transformer.fit_transform(genre_list)
    similarity = cosine_similarity(genres_vector, genres_vector)
    similarity = similarity.argsort()
    similarity = similarity[:, ::-1]
    return similarity

class Content:
    def __init__(self, type):
        if(type == "movie"):
            self.content_type = "movie"
            self.content_df = pd.read_csv("final_movie_upload_data.csv")            
            self.content_df['content_genre'] = self.content_df['content_genre'].str.split('\n').str[1].str.replace(" ", "").str.replace(',', " ")
            self.genre_similarity = setSimilarity(self.content_df['content_genre'])
        else:
            self.content_type = "book"
            self.content_df = pd.read_csv("final_movie_upload_data.csv")
        
    def recommendByUserEmotion(self, init_emotion, goal_emotion):
        try:
            user_init = pd.DataFrame([init_emotion])
            user_goal = pd.DataFrame([goal_emotion])
            
            #find the userinitmood -> usergoalmood vector
            user_vector = user_goal - user_init

            #find the userinitmood -> reviewmood vector
            user_vector_list = user_vector.values.tolist()

            if self.content_type == 'movie':
                review_emotionDF = pd.DataFrame(data=self.content_df, columns=['review_sadness', 'review_joy', 'review_fear', 'review_disgust', 'review_anger'])
                review_vector = pd.DataFrame(user_vector_list*(int)(review_emotionDF.size/5), columns=['review_sadness', 'review_joy', 'review_fear', 'review_disgust', 'review_anger'])
                review_vector = review_emotionDF - review_vector
                cosine_sim = linear_kernel(user_vector, review_vector)  
                sim_scores = list(enumerate(cosine_sim[0]))
                sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
                movie_indices = [i[0] for i in sim_scores[0:2]]
                closest_items = self.content_df.iloc[movie_indices]
                #closest_items = {self.content_df.iloc[sim_scores[1][0]], self.content_df.iloc[sim_scores[2][0]]}
            else:
                # reviewDF = pd.DataFrame(data=book_df, columns=['book_idx', 'review_sadness', 'review_joy', 'review_fear', 'review_disgust', 'review_anger'])
                # reviewDF['user_sadness'] = user_vector['sadness']
                # reviewDF['user_joy'] = user_vector['joy']
                # reviewDF['user_fear'] = user_vector['fear']
                # reviewDF['user_disgust'] = user_vector['disgust']
                # reviewDF['user_anger'] = user_vector['anger']
                # print(reviewDF)
                # reviewDF = reviewDF.dropna()
                # review_emotionDF = pd.DataFrame(data=reviewDF, columns=['review_sadness', 'review_joy', 'review_fear', 'review_disgust', 'review_anger'])
                # review_vector = pd.DataFrame(user_vector_list*(len(review_emotionDF)), columns=['review_sadness', 'review_joy', 'review_fear', 'review_disgust', 'review_anger'])
                # print(review_vector)
                # print(review_emotionDF)
                #review_vector = review_emotionDF - review_vector
                #cosine_sim = linear_kernel(user_vector, review_vector)  
                # sim_scores = list(enumerate(cosine_sim[0]))
                # sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
                #closest_item = book_df.iloc[sim_scores[1][0]]
                closest_items = 'temp'

            # #find the cosine similarity of vectors
            # cosine_sim = linear_kernel(user_vector, review_vector)  
            # sim_scores = list(enumerate(cosine_sim[0]))
            # sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
            # closest_item = movie_df.iloc[sim_scores[1][0]]
            #print(closest_item)

            #response = {"similar_description" : {closest_item.to_json(orient="columns")}}
            #return response
            return closest_items.to_json(orient="records")
        except:
            return error 

    def recommendByUserSentence(self, goal_sentence):
        vector_new = self.content_df['content_description'].copy().append(pd.Series([goal_sentence]), ignore_index=True)
        tfidf = TfidfVectorizer(stop_words='english')
        tfidf_matrix = tfidf.fit_transform(vector_new)

        cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
        idx = vector_new.size-1
        #print(cosine_sim.shape)
        sim_scores = list(enumerate(cosine_sim[idx]))
        #print(sim_scores)
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

        movie_indices = [i[0] for i in sim_scores[1:3]]
        closest_items = self.content_df.iloc[movie_indices]
        #items = self.content_df.iloc[sim_scores[1][0]]
        return closest_items.to_json(orient="records")

    def recommendByGenre(self, title, top):
            search_df = self.content_df[self.content_df['content_movietitle'] == title]
            search_df_index = search_df.index.values
            similarity_index = self.similarity[search_df_index, :top].reshape(-1)
            self.similarity = similarity_index[similarity_index != search_df_index]
            result = self.content_df.ioc[similarity_index][:10]
            return result

    def recommendByItemContent(self, items, top):                
        if self.content_type == 'movie':           

            #recommend by genre
            for i in items:
                title = self.content_df.loc[self.content_df['movie_idx'] == i]
                #recommendByGenre(title, top/2)

            #recommend by description
            #(optional) sort them by like/dislike counts

    

    # def recommendByDescription(self, title, top):