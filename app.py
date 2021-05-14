from flask import Flask, render_template, request
import getContent
import csv
import json

movieContent = getContent.Content("movie")
bookContent = getContent.Content("book")

app = Flask(__name__)

@app.route('/')
def main():
    return {'helloworld':'test'}

@app.route('/movie/sentence', methods=['POST'])
def getMovieBySentence():
    user_status = request.form 
    goal_sentence = user_status['sentence']
    recommendBySentence = movieContent.recommendByUserSentence(goal_sentence)
    
    return recommendBySentence
    #return (movieContent.RecommendByReviewEmotion(init_emotion, goal_emotion))
    #return (user_status)

@app.route('/movie/emotion', methods=['POST'])
def getMovieByEmotion():
    user_status = request.form
    init_emotion = json.loads(user_status['init_emotion'])
    goal_emotion = json.loads(user_status['goal_emotion']) 
    recommendByEmotion = movieContent.recommendByUserEmotion(init_emotion, goal_emotion)
    
    return recommendByEmotion
    #return user_status

@app.route('/movie/content', methods=['POST'])
def getMovieByItemContent():
    user_status = request.form
    item_list = json.loads(user_status['selected_movies'])
    recommendedMovieLists = movieContent.recommendByItemContent(item_list)
    
    return recommendedMovieLists

# @app.route('/book/sentence', methods=['POST'])
# def getBookBySentence():
#     user_status = request.form 
#     goal_sentence = user_status['sentence']
#     recommendBySentence = bookContent.recommendByUserSentence(goal_sentence)
    
#     return recommendBySentence

# @app.route('/book/emotion', methods=['POST'])
# def getBookByEmotion():
#     user_status = request.form
#     init_emotion = json.loads(user_status['init_emotion'])
#     goal_emotion = json.loads(user_status['goal_emotion']) 
#     recommendByEmotion = bookContent.recommendByUserEmotion(init_emotion, goal_emotion)
    
#     return recommendByEmotion

if __name__ == '__main__':
    app.run()