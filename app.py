from flask import Flask, render_template, request
import recommendBook
import recommendMovie
import csv
import json

movieContent = recommendMovie.Movie()
bookContent = recommendBook.Book()

app = Flask(__name__)

@app.route('/')
def main():
    return {'testpage':'hello world'}

@app.route('/movie/sentence', methods=['POST'])
def getMovieBySentence():
    user_status = request.form 
    goal_sentence = user_status['sentence']
    recommendBySentence = movieContent.recommendByUserSentence(goal_sentence)
    
    return recommendBySentence.to_json(orient="records")
    #return (movieContent.RecommendByReviewEmotion(init_emotion, goal_emotion))
    #return (user_status)

# @app.route('/movie/emotion', methods=['POST'])
# def getMovieByEmotion():
#     user_status = request.form
#     init_emotion = json.loads(user_status['init_emotion'])
#     goal_emotion = json.loads(user_status['goal_emotion']) 
#     recommendByEmotion = movieContent.recommendByUserEmotion(init_emotion, goal_emotion)
    
#     return recommendByEmotion.to_json(orient="records")
#     #return user_status

@app.route('/movie/content', methods=['POST', 'GET'])
def getMovieByItemContent():
    # user_status = json.loads(request.form)
    # item_list = user_status["selected_items"]
    item_list = ["Zack Snyder's Justice League", "Sound of Metal", "The Shawshank Redemption"]
    recommendedMovieLists = movieContent.recommendByItemContent(item_list, 30)
    
    return json.dumps(recommendedMovieLists)
    #return {1: 1}

@app.route('/book/sentence', methods=['POST', 'GET'])
def getBookBySentence():
    user_status = request.form 
    goal_sentence = user_status['sentence']
    recommendBySentence = bookContent.recommendByUserSentence(goal_sentence)
    return recommendBySentence.to_json(orient="records")

# @app.route('/book/emotion', methods=['POST'])
# def getBookByEmotion():
#     user_status = request.form
#     init_emotion = json.loads(user_status['init_emotion'])
#     goal_emotion = json.loads(user_status['goal_emotion']) 
#     recommendByEmotion = bookContent.recommendByUserEmotion(init_emotion, goal_emotion)
    
#     return recommendByEmotion

@app.route('/book/content', methods=['POST', 'GET'])
def getBookByItemContent():
    # user_status = json.loads(request.form)
    # item_list = user_status["selected_items"]
    item_list = ["Harry Potter and the Half-Blood Prince", "The Lost Continent: Travels in Small Town America"]
    recommendedBookLists = bookContent.recommendByItemContent(item_list, 30)
    
    return json.dumps(recommendedBookLists)

if __name__ == '__main__':
    app.run()