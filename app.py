from flask import Flask,request,jsonify
import App

app = Flask(__name__)

@app.route('/')
def hello_geek():
    return '<h1>Segarbox v1.0.0</h1>'

@app.route('/api/recommendation')
def recommendation():
    user_id = request.args.get('user_id')
    result = App.get.getRecommend(user_id)
    print(result)
    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True)
