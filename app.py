from flask import Flask
app = Flask(__name__)

@app.route('/')
def hello_geek():
    return '<h1>Segarbox v1.0.0</h1>'


if __name__ == "__main__":
    app.run(debug=True)
