from flask import Flask, render_template,  request
from joblib import load

# App definition
app = Flask(__name__)


def load_model():
    '''Load trained model'''
    with open('models\LR_model.pkl', 'rb') as file:
        return load(file)


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/result', methods=['POST'])
def get_result():
    if request.method == 'POST':
        input_text = request.form['text']
        
        if not input_text:
            return render_template("error.html", message="Input text is missing")
        
        data = [input_text]
        model = load_model()
        result = model.predict(data)
        
        if int(result) == 1:
            my_prediction = "This review is positive"
        else:
            my_prediction = "This review is negative"
            
        return render_template("result.html", prediction=my_prediction)


if __name__ == '__main__':
    app.run(debug=True)
