import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

"""
路径映射以及模型调用
"""
app = Flask(__name__)  # Initialize the flask App
model = pickle.load(open('model.pkl', 'rb'))  # Load the trained model


@app.route('/')  # Homepage
def home():
    return render_template('index.html')


# post请求，给出表格数据
@app.route('/predict', methods=['POST'])
def predict():
    # UI rendering the results
    # Retrieve values from a form
    init_features = [float(x) for x in request.form.values()]
    final_features = [np.array(init_features)]
    prediction = model.predict(final_features)  # Make a prediction

    return render_template('index.html',
                           prediction_text='Predicted Species: {}'.format(prediction))  # Render the predicted result


# post请求，给出json数据
@app.route('/predict_api', methods=['POST'])
def predict_api():
    try:
        # 获取请求中的数据
        data = request.get_json(force=True)
        # 将数据转换为numpy数组
        input_data = np.array(data['input']).reshape(1, -1)
        # 进行预测
        prediction = model.predict(input_data)
        # 返回预测结果
        return jsonify({'Predicted Species: ': int(prediction[0])})
    except Exception as e:
        return jsonify({'error': str(e)}), 400


if __name__ == "__main__":
    app.run(debug=True)
