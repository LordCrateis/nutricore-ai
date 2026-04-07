from flask import Flask, request, jsonify, render_template
from ml_model import predict, train
import threading

app = Flask(__name__)

_train_thread = threading.Thread(target=train, daemon=True)
_train_thread.start()


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/analyze', methods=['POST'])
def analyze():
    data = request.json

    if _train_thread.is_alive():
        return jsonify({'success': False, 'error': 'Model is still initialising, please retry in a moment.'}), 503

    required = ['age', 'weight', 'height', 'gender', 'activity', 'goal', 'diet', 'health']
    missing = [k for k in required if k not in data]
    if missing:
        return jsonify({'success': False, 'error': f'Missing required fields: {missing}'}), 400

    try:
        # deficiencies is optional — comes in as a list of strings from the frontend
        deficiencies = data.get('deficiencies', [])
        if isinstance(deficiencies, str):
            # Handle comma-separated string fallback
            deficiencies = [d.strip() for d in deficiencies.split(',') if d.strip()]

        result = predict(
            age          = int(data['age']),
            weight       = float(data['weight']),
            height       = float(data['height']),
            gender       = data['gender'],
            activity     = float(data['activity']),
            goal         = data['goal'],
            diet         = data['diet'],
            health       = data['health'],
            deficiencies = deficiencies,
        )
        return jsonify({'success': True, 'data': result})
    except (ValueError, KeyError) as e:
        return jsonify({'success': False, 'error': str(e)}), 400


if __name__ == '__main__':
    app.run(debug=False, port=5050)
