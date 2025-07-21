from flask import Flask, render_template, request, redirect, session, url_for
import sqlite3
import numpy as np
import pickle
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'  # Change this to something secure

# Load model and encoders
with open('Model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('crop_encoder.pkl', 'rb') as f:
    crop_encoder = pickle.load(f)
with open('soil_encoder.pkl', 'rb') as f:
    soil_encoder = pickle.load(f)

# -------------------- AUTH ROUTES --------------------

@app.route('/', methods=['GET'])
def root():
    if 'user' in session:
        return redirect('/homepage')
    return redirect('/login')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        hashed_pw = generate_password_hash(password)

        conn = sqlite3.connect('users.db')
        c = conn.cursor()
        try:
            c.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, hashed_pw))
            conn.commit()
            return redirect('/login')
        except sqlite3.IntegrityError:
            return "Username already exists! Try another one."
        finally:
            conn.close()

    return render_template('signup.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        conn = sqlite3.connect('users.db')
        c = conn.cursor()
        c.execute("SELECT password FROM users WHERE username = ?", (username,))
        user = c.fetchone()
        conn.close()

        if user and check_password_hash(user[0], password):
            session['user'] = username
            return redirect('/homepage')
        else:
            return "Invalid credentials. Try again."

    return render_template('login.html')


@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect('/login')

# -------------------- MAIN ROUTES --------------------

@app.route('/homepage')
def homepage():
    if 'user' not in session:
        return redirect('/login')
    return render_template('homepage.html')

@app.route('/choose')
def choose():
    if 'user' not in session:
        return redirect('/login')
    return render_template('Details.html')

# -------------------- Fertilizer Recommendation Logic --------------------

def recommend_fertilizer(crop, soil, temperature, humidity, nitrogen, phosphorous, potassium):
    crop = crop.lower()
    soil = soil.lower()

    if crop == 'rice':
        if soil == 'clayey' and nitrogen < 50:
            return 'Urea + Potash (MOP)'
        else:
            return 'NPK 20-20-0'
    
    elif crop == 'wheat':
        if phosphorous < 50:
            return 'DAP'
        else:
            return 'Balanced NPK'
    
    elif crop == 'maize':
        return 'NPK 20-20-0 + Compost'

    elif crop == 'cotton':
        if potassium < 50:
            return 'MOP + Zinc Sulphate'
        else:
            return 'Balanced Fertilizer'

    else:
        return 'Balanced NPK'


@app.route('/predict', methods=['POST'])
def prediction():
    try:
        if 'user' not in session:
            return redirect('/login')

        temperature = float(request.form['temperature'])
        humidity = float(request.form['humidity'])
        moisture = float(request.form['moisture'])
        soil = request.form['soil']
        nitrogen = float(request.form['nitrogen'])
        phosphorous = float(request.form['phosphorous'])
        potassium = float(request.form['potassium'])

        soil_encoded = soil_encoder.transform([soil])[0]
        features = np.array([[temperature, humidity, moisture, soil_encoded, nitrogen, potassium, phosphorous]])

        # Predict probabilities for all crops
        probs = model.predict_proba(features)[0]

        # Get top 5 crop indices (descending order)
        top_indices = probs.argsort()[::-1][:5]

        # Get crop names and their scores
        top_crops = crop_encoder.inverse_transform(top_indices)
        top_crops_with_scores = [(crop, round(probs[i] * 100, 2)) for crop, i in zip(top_crops, top_indices)]
        # Recommend fertilizer for each of the top 5 crops
        crop_fertilizer_pairs = []
        for crop in top_crops:
            fert = recommend_fertilizer(crop, soil, temperature, humidity, nitrogen, phosphorous, potassium)
            score = round(probs[crop_encoder.transform([crop])[0]] * 100, 2)
            crop_fertilizer_pairs.append((crop, score, fert))

        # Pass this list to the template
        return render_template('fertiliser_result.html', crop_fertilizer_pairs=crop_fertilizer_pairs)


    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"Something went wrong: {e}", 500

@app.route('/goback')
def goback():
    return redirect('/homepage')

if __name__ == '__main__':
    app.run(debug=True)
