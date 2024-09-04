from gevent import monkey
monkey.patch_all()

import os
from flask import Flask, render_template, request, jsonify, session
from flask_socketio import SocketIO, emit, join_room
from flask_session import Session
from flask_sqlalchemy import SQLAlchemy
from flask_wtf.csrf import CSRFProtect, generate_csrf
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from dotenv import load_dotenv
from flask_cors import CORS
import pymysql
import logging
from datetime import timedelta
import dramatiq
from dramatiq.brokers.redis import RedisBroker
import sys
from functools import wraps
from urllib.parse import quote_plus
from dramatiq.middleware import AgeLimit, TimeLimit, Retries

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
load_dotenv()

db = SQLAlchemy()
csrf = CSRFProtect()
limiter = Limiter(key_func=get_remote_address, default_limits=["10 per minute"], storage_uri="redis://localhost:6379/1")

socketio = SocketIO(cors_allowed_origins="*", message_queue='redis://localhost:6379/0', async_mode='gevent')

def dynamic_rate_limit():
    from database.database import Database
    def decorator(f):
        @wraps(f)
        def wrapped(*args, **kwargs):
            session_id = session.get('session_id')
            if session_id:
                db = Database()
                try:
                    task_status = db.fetchone("SELECT status FROM task_status WHERE session_id = %s", (session_id,))
                    if task_status and task_status['status'] == 'in_progress':
                        limit = '1 per 10 seconds'
                    else:
                        limit = '1000 per minute'
                finally:
                    db.close()
            else:
                limit = '1000 per minute'
            return limiter.limit(limit)(f)(*args, **kwargs)

        return wrapped

    return decorator

def create_app():
    app = Flask(__name__, static_folder='static')
    app.config['SECRET_KEY'] = os.getenv('SECRET_KEY')
    app.config['SESSION_TYPE'] = 'sqlalchemy'
    app.config['SESSION_SQLALCHEMY'] = db
    db_user = os.getenv('DB_USER')
    db_password = quote_plus(os.getenv('DB_PASSWORD'))
    db_host = os.getenv('DB_HOST')
    db_port = os.getenv('DB_PORT')
    db_name = os.getenv('DB_NAME')
    app.config['SQLALCHEMY_DATABASE_URI'] = f"mysql+pymysql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(minutes=5)
    app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'
    app.config['SESSION_COOKIE_SECURE'] = False
    app.config['SESSION_USE_SIGNER'] = True
    app.config['RATELIMIT_STORAGE_URL'] = 'redis://localhost:6379/1'

    db.init_app(app)
    Session(app)

    csrf.init_app(app)
    socketio.init_app(app, message_queue='redis://localhost:6379/0', manage_session=True)
    limiter.init_app(app)
    CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True)

    logging.basicConfig(level=logging.INFO)
    logging.info("Flask app created and configured")

    return app

redis_broker = RedisBroker(
    url="redis://localhost:6379/0",
    middleware=[
        AgeLimit(max_age=3600000),  # 1 hour
        TimeLimit(time_limit=3600000),  # 1 hour
        Retries(max_retries=3)
    ]
)
dramatiq.set_broker(redis_broker)

app = create_app()

from models.predict_new_coin import predict_and_save_async

@app.route('/')
def home():
    return render_template('index.html')

@app.before_request
def ensure_session():
    if 'session_id' not in session:
        session['session_id'] = os.urandom(24).hex()
        app.logger.info(f"New session created with ID: {session['session_id']}")
    else:
        app.logger.info(f"Existing session ID: {session['session_id']}")

@socketio.on('connect')
def on_connect():
    app.logger.info("WebSocket connection attempt")
    sid = session.get('session_id')
    if sid:
        join_room(sid)
        app.logger.info(f"Client connected with session_id: {sid}")
        emit('connected', {'message': f'Connected with session_id: {sid}'}, room=sid)
    else:
        emit('error', {'message': 'Session ID is missing'})
        app.logger.error("No session_id found for connected client")

@app.route('/predict', methods=['POST'])
@csrf.exempt
@dynamic_rate_limit()
def predict():
    from database.database import Database
    try:
        data = request.get_json()
        if not data:
            raise ValueError("No data received")

        symbol = data.get('symbol').strip().upper()
        sid = session.get('session_id')
        csrf_token = generate_csrf()
        app.logger.info(f"Received symbol: {symbol} with session_id: {sid} and csrf_token: {csrf_token}")

        db = Database()
        try:
            task_status = db.fetchone("SELECT status FROM task_status WHERE session_id = %s", (sid,))
            app.logger.info(f"Task status for session_id {sid}: {task_status}")
            if task_status and task_status['status'] == 'in_progress':
                app.logger.info("Prediction already in progress for this session.")
                return jsonify({'status': 'error', 'message': 'Prediction already in progress.'}), 429

            db.execute("REPLACE INTO task_status (session_id, status) VALUES (%s, %s)", (sid, 'in_progress'))
            app.logger.info(f"Task status set to in_progress for session_id: {sid}")
        finally:
            db.close()

        app.logger.info("Enqueuing the prediction task.")
        predict_and_save_async.send(symbol, sid, csrf_token)
        app.logger.info("Prediction task enqueued.")

        return jsonify({'status': 'pending', 'message': 'Prediction has been enqueued.'})
    except Exception as e:
        db = Database()
        try:
            db.execute("REPLACE INTO task_status (session_id, status) VALUES (%s, %s)", (sid, 'failed'))
            app.logger.info(f"Task status set to failed for session_id: {sid} due to error: {e}")
        finally:
            db.close()
        app.logger.error(f"Error in /predict endpoint: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/restore_state', methods=['GET'])
@csrf.exempt
@limiter.limit("30 per minute")
def restore_state():
    from database.database import Database
    try:
        if 'session_id' not in session:
            session['session_id'] = os.urandom(24).hex()
            app.logger.info(f"New session created with ID: {session['session_id']}")
        session_id = session.get('session_id')
        app.logger.info(f"Session ID: {session_id}")
        csrf_token = generate_csrf()

        db = Database()
        try:
            task_status = db.fetchone("SELECT status FROM task_status WHERE session_id = %s", (session_id,))
            if task_status:
                app.logger.info(f"Task status for session_id {session_id}: {task_status}")
                if task_status['status'] == 'in_progress':
                    app.logger.info("Prediction already in progress for this session.")
                    return jsonify({
                        'status': 'success',
                        'task_status': task_status['status'],
                        'csrf_token': csrf_token,
                    })
                elif task_status['status'] == 'done':
                    last_result = session.get('last_result', None)
                    return jsonify({
                        'status': 'success',
                        'task_status': task_status['status'],
                        'csrf_token': csrf_token,
                        'last_result': last_result
                    })
                else:
                    return jsonify({
                        'status': 'success',
                        'task_status': task_status['status'],
                        'csrf_token': csrf_token
                    })
            else:
                app.logger.info(f"No existing task status for session_id: {session_id}")
                return jsonify({
                    'status': 'success',
                    'task_status': 'no_task',
                    'csrf_token': csrf_token
                })
        finally:
            db.close()
    except Exception as e:
        app.logger.error(f"Error in /restore_state endpoint: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@socketio.on('progress')
def handle_progress(data):
    sid = session.get('session_id')
    if sid:
        emit('progress', data, room=sid)
        app.logger.info(f"Progress event emitted to room: {sid} with data: {data}")
    else:
        app.logger.error("No session_id found for emitting progress")

@socketio.on('prediction_result')
def handle_prediction_result(data):
    from database.database import Database
    app.logger.info("Received prediction result, resetting in_progress state")
    db = Database()
    try:
        db.execute("REPLACE INTO task_status (session_id, status) VALUES (%s, %s)", (session['session_id'], 'done'))
    finally:
        db.close()
    session['last_result'] = data
    session.modified = True
    sid = session.get('session_id')
    if sid:
        emit('prediction_result', data, room=sid)
        app.logger.info(f"Prediction result event emitted to room: {sid} with data: {data}")
    else:
        app.logger.error("No session_id found for emitting prediction_result")

@socketio.on('session_reset')
def handle_session_reset(data):
    from database.database import Database
    app.logger.info("Received session reset, updating session state")
    db = Database()
    try:
        db.execute("REPLACE INTO task_status (session_id, status) VALUES (%s, %s)", (data['sid'], 'done'))
    finally:
        db.close()
    session['csrf_token'] = data['csrf_token']
    session.modified = True
    sid = data['sid']
    if sid:
        emit('session_reset_ack', {'status': 'success'}, room=sid)
        app.logger.info(f"Session reset ack event emitted to room: {sid} with data: {'status': 'success'}")
    else:
        app.logger.error("No sid found for emitting session_reset_ack")

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=8000, debug=True)
