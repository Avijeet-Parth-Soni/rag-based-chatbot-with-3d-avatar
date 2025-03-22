from flask import Flask, send_from_directory
from flask_cors import CORS
from flask_socketio import SocketIO
import subprocess

app = Flask(__name__, static_folder='.')
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")

speaking = False

@app.route('/')
def index():
    return send_from_directory('.', 'test3.html')
def home():
    return "Server is running!"

@app.route('/AvatarSample_A(1.0).vrm')
def serve_vrm():
    return send_from_directory('static', 'AvatarSample_A(1.0).vrm')

# Serve the background image
@app.route('/test_bg.jpg')
def serve_background():
    return send_from_directory('static', 'test_bg.jpg')

@app.route('/audio/<path:filename>')
def serve_audio(filename):
    return send_from_directory('audio', filename)

@socketio.on('request_audio')
def handle_request(data):
    print("Audio created")
    socketio.emit('data_response', {'text': data['emotion'], 'audioPath': './audio/audio.wav'})
    print("Play requested")

@socketio.on('audio_complete')
def handle_audio_complete():
    print("Audio play complete")
    args = ('del', r'C:\Users\Shivanshu\Desktop\hack\combining\working_girlfriend_emotional\avi_sl\audio\audio.wav')
    print("Audio deleted")
    subprocess.call('%s %s' % args, shell=True)
    socketio.emit('speaking_complete')
    print("End signal sent")


if __name__ == '__main__':
    try:
        import eventlet
        eventlet.monkey_patch()  # Ensures compatibility
        print("Running with Eventlet server.")
        socketio.run(app, host='127.0.0.1', port=5000)
    except ImportError:
        print("Eventlet not found. Falling back to default Flask server.")
        app.run(debug=True, host='127.0.0.1', port=5000)