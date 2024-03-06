from flask import request, jsonify
from firebase_admin import credentials, auth, initialize_app

# Initialize Firebase Admin SDK
cred = credentials.Certificate("mergersFirebaseAdminKey.json")
initialize_app(cred)

def verify_firebase_token(token):
    try:
        decoded_token = auth.verify_id_token(token)
        return decoded_token
    except Exception as e:
        return None

def firebase_auth_required(f):
    def wrapper(*args, **kwargs):
        token = request.headers.get('Authorization')
        if not token:
            return jsonify({"error": "Authorization token required"}), 401
        user = verify_firebase_token(token)
        if not user:
            return jsonify({"error": "Invalid or expired token"}), 403
        return f(*args, **kwargs)
    wrapper.__name__ = f.__name__
    return wrapper