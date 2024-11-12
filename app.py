from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing import image
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from flask import Flask, request, jsonify
import os
import json
import requests
from io import BytesIO
import base64
import dropbox
from PIL import Image
from dotenv import load_dotenv

# Set to CPU-only mode to avoid GPU issues if not supported
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
app = Flask(__name__)
load_dotenv()
# Paths and settings
#FEATURES_FILE = 'features.txt'  # File to store extracted features
#UPLOAD_DIR = 'uploads/'  # Directory for uploaded images
# Dropbox app credentials
CLIENT_ID_DROPBOX = os.getenv('CLIENT_ID_DROPBOX')
CLIENT_SECRET_DROPBOX = os.getenv('CLIENT_SECRET_DROPBOX')
REFRESH_TOKEN_DROPBOX = os.getenv('REFRESH_TOKEN_DROPBOX')
# Dropbox settings

FEATURES_FILE_PATH = '/Feature_Storage_Unit/features.txt'  # Path in Dropbox
# Load pre-trained model
model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Salesforce API credentials
SALESFORCE_IMAGE_API = os.getenv('SALESFORCE_IMAGE_API')
CLIENT_ID = os.getenv('CLIENT_ID')
CLIENT_SECRET = os.getenv('CLIENT_SECRET')
REFRESH_TOKEN = os.getenv('REFRESH_TOKEN')

# Helper to resize images for lower memory use
def resize_image(image_bytes, max_size=(224, 224)):
    img = Image.open(BytesIO(image_bytes))
    img.thumbnail(max_size)
    img_byte_arr = BytesIO()
    img.save(img_byte_arr, format=img.format)
    return img_byte_arr.getvalue()

def refresh_dropbox_access_token():
    global new_access_token_DBOX
    url = "https://api.dropbox.com/oauth2/token"
    payload = {
        'grant_type': 'refresh_token',
        'client_id': CLIENT_ID_DROPBOX,
        'client_secret': CLIENT_SECRET_DROPBOX,
        'refresh_token': REFRESH_TOKEN_DROPBOX
    }
    response = requests.post(url, data=payload)
    if response.status_code == 200:
        new_access_token = response.json().get('access_token')
        print("New access token obtained: DROPBOX")
        new_access_token_DBOX = new_access_token
        return new_access_token
    else:
        print(f"Failed to refresh access token: {response.text}")
        return None
    
def get_new_access_token():
    url = "https://login.salesforce.com/services/oauth2/token"
    payload = {
        'grant_type': 'refresh_token',
        'client_id': CLIENT_ID,
        'client_secret': CLIENT_SECRET,
        'refresh_token': REFRESH_TOKEN
    }
    response = requests.post(url, data=payload)
    if response.status_code == 200:
        access_token = response.json().get('access_token')
        print("New access token obtained.")
        return access_token
    else:
        print(f"Failed to refresh access token: {response.text}")
        return None

# Function to fetch images with metadata from Salesforce via REST API
def fetch_images_from_salesforce():
    access_token = get_new_access_token()
    if not access_token:
        print("Unable to retrieve access token.")
        return []

    headers = {
        'Authorization': f'Bearer {access_token}',
        'Content-Type': 'application/json'
    }
    
    response = requests.get(SALESFORCE_IMAGE_API, headers=headers)
    if response.status_code == 200:
        images_metadata = response.json()
        print("Successfully fetched images from Salesforce")
        return images_metadata
    else:
        print(f"Failed to fetch images from Salesforce. Status Code: {response.status_code}")
        return []

# Function to extract features from an image using CNN
def extract_features_from_image_bytes(image_bytes):
    img = image.load_img(BytesIO(image_bytes), target_size=(224, 224))
    img_data = image.img_to_array(img)
    img_data = np.expand_dims(img_data, axis=0)
    img_data = preprocess_input(img_data)
    features = model.predict(img_data)
    return np.array(features.flatten())

# Dropbox functions
def upload_features_to_dropbox(features_data):
    DROPBOX_ACCESS_TOKEN = new_access_token_DBOX
    dbx = dropbox.Dropbox(DROPBOX_ACCESS_TOKEN)

    features_json = json.dumps(features_data)
    try:
        dbx.files_upload(features_json.encode(), FEATURES_FILE_PATH, mode=dropbox.files.WriteMode.overwrite)
        print("Successfully uploaded features.txt to Dropbox.")
    except dropbox.exceptions.ApiError as e:
        print(f"Error uploading features to Dropbox: {e}")

# DownLoad_Feature_from_dropbox
def download_features_from_dropbox():
    DROPBOX_ACCESS_TOKEN = new_access_token_DBOX
    dbx = dropbox.Dropbox(DROPBOX_ACCESS_TOKEN)
    try:
        metadata, res = dbx.files_download(FEATURES_FILE_PATH)
        features_data = json.loads(res.content.decode())
        print("Successfully downloaded features.txt from Dropbox.")
        return features_data
    except dropbox.exceptions.ApiError as e:
        # Check if the error is due to the file not existing
        if isinstance(e.error, dropbox.files.DownloadError) and e.error.is_path() and e.error.get_path().is_not_found():
            print("features.txt not found in Dropbox. Creating a new file.")
            # Initialize with an empty dictionary or any default feature data
            features_data = {}
            upload_features_to_dropbox(features_data)  # Create the file in Dropbox
            return features_data
        else:
            print(f"Error downloading file from Dropbox: {e}")
        return {}
# Load features from file or compute them if not present
def load_features():
    print("Loading features from Dropbox...")
    features = download_features_from_dropbox()
    if not features:
        print("No features found in Dropbox. Computing and saving new features.")
        features = compute_and_save_features()
    return features

def synchronize_features_with_salesforce():
    # Load existing features from the file
    features = load_features()

    # Fetch current images from Salesforce
    images_metadata = fetch_images_from_salesforce()

    # Create a set of current image identifiers from Salesforce
    current_image_ids = {f"{img['recordId']}_{img['fileName']}_{img['contentVersionId']}" for img in images_metadata}

    # Identify orphaned entries in features
    orphaned_keys = [key for key in features if key not in current_image_ids]

    # Remove orphaned entries
    for key in orphaned_keys:
        del features[key]
        print(f"Removed orphaned feature entry for {key}")

    # Save the updated features file
    
    upload_features_to_dropbox(features)
    print("Features file synchronized with Salesforce images.")



# Compute and save features, renaming images
def compute_and_save_features():
    features_data = {}
    images_metadata = fetch_images_from_salesforce()
    
    if not images_metadata:
        print("No images found to compute features.")
        return features_data
    
    for img_meta in images_metadata:
        record_id = img_meta['recordId']
        original_name = img_meta['fileName']
        content_version_id = img_meta['contentVersionId']
        base64_img = img_meta['imageData']
        
        # Decode the image and generate a new name
        image_bytes = resize_image(base64.b64decode(base64_img)) 
        
        renamed_image = f"{record_id}_{original_name}_{content_version_id}"
        
        # Extract and store features
        features = extract_features_from_image_bytes(image_bytes)
        features_data[renamed_image] = features.tolist()
        print(f"Computed features for {renamed_image}")

    # Save features to file
    upload_features_to_dropbox(features_data)
    
    return features_data

# Update features with any new images from Salesforce
def update_features():
    refresh_dropbox_access_token()
    global features_data
    features_data = load_features()

    # Fetch images from Salesforce to see if there are new entries
    images_metadata = fetch_images_from_salesforce()
    
    # Check if new images need feature extraction
    for img_meta in images_metadata:
        record_id = img_meta['recordId']
        original_name = img_meta['fileName']
        content_version_id = img_meta['contentVersionId']
        renamed_image = f"{record_id}_{original_name}_{content_version_id}"
        
        # If the image is new, extract and save its features
        if renamed_image not in features_data:
            image_bytes =resize_image(base64.b64decode(img_meta['imageData']))
            features_data[renamed_image] = extract_features_from_image_bytes(image_bytes).tolist()

    # Save updated features
    upload_features_to_dropbox(features_data)
    print("Updated features saved to Dropbox")
    synchronize_features_with_salesforce()
    features_data=load_features()
# Function to find best matches based on cosine similarity
def find_best_matches(object_image_bytes, threshold=0.15986160418804682):
    object_features = extract_features_from_image_bytes(object_image_bytes)
    matches = []

    for filename, repo_features in features_data.items():
        repo_features = np.array(repo_features)
        score = cosine_similarity([object_features], [repo_features])[0][0]
        if score >= threshold:
            matches.append((filename, score))

    matches = sorted(matches, key=lambda x: x[1], reverse=True)
    return matches
# API endpoint to update features
@app.route('/update_features', methods=['GET'])
def trigger_feature_update():
    
    update_features()  # Manually update features
    return jsonify({'status': 'Features updated successfully'}), 200
# API endpoint to upload an image and find matches
@app.route('/match-object', methods=['POST'])
def match_object():
    #update_features()
    
    if 'image' not in request.files:
        return jsonify({'error': 'Image file not provided'}), 400
    
    uploaded_file = request.files['image']
    object_image_bytes = uploaded_file.read()

    matches = find_best_matches(object_image_bytes)

    if matches:
        return jsonify({
            'matches': [{'image': filename, 'score': float(score)} for filename, score in matches]
        }), 200
    else:
        return jsonify({'error': 'No matching images found'}), 404

if __name__ == '__main__':
    #os.makedirs(UPLOAD_DIR, exist_ok=True)
    port = int(os.environ.get("PORT", 5000))
    update_features()  # Ensure features are updated on startup
    app.run(host='0.0.0.0', port=port)
