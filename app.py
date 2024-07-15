from flask import Flask, request, jsonify
from flask_cors import CORS
from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
import requests
import os
import time
import numpy as np
import pandas as pd
from PIL import Image
from keras.applications.resnet50 import preprocess_input, ResNet50
from keras.preprocessing import image
from keras.models import Sequential
from keras.layers import GlobalMaxPooling2D
from sklearn.metrics.pairwise import pairwise_distances
import json

app = Flask(__name__)
CORS(app)

options = webdriver.ChromeOptions()
options.add_argument('--headless')
options.add_argument('--disable-gpu')
options.add_argument('--no-sandbox')
options.add_argument('--disable-dev-shm-usage')

def scroll_to_bottom(driver):
    last_height = driver.execute_script("return document.body.scrollHeight")
    while True:
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(2)
        new_height = driver.execute_script("return document.body.scrollHeight")
        if new_height == last_height:
            break
        last_height = new_height

def clear_images_folder():
    folder = 'scraped_images'
    if os.path.exists(folder):
        for filename in os.listdir(folder):
            file_path = os.path.join(folder, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    os.rmdir(file_path)
            except Exception as e:
                print(f'Failed to delete {file_path}. Reason: {e}')

def scrape_current_page_images(driver, max_images=10):
    clear_images_folder()
    scroll_to_bottom(driver)
    WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.TAG_NAME, 'body')))
    soup = BeautifulSoup(driver.page_source, 'html.parser')
    images = soup.find_all('img')
    folder_path = os.path.join(os.getcwd(), 'scraped_images')
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Created folder at: {folder_path}")
    else:
        print(f"Folder already exists at: {folder_path}")

    downloaded_count = 0
    for index, img in enumerate(images):
        if downloaded_count >= max_images:
            print(f"Reached the maximum limit of {max_images} images.")
            break
        img_url = img.get('src') or img.get('data-src') or img.get('data-lazy-src')
        if img_url and img_url.startswith(('http', 'https')):
            try:
                img_data = requests.get(img_url).content
                with open(f'scraped_images/image_{index + 1}.jpg', 'wb') as handler:
                    handler.write(img_data)
                downloaded_count += 1
                print(f"Downloaded image_{index + 1}.jpg")
            except Exception as e:
                print(f"Failed to download image {img_url}: {e}")

def load_image_from_path(image_path):
    try:
        img = Image.open(image_path).convert('RGB')
        img = img.resize((224, 224))  # Resize image to fit ResNet50 input size
        img_array = np.array(img)
        return img, img_array
    except Exception as e:
        print(f"Error loading image from {image_path}: {e}")
        return None, None
    
def embeddings(model, img_array):
    x = np.expand_dims(img_array, axis=0)
    x = preprocess_input(x)
    return model.predict(x).reshape(-1)

def calculate_cosine_similarity(feature_vector, df_embs):
    feature_vector = feature_vector.reshape(1, -1)
    
    # Calculate cosine similarity
    cosine_similarity = 1 - pairwise_distances(df_embs, feature_vector, metric='cosine').flatten()
    
    return cosine_similarity

def get_recommendations(feature_vector, df_embs, top_n=5):
    # Calculate cosine similarity scores
    sim_scores = calculate_cosine_similarity(feature_vector, df_embs)
    
    sim_scores = sim_scores.flatten()

    # Sort the similarity scores in descending order
    sim_scores_indices = np.argsort(sim_scores)[::-1]
    
    idx_rec = sim_scores_indices[:top_n]
    idx_sim = sim_scores[idx_rec]
    
    return idx_rec, idx_sim

def recommender_folder(folder_path, df, df_embs, model, top_n=5):
    recommendations = []
    # Iterate through each image in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(folder_path, filename)
            
            img, img_array = load_image_from_path(image_path)
            if img is None:
                continue
                
            ref_features = embeddings(model, img_array)
            
            idx_rec, idx_sim = get_recommendations(ref_features, df_embs, top_n=top_n)

            # Collect recommendations
            for i, row_idx in enumerate(idx_rec):
                product_title = df.iloc[row_idx]['title']
                product_url = df.iloc[row_idx]['link']
                recommendations.append({
                    "title": product_title,
                    "link": product_url
                })
        else:
            continue
    print(recommendations)
    return recommendations

@app.route('/scrape', methods=['POST'])
def scrape():
    data = request.get_json()
    url = data['url']
    driver = webdriver.Chrome(service=ChromeService(ChromeDriverManager().install()), options=options)
    driver.get(url)
    MAX_IMAGES_TO_DOWNLOAD = 10
    scrape_current_page_images(driver, max_images=MAX_IMAGES_TO_DOWNLOAD)
    driver.quit()

    df = pd.read_csv("recommendation/processed_myntraDataset.csv")
    df_embs = pd.read_csv('recommendation/image_embeddings.csv', header=None).values

    # Initialize the ResNet50 model
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = False
    model = Sequential([
        base_model,
        GlobalMaxPooling2D()
    ])
    
    folder_path = './scraped_images'  

    # Get recommendations
    recommendations = recommender_folder(folder_path, df, df_embs, model, top_n=1)
    
    return jsonify(recommendations)

if __name__ == '__main__':
    app.run(debug=True)

