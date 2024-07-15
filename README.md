# Myntra Extension
The Myntra Extension provides users with recommendations for similar fashion items from Myntra based on the clothing items currently visible on their webpage.

## Components

- *Frontend*: 
  - popup.html provides the basic structure of the extension.
  - popup.js extracts the current URL of the page and dynamically updates the content to add links redirecting to similar items on Myntra.
- *Web Scraping*: 
  - The URL of the current page is used to scrape images with the help of BeautifulSoup and Selenium.
  - Scraped images are saved in the scrapped_images folder.
- *Recommendation System*: 
  - Uses Keras and cosine similarity to generate image embeddings.
  - Compares these embeddings against the Myntra database to find visually similar items.
  - Displays links to similar items within the extension.

## Requirements

- Python 3.7+
- The following Python packages:

  plaintext
  Flask
  flask-cors
  selenium
  webdriver-manager
  beautifulsoup4
  requests
  numpy
  pandas
  Pillow
  tensorflow
  scikit-learn

## Setting Up the Flask Server

To run the backend server for the extension, follow these steps:

### 1. **Clone the Repository**

If you haven't already cloned the repository, do so using:

bash
git clone https://github.com/bhumika-1127/Myntra-Extension.git
```
```bash
cd Myntra-Extension


### 2. **Install Requirements**
bash
pip install -r requirements.txt

### 3. **Run Flask Server**
bash
python app.py
```

## Loading the Chrome Extension

1. *Open Chrome* and navigate to chrome://extensions/.

2. *Enable Developer Mode*:
   - Toggle the "Developer mode" switch located in the top right corner of the Extensions page.

3. *Load Unpacked Extension*:
   - Click on the "Load unpacked" button.
   - In the file dialog that opens, select the Myntra-Extension directory.

4. *Verify the Extension*:
   - The extension should now appear in your list of installed extensions.
   - Ensure that the extension's icon is visible in the Chrome toolbar.

5. *Use the Extension*:
   - Visit a webpage with fashion images.
   - Click on the extension icon in the Chrome toolbar.
   - The extension will scrape images from the current page and display links to similar items on Myntra, Ensure setting the server on (app.py).
