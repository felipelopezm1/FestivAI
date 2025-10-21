import os
import random
import argparse
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from bs4 import BeautifulSoup

def download_image(url, save_path, session):
    response = session.get(url, stream=True)
    if response.status_code == 200:
        with open(save_path, 'wb') as file:
            for chunk in response.iter_content(1024):
                file.write(chunk)
        print(f"Downloaded: {save_path}")
    else:
        print(f"Failed to download image from {url}")

def scrape_met_museum_image(page_url, item_id, save_directory):
    session = requests.Session()
    retry = Retry(connect=3, backoff_factor=0.5)
    adapter = HTTPAdapter(max_retries=retry)
    session.mount('http://', adapter)
    session.mount('https://', adapter)

    response = session.get(page_url)
    if response.status_code != 200:
        print(f"Failed to access {page_url}")
        return

    soup = BeautifulSoup(response.text, 'html.parser')

    # Find the main image on the page (this selector may need adjustment if the site changes)
    image_tag = soup.find('img', class_='artwork__image')
    if not image_tag:
        print(f"No image found on {page_url}")
        return

    image_url = image_tag['src']
    if not image_url.startswith('http'):
        # Make the URL absolute if it's relative
        image_url = f"https://www.metmuseum.org{image_url}"

    # Extract a suitable filename from the URL
    image_name = str(item_id) + '.jpg'
    save_path = os.path.join(save_directory, image_name)

    # Download the image
    download_image(image_url, save_path, session)

def main():
    parser = argparse.ArgumentParser(description="Download random samples of artwork images from the Metropolitan Museum of Art website.")
    parser.add_argument('--start_id', type=int, default=1, help="The starting ID for the range of artwork IDs.")
    parser.add_argument('--end_id', type=int, default=100000, help="The ending ID for the range of artwork IDs.")
    parser.add_argument('--num_samples', type=int, required=True, help="The number of random samples to download.")
    parser.add_argument('--save_directory', type=str, default="class-datasets/met_images", help="The directory to save the downloaded images.")
    args = parser.parse_args()

    base_url = "https://www.metmuseum.org/art/collection/search/"
    save_directory = args.save_directory

    # Create the directory if it doesn't exist
    os.makedirs(save_directory, exist_ok=True)

    # Generate a random sample of IDs
    random_ids = random.sample(range(args.start_id, args.end_id + 1), args.num_samples)

    for item_id in random_ids:
        page_url = f"{base_url}{item_id}"
        print(f"Scraping {page_url}")
        scrape_met_museum_image(page_url, item_id, save_directory)

if __name__ == "__main__":
    main()
