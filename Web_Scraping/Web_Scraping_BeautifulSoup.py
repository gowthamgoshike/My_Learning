
import requests
from bs4 import BeautifulSoup
import pandas as pd
import re

# 1. THE REQUEST
url = "http://books.toscrape.com/"
response = requests.get(url)


if response.status_code == 200:
    print("Successfully connected to the website!")
    
    # 2. THE SOUP
    soup = BeautifulSoup(response.text, "html.parser")
    
    # 3. THE FIND
    books = soup.find_all("article", class_="product_pod")
    
    data_list = []

    # 4. THE LOOP
    for book in books:
        # Extract Title 
        title = book.find("h3").find("a")["title"]
        
        # Extract Price 
        price_text = book.find("p", class_="price_color").text
        clean_price_string = re.sub(r'[^\d.]', '', price_text)
        price = float(clean_price_string)
        
        # Extract Availability
        availability = book.find("p", class_="instock availability").text.strip()
        
        # Append to our list
        data_list.append({
            "Title": title,
            "Price": price,
            "Availability": availability
        })

    # 5. THE STORAGE
    df = pd.DataFrame(data_list)
    
    print("\nTop 5 rows of your new dataset:")
    print(df.head())
    
else:
    print("Failed to retrieve the webpage.")