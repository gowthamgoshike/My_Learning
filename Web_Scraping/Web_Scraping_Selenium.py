import time
from selenium import webdriver


driver = webdriver.Chrome() 

# STEP 1: GO TO THE SITE
print("Robot is going to the website.")
driver.get("http://quotes.toscrape.com/js/")

# STEP 2: WAIT

print("Robot is waiting for data to appear.")
time.sleep(5)

# STEP 3: GRAB DATA

print("Robot is grabbing the data...")
quote = driver.find_element("class name", "quote").text

print("SUCCESS! HERE IS THE DATA")
print(quote)

driver.quit()