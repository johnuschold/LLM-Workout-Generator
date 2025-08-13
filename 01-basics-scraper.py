import time
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from tqdm import tqdm

# Setup headless browser
options = webdriver.ChromeOptions()
options.add_argument('--headless')
options.add_argument('--disable-gpu')
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

muscle_groups = [
    "biceps",
    "chest",
    "quads",
    "traps",
    "triceps",
    "shoulders",
    "lats",
    "hamstrings",
    "glutes",
    "forearms",
    "calves",
    "abdominals",
    "lower-back",
    "obliques",
    "upper-pectoralis",
    "mid-and-lower-chest",
    "hands",
    "long-head-bicep",
    "short-head-bicep",
    "lower-abdominals",
    "upper-abdominals",
    "soleus",
    "tibialis",
    "gastrocnemius",
    "wrist-extensors",
    "wrist-flexors",
    "gluteus-medius",
    "gluteus-maximus",
    "medial-hamstrings",
    "lateral-hamstrings",
    "lateral-deltoid",
    "anterior-deltoid",
    "posterior-deltoid",
    "long-head-tricep",
    "lateral-head-triceps",
    "medial-head-triceps",
    "upper-traps",
    "lower-traps",
    "inner-thigh",
    "inner-quadriceps",
    "outer-quadricep",
    "rectus-femoris"
]

  # Expand as needed
gender = "male"

all_data = []

for muscle in tqdm(muscle_groups):
    page = 1
    while True:
        if page == 1:
            url = f"https://musclewiki.com/exercises/{gender}/{muscle}"
        else:
            url = f"https://musclewiki.com/exercises/{gender}/{muscle}/{page}"
        print(f"Scraping page {page} for {muscle} at {url}")
        driver.get(url)


        try:
            # Wait for exercises or timeout after 5 seconds
            WebDriverWait(driver, 5).until(
                EC.presence_of_all_elements_located((By.XPATH, '//h2[contains(@class, "text-xl")]'))
            )
            titles = driver.find_elements(By.XPATH, '//h2[contains(@class, "text-xl")]')
            if not titles:
                print(f"No exercises found on page {page}, stopping.")
                break

            # Collect all links on the current page
            links = []
            for h2 in titles:
                try:
                    parent = h2.find_element(By.XPATH, "./ancestor::a")
                    href = parent.get_attribute("href")
                    if href:
                        links.append((h2.text.strip(), href))
                except:
                    pass

            print(f"Found {len(links)} exercises on page {page}")

            # Scrape each exercise page
            for name, link in links:
                try:
                    driver.get(link)
                    WebDriverWait(driver, 10).until(
                        EC.presence_of_element_located((By.TAG_NAME, "body"))
                    )
                    try:
                        instr_elem = driver.find_element(
                            By.XPATH,
                            '//div[contains(@class, "Instructions_instructions__container__content")]'
                        )
                        instructions = instr_elem.text.strip()
                    except:
                        instructions = ""

                    all_data.append({
                        "Muscle Group": muscle,
                        "Exercise Name": name,
                        "Instructions": instructions,
                        "Exercise URL": link
                    })
                    print(f"  ✔ {name}")
                except Exception as e:
                    print(f"  ❌ Failed scraping {name}: {e}")

            page += 1
            time.sleep(1)  # polite pause before next page

        except Exception:
            print(f"No exercises found or page {page} doesn't exist. Stopping.")
            break

driver.quit()

# Save to CSV
df = pd.DataFrame(all_data)
df.to_csv("musclewiki_paginated_exercises.csv", index=False)
print("\n✅ Done! Saved to musclewiki_paginated_exercises.csv")
