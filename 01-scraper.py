import pandas as pd
import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.action_chains import ActionChains
from urllib.parse import urlparse
from tqdm import tqdm

# Setup headless browser
options = webdriver.ChromeOptions()
options.add_argument('--headless')
options.add_argument('--disable-gpu')
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

# Load existing CSV (adjust filename as needed)
df = pd.read_csv("musclewiki_paginated_exercises.csv")

# Add new columns to store extracted info
df["Video URL"] = ""
df["Instructions"] = ""
df["Detailed How-To"] = ""
df["Difficulty"] = ""
df["Tags"] = "" # Ensure Tags column exists

for idx, row in tqdm(df.iterrows()): # Using head(10) for testing, remove for full run
    url = row["Exercise URL"]
    print(f"🔎 Scraping: {url}")
    try:
        driver.get(url)
        WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.TAG_NAME, "body")))
        time.sleep(1) # Give a little more time for dynamic content to load

        # === Extract Difficulty (fallback: scan for known keywords) ===
        try:
            body_text = driver.find_element(By.TAG_NAME, "body").text
            for level in ["Beginner", "Novice", "Intermediate", "Advanced"]:
                if level in body_text:
                    df.at[idx, "Difficulty"] = level
                    break
            else:
                df.at[idx, "Difficulty"] = ""
        except:
            df.at[idx, "Difficulty"] = ""

        # === Extract Video URL ===
        try:
            video = driver.find_element(By.TAG_NAME, "video")
            source = video.find_element(By.TAG_NAME, "source")
            video_url = source.get_attribute("src")
            df.at[idx, "Video URL"] = video_url if video_url else ""
        except:
            df.at[idx, "Video URL"] = ""

        # === Extract Instructions (your 1st screenshot) ===
        try:
            instructions = driver.find_elements(By.XPATH, '//div[contains(@class, "border-gray-200") and contains(@class, "items-center")]')
            instruction_texts = [elem.text.strip() for elem in instructions if elem.text.strip()]
            df.at[idx, "Instructions"] = "\n".join(instruction_texts)
        except:
            df.at[idx, "Instructions"] = ""

        # === Extract Detailed How-To (Improved) ===
        try:
            # Find the main content area that contains both video and the how-to sections
            # This XPath targets the div that wraps the video and other content before the tags
            main_content_div = WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.XPATH, '//div[contains(@class, "lg:col-span-7") and contains(@class, "col-span-10")]'))
            )

            # Get all direct children of this main content div
            # We are looking for divs that contain the instructional text.
            # These often have padding classes like 'px-6 py-5' or similar.
            potential_howto_sections = main_content_div.find_elements(By.XPATH, './div[contains(@class, "px-6") or contains(@class, "py-5") or contains(@class, "prose")]')

            howto_text_parts = []
            for section in potential_howto_sections:
                # Check if the section contains text and is not part of the header/video container
                # and not the tags section.
                # You might need to adjust these conditions based on other pages.
                if section.text.strip() and "aspect-video" not in section.get_attribute("class") and "flex-wrap" not in section.get_attribute("class"):
                    # Check for common headings that might indicate the start of a new, unrelated section
                    # This helps to avoid picking up unrelated divs
                    if "Benefits" in section.text or "Tips" in section.text or "Variations" in section.text or "Muscles worked" in section.text:
                         # If it's a known instruction section, add its text
                        howto_text_parts.append(section.text.strip())
                    elif section.find_elements(By.XPATH, ".//h2"): # Look for h2 as a main section divider
                        # If a section starts with an h2, consider it a new part of how-to or a new logical section
                        howto_text_parts.append(section.text.strip())
                    elif not section.find_elements(By.XPATH, ".//div[contains(@class, 'aspect-w-16')]"): # Exclude video container again
                        # If it doesn't have a video, and has text, include it.
                        howto_text_parts.append(section.text.strip())


            # Filter out empty strings and join
            df.at[idx, "Detailed How-To"] = "\n\n".join(filter(None, howto_text_parts))

        except Exception as e:
            print(f"  ❌ Failed to extract Detailed How-To for {url}: {e}")
            df.at[idx, "Detailed How-To"] = ""

        # === Extract Tags ===
        try:
            tags_container = driver.find_element(By.XPATH,
                                                 '//div[contains(@class, "flex-wrap") and contains(@class, "rounded-md") and contains(@class, "bg-white")]')
            tags = tags_container.find_elements(By.TAG_NAME, "a")
            tag_texts = [tag.text.strip() for tag in tags if tag.text.strip()]
            df.at[idx, "Tags"] = ", ".join(tag_texts)
        except:
            df.at[idx, "Tags"] = ""

        print(f"  ✔ Done: {row['Exercise Name']}")

    except Exception as e:
        print(f"❌ Failed on {url}: {e}")
        continue

# Save updated CSV
df.to_csv("musclewiki_exercises_enriched.csv", index=False)
print("\n✅ All done! Saved to musclewiki_exercises_enriched.csv")

driver.quit()