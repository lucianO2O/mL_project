import pandas as pd
import numpy as np

# read csv file
gamesData = pd.read_csv("games_march2025_full.csv", 
                        engine = 'python') # helps with crashes
# show all columns
pd.set_option('display.max_columns', None) 
# getting rid of any duplicate rows
gamesData = gamesData.drop_duplicates()
# get rid of any columns not needed in training
gamesData = gamesData.drop(columns = {"appid", "release_date", "price", 
                                      "dlc_count", "required_age", 
                                      "detailed_description", "about_the_game", 
                                      "short_description", "reviews", 
                                      "header_image", "website", "support_url",
                                      "support_email", "metacritic_url", "achievements",
                                      "recommendations", "notes", "supported_languages",
                                      "full_audio_languages", "packages", "developers",
                                      "publishers", "user_score", "score_rank",
                                      "estimated_owners", "average_playtime_forever",
                                      "average_playtime_2weeks", "median_playtime_forever",
                                      "median_playtime_2weeks", "peak_ccu", "positive",
                                      "negative", "pct_pos_recent", "screenshots", "movies", 
                                      "num_reviews_recent", "discount"})
# want to remove # of people who voted for tags and convert column to just string
gamesData["tags"] = gamesData["tags"].str.replace(r'[0-9:]', '', 
                                                  regex=True)
gamesData["tags"] = gamesData["tags"].str.rstrip()
# convert column to string
gamesData["tags"] = gamesData["tags"].astype("string")
print(gamesData["tags"].head())
print("Hello WOrld !")