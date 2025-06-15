# ⚽ Premier League Match Predictor
A web application that predicts the result of a Premier League match between two teams.
## 📌 Project Overview
## data scraping
1. this script scrapes Premier League match data for the last two seasons from fbref.com.
2. For each team, it collects match results (Scores & Fixtures) and shooting stats,
3. merges them on the match date, filters for Premier League matches, and adds season/team info.
4. Finally, all team data is combined and saved to a CSV file named 'matches.csv'.
## model training
1. Loaded and cleaned match data.
2. Created features for model training including venue, time, opponent, and performance-based rolling stats.
3. Trained a Random Forest model to predict win/loss.
4. Evaluated prediction quality using accuracy and precision.
5. Merged predictions to simulate matches between teams.
6. Set up structure to compare mutual predictions in head-to-head matches.
## Installation
-JupyerLab
-Python 3.8+
-Python packages
  -pandas
  -selenium
  -BeautifulSoup
  -scikit-learn
  -fast-api
  -uvicorn
  -pydantic
## To start fastapi backend server
1. Go to backend directory in your command prompt
2. Run this command: uvicorn main:app --reload
## 🚀 How to Run
### 1. Clone the repository
### 2. Install dependencies
### 3. Run the backend server
### 4. Open the frontend
