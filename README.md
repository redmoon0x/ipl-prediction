# IPL Match Winner Predictor

A machine learning-based application that predicts the winner of Indian Premier League (IPL) cricket matches using historical data and advanced statistical analysis.

## Features

- Predicts match winners based on multiple factors:
  - Historical team performance
  - Venue statistics
  - Toss decision impact
  - Team-specific metrics (batting/bowling averages)
- Provides win probability percentage
- Interactive command-line interface
- Detailed statistical analysis of predictions
- Support for current and future IPL seasons

## Requirements

- Python 3.x
- Required Python packages (install using `pip install -r requirements.txt`):
  - pandas
  - numpy
  - scikit-learn
  - os

## Dataset

The model uses two primary datasets:
- `matches.csv`: Historical IPL match data
- `deliveries.csv`: Ball-by-ball match details

Note: These files are not included in the repository due to size constraints.

## Installation

1. Clone the repository
2. Install required packages:
```bash
pip install -r requirements.txt
```

## Usage

Run the application:
```bash
python app.py
```

Follow the interactive prompts to:
1. Select teams from available options
2. Choose the match venue
3. Enter the season (year)
4. Specify toss winner and decision

The program will display:
- Match prediction with win probability
- Key factors influencing the prediction
- Detailed team statistics
- Net run rates comparison

## Model Details

The prediction model uses:
- Random Forest Classifier
- Feature engineering for team statistics
- Balanced class weights
- Standardized numerical features
- Categorical encoding for teams and venues

## Project Structure

- `app.py`: Main application file with ML model and UI
- `requirements.txt`: Required Python packages
- `test_predictions.py`: Test cases for prediction accuracy
- `.gitignore`: Git ignore file
- `README.md`: Project documentation

## Contributing

Feel free to open issues or submit pull requests to improve the project.

## License

This project is open source and available under the MIT License.
