import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import os

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def print_header():
    clear_screen()
    print("\n" + "="*50)
    print("IPL MATCH WINNER PREDICTOR")
    print("="*50 + "\n")

# Load data
matches = pd.read_csv("matches.csv")
deliveries = pd.read_csv("deliveries.csv")

# Calculate team statistics from deliveries data
def calculate_team_stats(deliveries, matches):
    # Calculate batting statistics
    batting_stats = deliveries.groupby(['match_id', 'batting_team']).agg({
        'total_runs': 'sum',
        'ball': 'count',
        'is_wicket': 'sum'
    }).reset_index()

    batting_stats = batting_stats.groupby('batting_team').agg({
        'total_runs': ['mean', 'std'],
        'ball': 'mean',
        'is_wicket': 'mean'
    }).round(2)

    batting_stats.columns = ['avg_runs', 'std_runs', 'avg_balls_faced', 'avg_wickets_lost']
    batting_stats = batting_stats.reset_index()

    # Calculate bowling statistics
    bowling_stats = deliveries.groupby(['match_id', 'bowling_team']).agg({
        'total_runs': 'sum',
        'is_wicket': 'sum'
    }).reset_index()

    bowling_stats = bowling_stats.groupby('bowling_team').agg({
        'total_runs': 'mean',
        'is_wicket': 'mean'
    }).round(2)

    bowling_stats.columns = ['avg_runs_conceded', 'avg_wickets_taken']
    bowling_stats = bowling_stats.reset_index()

    return batting_stats, bowling_stats

# Calculate team statistics
batting_stats, bowling_stats = calculate_team_stats(deliveries, matches)

# Prepare the main dataframe with additional features
df = matches[['season', 'venue', 'team1', 'team2', 'toss_winner', 'toss_decision', 'winner']].dropna()

# Add head-to-head records
def calculate_h2h_advantage(row):
    team1, team2 = row['team1'], row['team2']

    team1_wins = len(matches[(matches['team1'] == team1) & (matches['team2'] == team2) & (matches['winner'] == team1)]) + \
                 len(matches[(matches['team1'] == team2) & (matches['team2'] == team1) & (matches['winner'] == team1)])

    team2_wins = len(matches[(matches['team1'] == team1) & (matches['team2'] == team2) & (matches['winner'] == team2)]) + \
                 len(matches[(matches['team1'] == team2) & (matches['team2'] == team1) & (matches['winner'] == team2)])

    total_matches = team1_wins + team2_wins
    h2h_advantage = 0
    if total_matches > 0:
        if team1_wins > team2_wins:
            h2h_advantage = (team1_wins / total_matches) * 10  # Scale to 0-10
        elif team2_wins > team1_wins:
            h2h_advantage = -(team2_wins / total_matches) * 10  # Negative for team2 advantage

    return h2h_advantage

# Add home advantage
def calculate_home_advantage(row):
    venue, team1, team2 = row['venue'], row['team1'], row['team2']

    if venue in team1:
        return 5  # Team1 has home advantage
    elif venue in team2:
        return -5  # Team2 has home advantage
    return 0

# Calculate toss impact at each venue
def calculate_toss_impact(row):
    venue, toss_winner, toss_decision, winner = row['venue'], row['toss_winner'], row['toss_decision'], row['winner']

    # Check if toss winner also won the match
    toss_win_match_win = 1 if toss_winner == winner else 0

    # Check if batting first or second was advantageous at this venue
    batting_first_advantage = 1 if toss_decision == 'bat' else -1

    return toss_win_match_win, batting_first_advantage

# Calculate recent form (last 3 matches)
def calculate_recent_form(row):
    team1, team2 = row['team1'], row['team2']

    # Get all matches involving team1 and team2
    team1_matches = matches[(matches['team1'] == team1) | (matches['team2'] == team1)].sort_values('date', ascending=False)
    team2_matches = matches[(matches['team1'] == team2) | (matches['team2'] == team2)].sort_values('date', ascending=False)

    # Calculate recent win rate (last 3 matches)
    team1_recent_wins = 0
    team1_recent_matches = 0
    for _, match in team1_matches.head(3).iterrows():
        if pd.notna(match['winner']):
            team1_recent_matches += 1
            if match['winner'] == team1:
                team1_recent_wins += 1

    team2_recent_wins = 0
    team2_recent_matches = 0
    for _, match in team2_matches.head(3).iterrows():
        if pd.notna(match['winner']):
            team2_recent_matches += 1
            if match['winner'] == team2:
                team2_recent_wins += 1

    team1_form = team1_recent_wins / team1_recent_matches if team1_recent_matches > 0 else 0.5
    team2_form = team2_recent_wins / team2_recent_matches if team2_recent_matches > 0 else 0.5

    form_advantage = team1_form - team2_form
    return form_advantage

# Add these new features to the dataframe
df['h2h_advantage'] = df.apply(calculate_h2h_advantage, axis=1)
df['home_advantage'] = df.apply(calculate_home_advantage, axis=1)

# Add toss impact features
toss_features = df.apply(calculate_toss_impact, axis=1, result_type='expand')
df['toss_win_match_win'] = toss_features[0]
df['batting_first_advantage'] = toss_features[1]

# Add form advantage
df['form_advantage'] = df.apply(calculate_recent_form, axis=1)

# Add team statistics for both teams
def add_team_stats(row):
    team1_batting = batting_stats[batting_stats['batting_team'] == row['team1']].iloc[0]
    team2_batting = batting_stats[batting_stats['batting_team'] == row['team2']].iloc[0]
    team1_bowling = bowling_stats[bowling_stats['bowling_team'] == row['team1']].iloc[0]
    team2_bowling = bowling_stats[bowling_stats['bowling_team'] == row['team2']].iloc[0]

    return pd.Series({
        'team1_avg_runs': team1_batting['avg_runs'],
        'team1_std_runs': team1_batting['std_runs'],
        'team1_avg_wickets_lost': team1_batting['avg_wickets_lost'],
        'team1_avg_wickets_taken': team1_bowling['avg_wickets_taken'],
        'team1_avg_runs_conceded': team1_bowling['avg_runs_conceded'],
        'team2_avg_runs': team2_batting['avg_runs'],
        'team2_std_runs': team2_batting['std_runs'],
        'team2_avg_wickets_lost': team2_batting['avg_wickets_lost'],
        'team2_avg_wickets_taken': team2_bowling['avg_wickets_taken'],
        'team2_avg_runs_conceded': team2_bowling['avg_runs_conceded']
    })

# Add team statistics to the main dataframe
df = pd.concat([df, df.apply(add_team_stats, axis=1)], axis=1)

# Encode categorical features and scale numerical features
le_dict = {}
for col in ['season', 'venue', 'team1', 'team2', 'toss_winner', 'toss_decision', 'winner']:
    le_dict[col] = LabelEncoder()
    df[col] = le_dict[col].fit_transform(df[col])

scaler = StandardScaler()
numerical_cols = [col for col in df.columns if col not in ['season', 'venue', 'team1', 'team2', 'toss_winner', 'toss_decision', 'winner']]
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

# Features and target
X = df.drop('winner', axis=1)
y = df['winner']

# Print feature importance after model training
def print_feature_importance(model, feature_names):
    importances = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    })
    print("\nFeature Importances:")
    print(importances.sort_values('importance', ascending=False))

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train an improved model with more trees and better hyperparameters
model = RandomForestClassifier(
    n_estimators=1000,  # Increased from 500 for better stability
    random_state=42,
    class_weight='balanced',
    max_depth=15,       # Increased from 10 to capture more complex patterns
    min_samples_split=4,
    min_samples_leaf=2,
    bootstrap=True,     # Enable bootstrapping for better generalization
    max_features='sqrt' # Use sqrt of features for each tree to reduce overfitting
)

# Fit model with balanced dataset
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Suppress the initial model evaluation output
import sys
import os
original_stdout = sys.stdout
sys.stdout = open(os.devnull, 'w')
print("Accuracy:", accuracy)
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nModel Evaluation:")
print("================")
print(f"Accuracy: {accuracy:.2f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
sys.stdout = original_stdout

# Enhanced prediction function with better handling of edge cases and improved probability calculation
def predict_winner(season, venue, team1, team2, toss_winner, toss_decision):
    # Create input DataFrame
    input_df = pd.DataFrame({
        'season': [season],
        'venue': [venue],
        'team1': [team1],
        'team2': [team2],
        'toss_winner': [toss_winner],
        'toss_decision': [toss_decision]
    })

    try:
        # Add team statistics with error handling for unknown teams
        try:
            team1_batting = batting_stats[batting_stats['batting_team'] == team1].iloc[0]
            team2_batting = batting_stats[batting_stats['batting_team'] == team2].iloc[0]
            team1_bowling = bowling_stats[bowling_stats['bowling_team'] == team1].iloc[0]
            team2_bowling = bowling_stats[bowling_stats['bowling_team'] == team2].iloc[0]
        except (IndexError, KeyError) as e:
            # If team not found, use average stats as fallback
            print(f"Warning: Team stats not found. Using average values. Error: {str(e)}")
            avg_batting = batting_stats.mean()
            avg_bowling = bowling_stats.mean()

            # Create default stats for missing teams
            if team1 not in batting_stats['batting_team'].values:
                team1_batting = pd.Series({
                    'avg_runs': avg_batting['avg_runs'],
                    'std_runs': avg_batting['std_runs'],
                    'avg_wickets_lost': avg_batting['avg_wickets_lost']
                })
                team1_bowling = pd.Series({
                    'avg_wickets_taken': avg_bowling['avg_wickets_taken'],
                    'avg_runs_conceded': avg_bowling['avg_runs_conceded']
                })

            if team2 not in batting_stats['batting_team'].values:
                team2_batting = pd.Series({
                    'avg_runs': avg_batting['avg_runs'],
                    'std_runs': avg_batting['std_runs'],
                    'avg_wickets_lost': avg_batting['avg_wickets_lost']
                })
                team2_bowling = pd.Series({
                    'avg_wickets_taken': avg_bowling['avg_wickets_taken'],
                    'avg_runs_conceded': avg_bowling['avg_runs_conceded']
                })

        # Calculate head-to-head record (new feature)
        team1_wins = len(matches[(matches['team1'] == team1) & (matches['team2'] == team2) & (matches['winner'] == team1)]) + \
                     len(matches[(matches['team1'] == team2) & (matches['team2'] == team1) & (matches['winner'] == team1)])

        team2_wins = len(matches[(matches['team1'] == team1) & (matches['team2'] == team2) & (matches['winner'] == team2)]) + \
                     len(matches[(matches['team1'] == team2) & (matches['team2'] == team1) & (matches['winner'] == team2)])

        total_matches = team1_wins + team2_wins
        h2h_advantage = 0
        if total_matches > 0:
            if team1_wins > team2_wins:
                h2h_advantage = (team1_wins / total_matches) * 10  # Scale to 0-10
            elif team2_wins > team1_wins:
                h2h_advantage = -(team2_wins / total_matches) * 10  # Negative for team2 advantage

        # Calculate home advantage (new feature)
        home_advantage = 0
        if venue in team1:
            home_advantage = 5  # Team1 has home advantage
        elif venue in team2:
            home_advantage = -5  # Team2 has home advantage

        # Calculate toss impact at this venue
        venue_matches = matches[matches['venue'] == venue]
        toss_win_match_win = 0
        batting_first_advantage = 0

        if len(venue_matches) > 0:
            # Calculate how often toss winner wins at this venue
            toss_winners = venue_matches[venue_matches['toss_winner'] == venue_matches['winner']]
            toss_win_match_win = len(toss_winners) / len(venue_matches) * 10  # Scale to 0-10

            # Calculate batting first vs second advantage
            bat_first_wins = venue_matches[(venue_matches['toss_decision'] == 'bat') &
                                          (venue_matches['toss_winner'] == venue_matches['winner'])]
            field_first_wins = venue_matches[(venue_matches['toss_decision'] == 'field') &
                                           (venue_matches['toss_winner'] == venue_matches['winner'])]

            if len(bat_first_wins) > len(field_first_wins):
                batting_first_advantage = 5  # Batting first is advantageous
            elif len(field_first_wins) > len(bat_first_wins):
                batting_first_advantage = -5  # Batting second is advantageous

        # Calculate recent form (last 3 matches)
        team1_matches = matches[(matches['team1'] == team1) | (matches['team2'] == team1)].sort_values('date', ascending=False)
        team2_matches = matches[(matches['team1'] == team2) | (matches['team2'] == team2)].sort_values('date', ascending=False)

        team1_recent_wins = 0
        team1_recent_matches = 0
        for _, match in team1_matches.head(3).iterrows():
            if pd.notna(match['winner']):
                team1_recent_matches += 1
                if match['winner'] == team1:
                    team1_recent_wins += 1

        team2_recent_wins = 0
        team2_recent_matches = 0
        for _, match in team2_matches.head(3).iterrows():
            if pd.notna(match['winner']):
                team2_recent_matches += 1
                if match['winner'] == team2:
                    team2_recent_wins += 1

        team1_form = team1_recent_wins / team1_recent_matches if team1_recent_matches > 0 else 0.5
        team2_form = team2_recent_wins / team2_recent_matches if team2_recent_matches > 0 else 0.5

        form_advantage = (team1_form - team2_form) * 10  # Scale to similar range as other features

        # Add team statistics with additional features
        stats_dict = {
            'team1_avg_runs': float(team1_batting['avg_runs']),
            'team1_std_runs': float(team1_batting['std_runs']),
            'team1_avg_wickets_lost': float(team1_batting['avg_wickets_lost']),
            'team1_avg_wickets_taken': float(team1_bowling['avg_wickets_taken']),
            'team1_avg_runs_conceded': float(team1_bowling['avg_runs_conceded']),
            'team2_avg_runs': float(team2_batting['avg_runs']),
            'team2_std_runs': float(team2_batting['std_runs']),
            'team2_avg_wickets_lost': float(team2_batting['avg_wickets_lost']),
            'team2_avg_wickets_taken': float(team2_bowling['avg_wickets_taken']),
            'team2_avg_runs_conceded': float(team2_bowling['avg_runs_conceded']),
            'h2h_advantage': h2h_advantage,
            'home_advantage': home_advantage,
            'toss_win_match_win': toss_win_match_win,
            'batting_first_advantage': batting_first_advantage,
            'form_advantage': form_advantage
        }

        for key, value in stats_dict.items():
            input_df[key] = value

        # Handle future seasons with better error handling
        known_seasons = matches['season'].unique()

        # Convert seasons to standardized format (YYYY)
        def standardize_season(s):
            if '/' in str(s):
                return str(s).split('/')[0]
            return str(s)

        # Get max season in standardized format
        max_season = max(int(standardize_season(s)) for s in known_seasons)
        current_season = int(standardize_season(season))

        # For encoding future seasons, use an incremental approach
        if current_season > max_season:
            season_encoder = LabelEncoder()
            all_seasons = np.append(known_seasons, [str(year) for year in range(max_season + 1, current_season + 1)])
            season_encoder.fit(all_seasons)
            input_df['season'] = season_encoder.transform([str(current_season)])
        else:
            input_df['season'] = le_dict['season'].transform([str(current_season)])

        # Handle unknown venues with better error handling
        for col in ['venue', 'team1', 'team2', 'toss_winner', 'toss_decision']:
            try:
                input_df[col] = le_dict[col].transform(input_df[col])
            except ValueError as e:
                print(f"Warning: Unknown value in {col}. Using fallback. Error: {str(e)}")
                # For unknown values, use the most common value as fallback
                most_common_idx = np.argmax(np.bincount(le_dict[col].transform(matches[col])))
                input_df[col] = most_common_idx

        # Scale numerical features with error handling for new features
        try:
            # Only scale the columns that were in the original training data
            common_cols = [col for col in numerical_cols if col in input_df.columns]
            input_df[common_cols] = scaler.transform(input_df[common_cols])

            # For new columns, standardize them manually
            new_cols = [col for col in input_df.columns if col not in numerical_cols and col not in ['season', 'venue', 'team1', 'team2', 'toss_winner', 'toss_decision']]
            for col in new_cols:
                input_df[col] = (input_df[col] - input_df[col].mean()) / (input_df[col].std() if input_df[col].std() > 0 else 1)
        except Exception as e:
            print(f"Warning: Error scaling features. Using unscaled values. Error: {str(e)}")

        # Make prediction
        # Drop columns that weren't in the training data to avoid errors
        prediction_cols = [col for col in input_df.columns if col in X.columns]
        missing_cols = [col for col in X.columns if col not in input_df.columns]

        # Add missing columns with default values
        for col in missing_cols:
            input_df[col] = 0

        # Ensure columns are in the same order as during training
        input_df = input_df[X.columns]

        # Make prediction
        prediction = model.predict(input_df)

        # Get winner name
        winner = le_dict['winner'].inverse_transform(prediction)[0]

        # Calculate adjusted win probability with improved calibration
        proba = model.predict_proba(input_df)
        base_probability = np.max(proba) * 100

        # Calculate team strength metrics
        team1_nrr = float(team1_batting['avg_runs']) - float(team1_bowling['avg_runs_conceded'])
        team2_nrr = float(team2_batting['avg_runs']) - float(team2_bowling['avg_runs_conceded'])
        stats_diff = team1_nrr - team2_nrr

        # Adjust probability based on multiple factors
        win_probability = base_probability

        # 1. Adjust for team strength difference
        if abs(stats_diff) < 5:  # Teams are closely matched
            # Regress extreme probabilities toward the mean
            if win_probability > 60:
                win_probability = 60 + (win_probability - 60) * 0.5
            elif win_probability < 40:
                win_probability = 40 + (win_probability - 40) * 0.5
        else:  # Clear statistical advantage
            # Enhance the probability in the direction of the stronger team
            if (stats_diff > 0 and winner == team1) or (stats_diff < 0 and winner == team2):
                # Prediction aligns with team strength, increase confidence slightly
                win_probability = min(win_probability * 1.1, 95)

        # 2. Adjust for head-to-head record
        if h2h_advantage != 0:
            if (h2h_advantage > 0 and winner == team1) or (h2h_advantage < 0 and winner == team2):
                # Prediction aligns with head-to-head history
                win_probability = min(win_probability + abs(h2h_advantage), 95)
            else:
                # Prediction contradicts head-to-head history
                win_probability = max(win_probability - abs(h2h_advantage) * 0.5, 55)

        # 3. Adjust for home advantage
        if home_advantage != 0:
            if (home_advantage > 0 and winner == team1) or (home_advantage < 0 and winner == team2):
                # Prediction aligns with home advantage
                win_probability = min(win_probability + abs(home_advantage) * 0.5, 95)

        # 4. Toss advantage adjustment based on venue statistics
        toss_impact = 0
        if toss_winner == winner:
            # Base toss advantage
            toss_impact += 3

            # Add venue-specific toss impact
            toss_impact += toss_win_match_win * 0.3

            # Add batting/fielding first advantage based on venue
            if (toss_decision == 'bat' and batting_first_advantage > 0) or \
               (toss_decision == 'field' and batting_first_advantage < 0):
                toss_impact += abs(batting_first_advantage) * 0.5

            win_probability = min(win_probability + toss_impact, 95)

        # 5. Recent form adjustment
        if (form_advantage > 0 and winner == team1) or (form_advantage < 0 and winner == team2):
            # Prediction aligns with recent form
            form_impact = abs(form_advantage) * 0.4  # Scale down the impact
            win_probability = min(win_probability + form_impact, 95)

        # 5. Final calibration to avoid extreme probabilities
        if win_probability > 90:
            win_probability = 90 + (win_probability - 90) * 0.5
        elif win_probability < 55:
            win_probability = 55 + (win_probability - 55) * 0.5

        return winner, round(win_probability, 2)

    except Exception as e:
        print(f"Prediction error: {str(e)}")
        # More intelligent fallback based on team stats if available
        try:
            team1_batting = batting_stats[batting_stats['batting_team'] == team1].iloc[0]
            team2_batting = batting_stats[batting_stats['batting_team'] == team2].iloc[0]

            if float(team1_batting['avg_runs']) > float(team2_batting['avg_runs']):
                return team1, 60.0
            else:
                return team2, 60.0
        except:
            # Ultimate fallback if everything else fails
            return team1, 50.0

def get_user_input(matches):
    print("\nEnter Match Details:")
    print("-" * 20)

    # Get unique values
    unique_venues = sorted(matches['venue'].unique())
    unique_teams = sorted(matches['team1'].unique())

    # Print available teams
    print("\nAvailable Teams:")
    for i, team in enumerate(unique_teams, 1):
        print(f"{i}. {team}")

    # Get team selections
    while True:
        try:
            print("\nSelect teams by number:")
            team1_idx = int(input("Team 1 number: ")) - 1
            team2_idx = int(input("Team 2 number: ")) - 1

            if team1_idx == team2_idx:
                print("Error: Please select different teams")
                continue

            team1 = unique_teams[team1_idx]
            team2 = unique_teams[team2_idx]
            break
        except (ValueError, IndexError):
            print("Error: Please enter valid team numbers")

    # Print available venues
    print("\nAvailable Venues:")
    for i, venue in enumerate(unique_venues, 1):
        print(f"{i}. {venue}")

    # Get venue selection
    while True:
        try:
            venue_idx = int(input("\nSelect venue number: ")) - 1
            venue = unique_venues[venue_idx]
            break
        except (ValueError, IndexError):
            print("Error: Please enter a valid venue number")

    # Get season
    while True:
        try:
            season = input("\nEnter season (YYYY): ")
            if season.isdigit() and len(season) == 4:
                break
            print("Error: Please enter a valid year (YYYY)")
        except ValueError:
            print("Error: Please enter a valid year")

    # Get toss details
    while True:
        try:
            print(f"\nToss Winner:")
            print(f"1. {team1}")
            print(f"2. {team2}")
            toss_choice = int(input("Select toss winner (1/2): "))
            if toss_choice in [1, 2]:
                toss_winner = team1 if toss_choice == 1 else team2
                break
            print("Error: Please enter 1 or 2")
        except ValueError:
            print("Error: Please enter 1 or 2")

    while True:
        print("\nToss Decision:")
        print("1. bat")
        print("2. field")
        try:
            decision_choice = int(input("Select decision (1/2): "))
            if decision_choice in [1, 2]:
                toss_decision = 'bat' if decision_choice == 1 else 'field'
                break
            print("Error: Please enter 1 or 2")
        except ValueError:
            print("Error: Please enter 1 or 2")

    return season, venue, team1, team2, toss_winner, toss_decision

def display_prediction(season, venue, team1, team2, toss_winner, toss_decision, predicted_winner, win_probability, batting_stats, bowling_stats):
    clear_screen()
    print("\nMatch Details:")
    print("-" * 50)
    print(f"Season: {season}")
    print(f"Teams: {team1} vs {team2}")
    print(f"Venue: {venue}")
    print(f"Toss: {toss_winner} chose to {toss_decision}")

    print("\nPrediction Result:")
    print("-" * 50)
    print(f"Predicted Winner: {predicted_winner}")
    print(f"Win Probability: {win_probability:.1f}%")

    # Calculate confidence level description
    confidence_desc = "Very High" if win_probability >= 80 else \
                     "High" if win_probability >= 70 else \
                     "Moderate" if win_probability >= 60 else \
                     "Low" if win_probability >= 50 else "Very Low"
    print(f"Confidence Level: {confidence_desc}")

    print("\nKey Factors Influencing Prediction:")
    print("-" * 50)

    # Home advantage analysis
    home_advantage = None
    if venue in team1:  # Check if it's home ground for team1
        home_advantage = team1
        print(f"- {venue} is {team1}'s home ground (advantage)")
    elif venue in team2:  # Check if it's home ground for team2
        home_advantage = team2
        print(f"- {venue} is {team2}'s home ground (advantage)")
    else:
        print(f"- Neutral venue (no home advantage)")

    # Calculate team strength metrics
    team1_batting_avg = float(batting_stats[batting_stats['batting_team'] == team1]['avg_runs'].iloc[0])
    team1_bowling_avg = float(bowling_stats[bowling_stats['bowling_team'] == team1]['avg_runs_conceded'].iloc[0])
    team2_batting_avg = float(batting_stats[batting_stats['batting_team'] == team2]['avg_runs'].iloc[0])
    team2_bowling_avg = float(bowling_stats[bowling_stats['bowling_team'] == team2]['avg_runs_conceded'].iloc[0])

    team1_nrr = team1_batting_avg - team1_bowling_avg
    team2_nrr = team2_batting_avg - team2_bowling_avg

    print(f"- {team1} Net Run Rate: {team1_nrr:.1f}")
    print(f"- {team2} Net Run Rate: {team2_nrr:.1f}")

    # Team form analysis
    stronger_team = team1 if team1_nrr > team2_nrr else team2
    nrr_diff = abs(team1_nrr - team2_nrr)
    if nrr_diff > 10:
        form_advantage = "significant"
    elif nrr_diff > 5:
        form_advantage = "moderate"
    elif nrr_diff > 2:
        form_advantage = "slight"
    else:
        form_advantage = "minimal"

    if nrr_diff > 2:  # Only mention if there's a meaningful difference
        print(f"- {stronger_team} has {form_advantage} statistical advantage")
    else:
        print(f"- Teams are statistically well-matched")

    # Head-to-head analysis
    team1_wins = len(matches[(matches['team1'] == team1) & (matches['team2'] == team2) & (matches['winner'] == team1)]) + \
                 len(matches[(matches['team1'] == team2) & (matches['team2'] == team1) & (matches['winner'] == team1)])

    team2_wins = len(matches[(matches['team1'] == team1) & (matches['team2'] == team2) & (matches['winner'] == team2)]) + \
                 len(matches[(matches['team1'] == team2) & (matches['team2'] == team1) & (matches['winner'] == team2)])

    total_h2h = team1_wins + team2_wins

    if total_h2h > 0:
        print(f"- Head-to-head record: {team1} {team1_wins}-{team2_wins} {team2}")
        if team1_wins > team2_wins and team1_wins / total_h2h > 0.6:
            print(f"  {team1} has historical advantage over {team2}")
        elif team2_wins > team1_wins and team2_wins / total_h2h > 0.6:
            print(f"  {team2} has historical advantage over {team1}")

    # Toss impact analysis
    toss_advantage = "batting first" if toss_decision == "bat" else "chasing"
    print(f"- {toss_winner} won the toss and chose {toss_advantage}")

    # Calculate venue-specific toss impact
    venue_matches = matches[matches['venue'] == venue]
    if len(venue_matches) > 0:
        toss_winners = venue_matches[venue_matches['toss_winner'] == venue_matches['winner']]
        toss_win_rate = len(toss_winners) / len(venue_matches) * 100
        print(f"- At {venue}, toss winners win {toss_win_rate:.1f}% of matches")

        # Batting vs fielding first analysis
        bat_first_wins = venue_matches[(venue_matches['toss_decision'] == 'bat') &
                                      (venue_matches['toss_winner'] == venue_matches['winner'])]
        field_first_wins = venue_matches[(venue_matches['toss_decision'] == 'field') &
                                       (venue_matches['toss_winner'] == venue_matches['winner'])]

        if len(bat_first_wins) > len(field_first_wins):
            print(f"- Teams batting first have an advantage at this venue")
        elif len(field_first_wins) > len(bat_first_wins):
            print(f"- Teams chasing have an advantage at this venue")

    # Recent form analysis
    team1_matches = matches[(matches['team1'] == team1) | (matches['team2'] == team1)].sort_values('date', ascending=False)
    team2_matches = matches[(matches['team1'] == team2) | (matches['team2'] == team2)].sort_values('date', ascending=False)

    team1_recent_wins = 0
    team1_recent_matches = 0
    for _, match in team1_matches.head(3).iterrows():
        if pd.notna(match['winner']):
            team1_recent_matches += 1
            if match['winner'] == team1:
                team1_recent_wins += 1

    team2_recent_wins = 0
    team2_recent_matches = 0
    for _, match in team2_matches.head(3).iterrows():
        if pd.notna(match['winner']):
            team2_recent_matches += 1
            if match['winner'] == team2:
                team2_recent_wins += 1

    if team1_recent_matches > 0 and team2_recent_matches > 0:
        team1_form = team1_recent_wins / team1_recent_matches * 100
        team2_form = team2_recent_wins / team2_recent_matches * 100
        print(f"- Recent form (last 3 matches): {team1}: {team1_form:.1f}%, {team2}: {team2_form:.1f}%")

    # Alignment analysis
    factors_favoring_winner = 0
    if predicted_winner == stronger_team:
        factors_favoring_winner += 1
    if home_advantage and predicted_winner == home_advantage:
        factors_favoring_winner += 1
    if total_h2h > 5:  # Only consider if enough h2h matches
        if predicted_winner == team1 and team1_wins > team2_wins:
            factors_favoring_winner += 1
        elif predicted_winner == team2 and team2_wins > team1_wins:
            factors_favoring_winner += 1
    if predicted_winner == toss_winner:
        factors_favoring_winner += 1

    if factors_favoring_winner >= 3:
        print(f"- Multiple factors strongly favor {predicted_winner}")
    elif factors_favoring_winner == 0:
        print(f"- Prediction is based on subtle factors not listed above")

    print("\nDetailed Team Statistics:")
    print("-" * 50)

    def print_team_stats(team):
        batting_avg = batting_stats[batting_stats['batting_team'] == team]['avg_runs'].iloc[0]
        batting_std = batting_stats[batting_stats['batting_team'] == team]['std_runs'].iloc[0]
        bowling_wickets = bowling_stats[bowling_stats['bowling_team'] == team]['avg_wickets_taken'].iloc[0]
        bowling_runs = bowling_stats[bowling_stats['bowling_team'] == team]['avg_runs_conceded'].iloc[0]

        consistency = "High" if float(batting_std) < 30 else "Medium" if float(batting_std) < 50 else "Low"

        print(f"\n{team}:")
        print(f"- Average Runs Scored: {float(batting_avg):.1f}")
        print(f"- Batting Consistency: {consistency} (std: {float(batting_std):.1f})")
        print(f"- Average Wickets Taken: {float(bowling_wickets):.1f}")
        print(f"- Average Runs Conceded: {float(bowling_runs):.1f}")
        print(f"- Net Run Rate: {float(batting_avg) - float(bowling_runs):.1f}")

    print_team_stats(team1)
    print_team_stats(team2)

    # Add prediction confidence explanation
    print("\nConfidence Analysis:")
    print("-" * 50)
    if win_probability >= 70:
        print("High confidence prediction based on strong statistical evidence")
    elif win_probability >= 60:
        print("Moderate confidence prediction with good statistical support")
    else:
        print("This match appears to be closely contested with multiple factors in play")
        print("The prediction has lower confidence due to the competitive balance between teams")

def main():
    while True:
        print_header()
        season, venue, team1, team2, toss_winner, toss_decision = get_user_input(matches)

        try:
            predicted_winner, win_probability = predict_winner(season, venue, team1, team2, toss_winner, toss_decision)
            display_prediction(season, venue, team1, team2, toss_winner, toss_decision, predicted_winner, win_probability, batting_stats, bowling_stats)

            choice = input("\nMake another prediction? (y/n): ").lower()
            if choice != 'y':
                break

        except Exception as e:
            print(f"\nError in prediction: {str(e)}")
            input("\nPress Enter to continue...")

if __name__ == "__main__":
    main()
