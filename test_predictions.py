from app import predict_winner, matches, batting_stats, bowling_stats

def parse_match(lines):
    match_data = {}
    for line in lines:
        if '=' in line and not line.startswith('#'):
            key, value = line.split('=', 1)
            key = key.strip()
            value = eval(value.strip())  # Using eval since values are quoted strings
            
            # Handle new venue cases by mapping to nearest known venue
            if key == 'venue' and value not in matches['venue'].unique():
                known_venues = matches['venue'].unique()
                if 'Stadium' in value:
                    # Try to find a similar stadium
                    similar_venues = [v for v in known_venues if 'Stadium' in v]
                    if similar_venues:
                        value = similar_venues[0]
                    else:
                        value = known_venues[0]
            
            match_data[key] = value
    return match_data

def test_predictions():
    print("\nTesting Match Predictions for 2025 Season")
    print("=" * 50)
    
    correct_predictions = 0
    total_matches = 0
    
    with open('test_matches.txt', 'r') as f:
        matches_text = f.read()
    
    # Split into individual matches
    matches = matches_text.split('\n\n')
    
    for match in matches:
        if not match.strip():
            continue
            
        match_data = parse_match(match.split('\n'))
        if not match_data:
            continue
            
        total_matches += 1
        
        try:
            predicted_winner, win_probability = predict_winner(
                match_data['season'],
                match_data['venue'],
                match_data['team1'],
                match_data['team2'],
                match_data['toss_winner'],
                match_data['toss_decision']
            )
            
            # Get team stats
            team1_stats = batting_stats[batting_stats['batting_team'] == match_data['team1']].iloc[0]
            team2_stats = batting_stats[batting_stats['batting_team'] == match_data['team2']].iloc[0]
            
            # Print match details and prediction
            print(f"\nMatch {total_matches}:")
            print(f"Venue: {match_data['venue']}")
            print(f"Teams: {match_data['team1']} vs {match_data['team2']}")
            print(f"Form Analysis:")
            print(f"- {match_data['team1']}: {team1_stats['avg_runs']:.1f} runs/match")
            print(f"- {match_data['team2']}: {team2_stats['avg_runs']:.1f} runs/match")
            print(f"\nActual Winner: {match_data['winner']}")
            print(f"Predicted Winner: {predicted_winner}")
            print(f"Prediction Confidence: {win_probability:.1f}%")
            
            if predicted_winner == match_data['winner']:
                correct_predictions += 1
                print("✓ Prediction Correct!")
            else:
                print("✗ Prediction Incorrect")
                
        except Exception as e:
            print(f"Error processing match: {str(e)}")
            continue
    
    # Calculate and display accuracy
    accuracy = (correct_predictions / total_matches) * 100 if total_matches > 0 else 0
    print("\nPrediction Results:")
    print("=" * 50)
    print(f"Total Matches: {total_matches}")
    print(f"Correct Predictions: {correct_predictions}")
    print(f"Accuracy: {accuracy:.1f}%")

if __name__ == "__main__":
    test_predictions()
