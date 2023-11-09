from datetime import timedelta

import pandas as pd


def gameplay_analysis(activity: pd.DataFrame) -> pd.DataFrame:
    activity['event_date'] = pd.to_datetime(activity['event_date'])

    # Finding the first login date for each player
    first_login = activity.groupby('player_id')['event_date'].min().reset_index()
    first_login.columns = ['player_id', 'first_login_date']

    # Merging the first login date with the original activity data
    merged_data = pd.merge(activity, first_login, on='player_id')

    # Finding players who logged in the day after their first login
    merged_data['next_day'] = merged_data['first_login_date'] + timedelta(days=1)
    players_logged_next_day = merged_data[
        (merged_data['event_date'] == merged_data['next_day'])
    ].drop_duplicates(subset='player_id')

    # Calculating the fraction
    total_players = activity['player_id'].nunique()
    players_logged_next_day_count = len(players_logged_next_day)
    fraction = players_logged_next_day_count / total_players

    fraction_rounded = round(fraction, 2)

    result_df = pd.DataFrame({'fraction': [fraction_rounded]})

    return result_df
