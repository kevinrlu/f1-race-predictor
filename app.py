from flask import Flask, render_template
import logging
from datetime import date
import time

import fastf1
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

# Config
logging.getLogger('fastf1').setLevel(logging.ERROR)
fastf1.Cache.enable_cache('f1_cache')
YEAR = 2025

# Lap count per circuit (sourced from official F1 race data)
lap_counts = {
    'Australian Grand Prix': 58,
    'Chinese Grand Prix': 56,
    'Japanese Grand Prix': 53,
    'Bahrain Grand Prix': 57,
    'Saudi Arabian Grand Prix': 50,
    'Miami Grand Prix': 57,
    'Emilia Romagna Grand Prix': 63,
    'Monaco Grand Prix': 78,
    'Spanish Grand Prix': 66,
    'Canadian Grand Prix': 70,
    'Austrian Grand Prix': 71,
    'British Grand Prix': 52,
    'Belgian Grand Prix': 44,
    'Hungarian Grand Prix': 70,
    'Dutch Grand Prix': 72,
    'Italian Grand Prix': 53,
    'Azerbaijan Grand Prix': 51,
    'Singapore Grand Prix': 61,
    'United States Grand Prix': 56,
    'Mexican Grand Prix': 71,
    'Brazilian Grand Prix': 71,
    'Las Vegas Grand Prix': 50,
    'Qatar Grand Prix': 57,
    'Abu Dhabi Grand Prix': 55
}

app = Flask(__name__)


def fetch_times(gp: str, session: str) -> pd.Series:
    sess = fastf1.get_session(YEAR, gp, session)
    sess.load(laps=True, telemetry=False, weather=False, messages=False)
    grp = sess.laps.dropna(subset=['Driver', 'LapTime']).groupby('Driver')['LapTime']
    return (grp.min() if session == 'Q' else grp.sum()).dt.total_seconds()


def format_hms(seconds: float) -> str:
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{int(h)}:{int(m):02d}:{s:06.3f}"


@app.route('/')
def index():
    start = time.time()
    try:
        sched = fastf1.get_event_schedule(YEAR)
        sched['Date'] = pd.to_datetime(sched['EventDate']).dt.date
        races = sched[sched['RoundNumber'] > 0]
        today = date.today()
        past = races[races['Date'] < today]
        upcoming = races[races['Date'] >= today]

        if past.empty:
            run_time = f"{time.time()-start:.3f}"
            return render_template('index.html', error='No past races to train on.', run_time=run_time)

        trains = []
        for gp in past['EventName']:
            q = fetch_times(gp, 'Q').rename('Quali_s')
            r = fetch_times(gp, 'R').rename('Total_s')
            laps = lap_counts.get(gp, pd.NA)
            df = pd.concat([q, r], axis=1)
            df['Laps'] = laps
            trains.append(df)
        data = pd.concat(trains).dropna().reset_index()

        X = data[['Quali_s', 'Laps']]
        y = data['Total_s']
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
        model = LinearRegression().fit(Xtr, ytr)
        mae_val = mean_absolute_error(yte, model.predict(Xte))
        coef_quali, coef_laps = model.coef_

        if upcoming.empty:
            run_time = f"{time.time()-start:.3f}"
            return render_template('index.html', error='No upcoming races.', mae=f"{mae_val:.3f}", coef_quali=f"{coef_quali:.2f}", coef_laps=f"{coef_laps:.2f}", run_time=run_time)

        next_gp = upcoming.iloc[0]['EventName']
        laps_next = lap_counts.get(next_gp, pd.NA)
        quali = fetch_times(next_gp, 'Q').rename('Quali_s')
        X_next = pd.DataFrame({'Quali_s': quali, 'Laps': laps_next})
        pred_s = model.predict(X_next)

        preds_df = pd.DataFrame({'Driver': quali.index, 'Pred_s': pred_s})
        preds_df['PredictedTime'] = preds_df['Pred_s'].apply(format_hms)
        preds_df = preds_df.sort_values('Pred_s').reset_index(drop=True)
        preds_df['Pos'] = preds_df.index + 1
        preds = preds_df[['Pos', 'Driver', 'PredictedTime']].to_dict(orient='records')

        run_time = f"{time.time()-start:.3f}"
        return render_template('index.html', error=None, gp_name=next_gp,
                               mae=f"{mae_val:.3f}", coef_quali=f"{coef_quali:.2f}",
                               coef_laps=f"{coef_laps:.2f}", run_time=run_time,
                               preds=preds)
    except Exception as e:
        run_time = f"{time.time()-start:.3f}"
        return render_template('index.html', error=str(e), run_time=run_time)


if __name__ == '__main__':
    app.run(debug=True)