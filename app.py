#!/usr/bin/env python3
"""Lightweight F1 race‑time predictor (2025).

Trains once on every completed round (Quali + Race) and publishes a
position‑sorted table for the next Grand Prix as soon as FastF1 releases
qualifying timing data.
"""
# ─── IMPORTS ────────────────────────────────────────────────────────────────────
from flask import Flask, render_template
import logging, time
from datetime import date

import fastf1
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GroupShuffleSplit

# ─── CONFIG ────────────────────────────────────────────────────────────────────
logging.getLogger("fastf1").setLevel(logging.ERROR)
fastf1.Cache.enable_cache("f1_cache")      # disk cache
YEAR = 2025

lap_counts = {  # unchanged
    "Australian Grand Prix": 57,  "Chinese Grand Prix": 56,   "Japanese Grand Prix": 53,
    "Bahrain Grand Prix": 57,     "Saudi Arabian Grand Prix": 50,  "Miami Grand Prix": 57,
    "Emilia Romagna Grand Prix": 63, "Monaco Grand Prix": 78, "Spanish Grand Prix": 66,
    "Canadian Grand Prix": 70,    "Austrian Grand Prix": 71,  "British Grand Prix": 52,
    "Belgian Grand Prix": 44,     "Hungarian Grand Prix": 70, "Dutch Grand Prix": 72,
    "Italian Grand Prix": 53,     "Azerbaijan Grand Prix": 51,"Singapore Grand Prix": 61,
    "United States Grand Prix": 56,"Mexican Grand Prix": 71,  "Brazilian Grand Prix": 71,
    "Las Vegas Grand Prix": 50,   "Qatar Grand Prix": 57,     "Abu Dhabi Grand Prix": 55,
}

app = Flask(__name__)

# ─── HELPERS ────────────────────────────────────────────────────────────────────
def fetch_times(year: int, gp: str, session: str) -> pd.Series:
    """Fastest lap (Q) or total time (R) per driver."""
    ses = fastf1.get_session(year, gp, session)
    ses.load(laps=True, telemetry=False, weather=False, messages=False)
    grp = ses.laps.dropna(subset=["Driver", "LapTime"]).groupby("Driver")["LapTime"]
    return (grp.min() if session == "Q" else grp.sum()).dt.total_seconds()

def driver_code_to_fullname(session) -> dict[str, str]:
    return {info["Abbreviation"]: info["FullName"]
            for drv in session.drivers
            if (info := session.get_driver(drv)) is not None}

def format_hms(seconds: float) -> str:
    h, rem = divmod(seconds, 3600); m, s = divmod(rem, 60)
    return f"{int(h)}:{int(m):02d}:{s:06.3f}"

# ─── ROUTES ─────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    t0 = time.time()
    try:
        # 1 ⇢ schedule split -----------------------------------------------------
        sched = fastf1.get_event_schedule(YEAR)
        sched["Date"] = pd.to_datetime(sched["EventDate"]).dt.date
        races      = sched[sched["RoundNumber"] > 0]
        today      = date.today()
        past       = races[races["Date"] < today]
        upcoming   = races[races["Date"] >= today]

        if past.empty:
            return render_template("index.html",
                                   gp_name="No upcoming race",
                                   upcoming_date="No upcoming race",
                                   error="No past races to train on.",
                                   run_time=f"{time.time()-t0:.3f}")

        # 2 ⇢ build training frame ---------------------------------------------
        frames = []
        for gp in past["EventName"]:
            q = fetch_times(YEAR, gp, "Q").rename("Quali_s")
            r = fetch_times(YEAR, gp, "R").rename("Total_s")
            df = pd.concat([q, r], axis=1)
            df["Laps"]  = lap_counts.get(gp, pd.NA)
            df["Event"] = gp
            frames.append(df)

        data = pd.concat(frames).reset_index().rename(columns={"index": "Driver"})
        data = data.dropna(subset=["Quali_s", "Total_s"])

        # 3 ⇢ leakage‑safe split ----------------------------------------------
        X = data[["Quali_s", "Laps"]]
        y = data["Total_s"]
        gsplit = GroupShuffleSplit(test_size=0.2, random_state=42)
        train_idx, test_idx = next(gsplit.split(X, groups=data["Event"]))
        Xtr, Xte, ytr, yte = X.iloc[train_idx], X.iloc[test_idx], y.iloc[train_idx], y.iloc[test_idx]

        # 4 ⇢ model ------------------------------------------------------------
        model = LinearRegression().fit(Xtr, ytr)
        mae_val = mean_absolute_error(yte, model.predict(Xte))

        # 5 ⇢ no upcoming race? ------------------------------------------------
        if upcoming.empty:
            return render_template("index.html",
                                   gp_name="[no data]",
                                   upcoming_date="[Predictions appear only after the qualifying session finishes and FastF1 publishes timing data]",
                                   error=None,
                                   mae=f"{mae_val:.3f}",
                                   run_time=f"{time.time()-t0:.3f}",
                                   preds=[])

        # 6 ⇢ prepare upcoming GP ----------------------------------------------
        next_gp   = upcoming.iloc[0]["EventName"]
        next_date = upcoming.iloc[0]["Date"]
        laps_next = lap_counts.get(next_gp, pd.NA)

        q_sess = fastf1.get_session(YEAR, next_gp, "Q")
        q_sess.load(laps=True, telemetry=False, weather=False, messages=False)
        name_map = driver_code_to_fullname(q_sess)
        team_map = (q_sess.laps.dropna(subset=["Driver", "Team"])
                    .loc[:, ["Driver", "Team"]]
                    .drop_duplicates()
                    .set_index("Driver")["Team"].to_dict())

        quali_next = fetch_times(YEAR, next_gp, "Q").rename("Quali_s")
        X_next     = pd.DataFrame({"Quali_s": quali_next, "Laps": laps_next})
        preds_sec  = model.predict(X_next)

        preds_df = (pd.DataFrame({"DriverCode": X_next.index, "Pred_s": preds_sec})
                    .assign(PredictedTime=lambda d: d["Pred_s"].apply(format_hms),
                            Driver=lambda d: d["DriverCode"].map(name_map),
                            Team=lambda d: d["DriverCode"].map(team_map))
                    .sort_values("Pred_s")
                    .reset_index(drop=True))
        preds_df["Pos"] = preds_df.index + 1
        preds = preds_df[["Pos", "Driver", "Team", "PredictedTime"]].to_dict("records")

        # 7 ⇢ render ------------------------------------------------------------
        return render_template("index.html",
                               gp_name=next_gp,
                               upcoming_date=next_date.isoformat(),
                               mae=f"{mae_val:.3f}",
                               run_time=f"{time.time()-t0:.3f}",
                               preds=preds,
                               error=None)

    except Exception as err:
        return render_template(
            "index.html",
            gp_name="[error]",
            upcoming_date="[Predictions appear only after the qualifying session finishes and FastF1 publishes timing data]",
            error=str(err),
            run_time=f"{time.time()-t0:.3f}"
        )

# ─── MAIN ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=True)
