#!/usr/bin/env python3
"""Refactored F1 race‑time predictor (2025) following clean code principles."""
import logging
import time
from datetime import date
from functools import lru_cache
from pathlib import Path

import fastf1
import pandas as pd
from flask import Flask, render_template
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GroupShuffleSplit

# ─── CONFIG ────────────────────────────────────────────────────────────────────
logging.getLogger("fastf1").setLevel(logging.ERROR)
CACHE_DIR = Path(__file__).parent / "f1_cache"
fastf1.Cache.enable_cache(str(CACHE_DIR))  # disk cache
YEAR = 2025
LAP_COUNTS: dict[str, int] = {
    "Australian Grand Prix": 57,  "Chinese Grand Prix": 56,   "Japanese Grand Prix": 53,
    "Bahrain Grand Prix": 57,     "Saudi Arabian Grand Prix": 50,  "Miami Grand Prix": 57,
    "Emilia Romagna Grand Prix": 63, "Monaco Grand Prix": 78, "Spanish Grand Prix": 66,
    "Canadian Grand Prix": 70,    "Austrian Grand Prix": 71,  "British Grand Prix": 52,
    "Belgian Grand Prix": 44,     "Hungarian Grand Prix": 70, "Dutch Grand Prix": 72,
    "Italian Grand Prix": 53,     "Azerbaijan Grand Prix": 51, "Singapore Grand Prix": 61,
    "United States Grand Prix": 56, "Mexican Grand Prix": 71,  "Brazilian Grand Prix": 71,
    "Las Vegas Grand Prix": 50,   "Qatar Grand Prix": 57,     "Abu Dhabi Grand Prix": 55,
}

def create_app() -> Flask:
    """Flask application factory."""
    app = Flask(__name__)

    @app.route("/")
    def index():
        start_time = time.time()
        predictor = RacePredictor(year=YEAR, lap_counts=LAP_COUNTS)
        try:
            context = predictor.run()
            context["run_time"] = f"{time.time() - start_time:.3f}"
            return render_template("index.html", **context)
        except Exception as err:
            # Centralized error handling
            return render_template(
                "index.html",
                gp_name="[Predictions appear only after the qualifying session finishes and FastF1 publishes timing data]",
                upcoming_date="[Predictions appear only after the qualifying session finishes and FastF1 publishes timing data]",
                error=str(err),
                run_time=f"{time.time() - start_time:.3f}"
            )

    return app

class RacePredictor:
    """Encapsulates data retrieval, model training, and prediction."""

    def __init__(self, year: int, lap_counts: dict[str, int]) -> None:
        self.year = year
        self.lap_counts = lap_counts

    def run(self) -> dict:
        schedule = self._get_schedule()
        past, upcoming = self._split_races(schedule)

        if past.empty:
            raise ValueError("No past races to train on.")

        training_df = self._build_training_data(past)
        model, mae = self._train_model(training_df)

        if upcoming.empty:
            return {
                "gp_name": "[Predictions appear only after the qualifying session finishes and FastF1 publishes timing data]",
                "upcoming_date": "[Predictions appear only after the qualifying session finishes and FastF1 publishes timing data]",
                "mae": f"{mae:.3f}",
                "error": None,
                "preds": []
            }

        return self._predict_next(upcoming.iloc[0], model, mae)

    def _get_schedule(self) -> pd.DataFrame:
        sched = fastf1.get_event_schedule(self.year)
        sched["Date"] = pd.to_datetime(sched["EventDate"]).dt.date
        return sched[sched["RoundNumber"] > 0]

    def _split_races(self, schedule: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        today = date.today()
        past = schedule[schedule["Date"] < today]
        upcoming = schedule[schedule["Date"] >= today]
        return past, upcoming

    def _build_training_data(self, past: pd.DataFrame) -> pd.DataFrame:
        frames = [self._race_frame(gp) for gp in past["EventName"]]
        df = pd.concat(frames, ignore_index=True)
        return df.dropna(subset=["Quali_s", "Total_s"])

    @lru_cache(maxsize=None)
    def _race_frame(self, gp: str) -> pd.DataFrame:
        # Cache individual race data to avoid redundant FastF1 calls
        q = _fetch_times(self.year, gp, "Q").rename("Quali_s")
        r = _fetch_times(self.year, gp, "R").rename("Total_s")
        df = pd.concat([q, r], axis=1)
        df["Laps"] = self.lap_counts.get(gp)
        df["Event"] = gp
        return df

    def _train_model(self, df: pd.DataFrame) -> tuple[LinearRegression, float]:
        X = df[["Quali_s", "Laps"]]
        y = df["Total_s"]
        splitter = GroupShuffleSplit(test_size=0.2, random_state=42)
        train_idx, test_idx = next(splitter.split(X, groups=df["Event"]))
        model = LinearRegression().fit(X.iloc[train_idx], y.iloc[train_idx])
        mae = mean_absolute_error(y.iloc[test_idx], model.predict(X.iloc[test_idx]))
        return model, mae

    def _predict_next(self, race_info: pd.Series, model: LinearRegression, mae: float) -> dict:
        gp = race_info["EventName"]
        date_iso = race_info["Date"].isoformat()
        laps = self.lap_counts.get(gp)

        # Prepare qualifying data
        q_series = _fetch_times(self.year, gp, "Q")
        X_next = pd.DataFrame({"Quali_s": q_series, "Laps": laps})
        preds_seconds = model.predict(X_next)

        # Map driver info
        session = fastf1.get_session(self.year, gp, "Q")
        session.load(laps=True, telemetry=False, weather=False, messages=False)
        name_map = _map_abbr_to_fullname(session)
        team_map = _map_driver_to_team(session)

        preds_df = (
            pd.DataFrame({"DriverCode": X_next.index, "Pred_s": preds_seconds})
            .assign(
                PredictedTime=lambda df: df["Pred_s"].apply(_format_hms),
                Driver=lambda df: df["DriverCode"].map(name_map),
                Team=lambda df: df["DriverCode"].map(team_map)
            )
            .sort_values("Pred_s")
            .reset_index(drop=True)
        )
        preds_df["Pos"] = preds_df.index + 1

        return {
            "gp_name": gp,
            "upcoming_date": date_iso,
            "mae": f"{mae:.3f}",
            "preds": preds_df[["Pos", "Driver", "Team", "PredictedTime"]].to_dict("records"),
            "error": None
        }


def _fetch_times(year: int, gp: str, session_type: str) -> pd.Series:
    """Fetch fastest lap (Q) or total race time (R) per driver."""
    sess = fastf1.get_session(year, gp, session_type)
    sess.load(laps=True, telemetry=False, weather=False, messages=False)
    laps = sess.laps.dropna(subset=["Driver", "LapTime"])
    grouped = laps.groupby("Driver")["LapTime"]
    times = grouped.min() if session_type == "Q" else grouped.sum()
    return times.dt.total_seconds()


def _map_abbr_to_fullname(session) -> dict[str, str]:
    return {
        info["Abbreviation"]: info["FullName"]
        for drv in session.drivers
        if (info := session.get_driver(drv)) is not None
    }


def _map_driver_to_team(session) -> dict[str, str]:
    df = (
        session.laps
        .dropna(subset=["Driver", "Team"])
        [["Driver", "Team"]]
        .drop_duplicates()
        .set_index("Driver")["Team"]
    )
    return df.to_dict()


def _format_hms(seconds: float) -> str:
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{int(h)}:{int(m):02d}:{s:06.3f}"


if __name__ == "__main__":
    app = create_app()
    app.run(debug=True)  # Consider disabling debug in production
