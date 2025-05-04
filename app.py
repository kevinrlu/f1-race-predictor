import logging
import time
from datetime import datetime, date
from functools import lru_cache
from pathlib import Path
import fastf1
import pandas as pd
from flask import Flask, render_template
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GroupShuffleSplit

logging.getLogger("fastf1").setLevel(logging.WARNING)
CACHE_DIR = Path(__file__).parent / "f1_cache"
fastf1.Cache.enable_cache(str(CACHE_DIR))
YEAR = datetime.now().year
LAP_COUNTS: dict[str, int] = {
    "Australian Grand Prix": 57,
    "Chinese Grand Prix": 56,
    "Japanese Grand Prix": 53,
    "Bahrain Grand Prix": 57,
    "Saudi Arabian Grand Prix": 50,
    "Miami Grand Prix": 57,
    "Emilia Romagna Grand Prix": 63,
    "Monaco Grand Prix": 78,
    "Spanish Grand Prix": 66,
    "Canadian Grand Prix": 70,
    "Austrian Grand Prix": 71,
    "British Grand Prix": 52,
    "Belgian Grand Prix": 44,
    "Hungarian Grand Prix": 70,
    "Dutch Grand Prix": 72,
    "Italian Grand Prix": 53,
    "Azerbaijan Grand Prix": 51,
    "Singapore Grand Prix": 61,
    "United States Grand Prix": 56,
    "Mexican Grand Prix": 71,
    "Brazilian Grand Prix": 71,
    "Las Vegas Grand Prix": 50,
    "Qatar Grand Prix": 57,
    "Abu Dhabi Grand Prix": 55,
}

def create_app() -> Flask:
    app = Flask(__name__)

    @app.route("/")
    def index():
        start_time = time.time()
        predictor = RacePredictor(YEAR, LAP_COUNTS)
        try:
            context = predictor.run()
        except Exception:
            context = {
                "error": (
                    "⚠️ Predictions will be available only after the qualifying "
                    "session ends AND FastF1 publishes the data ⚠️"
                )
            }
        context["run_time"] = f"{time.time() - start_time:.3f}"
        return render_template("index.html", **context)
    return app

class RacePredictor:
    def __init__(self, year: int, lap_counts: dict[str, int]) -> None:
        self.year = year
        self.lap_counts = lap_counts

    def run(self) -> dict:
        schedule = self._get_schedule()
        past, upcoming = self._split_schedule(schedule)

        if upcoming.empty:
            raise ValueError("No upcoming races found.")

        training_data = self._assemble_training_data(past)
        if training_data.empty:
            raise ValueError("No valid past training data available.")

        model, mae = self._train_model(training_data)
        return self._predict_next(upcoming.iloc[0], model, mae)

    def _get_schedule(self) -> pd.DataFrame:
        schedule = fastf1.get_event_schedule(self.year)
        schedule["Date"] = pd.to_datetime(schedule["EventDate"]).dt.date
        return schedule[schedule["RoundNumber"] > 0]

    def _split_schedule(self, schedule: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        today = date.today()
        return schedule[schedule["Date"] < today], schedule[schedule["Date"] >= today]

    def _assemble_training_data(self, past_races: pd.DataFrame) -> pd.DataFrame:
        frames = [self._race_frame(gp) for gp in past_races["EventName"]]
        combined = pd.concat(frames, ignore_index=True)
        return combined.dropna(subset=["Quali_s", "Total_s"])

    @lru_cache(maxsize=None)
    def _race_frame(self, gp: str) -> pd.DataFrame:
        quali = _fetch_times(self.year, gp, "Q").rename("Quali_s")
        race = _fetch_times(self.year, gp, "R").rename("Total_s")
        return pd.concat([quali, race], axis=1).assign(
            Laps=self.lap_counts.get(gp, 0),
            Event=gp
        )

    def _train_model(self, df: pd.DataFrame) -> tuple[LinearRegression, float]:
        X = df[["Quali_s", "Laps"]]
        y = df["Total_s"]
        groups = df["Event"]

        splitter = GroupShuffleSplit(test_size=0.2, random_state=42)
        train_idx, test_idx = next(splitter.split(X, groups=groups))

        model = LinearRegression().fit(X.iloc[train_idx], y.iloc[train_idx])
        preds = model.predict(X.iloc[test_idx])
        mae = mean_absolute_error(y.iloc[test_idx], preds)
        return model, mae

    def _predict_next(self, race: pd.Series, model: LinearRegression, mae: float) -> dict:
        gp = race["EventName"]
        laps = self.lap_counts.get(gp)
        if not laps:
            raise ValueError(f"Lap count not found for {gp}")

        q_times = _fetch_times(self.year, gp, "Q")
        if q_times.empty:
            raise ValueError("No qualifying data available.")

        X_next = pd.DataFrame({
            "Quali_s": q_times,
            "Laps": laps
        })

        pred_secs = model.predict(X_next)

        session = fastf1.get_session(self.year, gp, "Q")
        session.load(laps=True, telemetry=False, weather=False, messages=False)

        name_map = _map_abbr_to_fullname(session)
        team_map = _map_driver_to_team(session)

        preds_df = pd.DataFrame({
            "DriverCode": X_next.index,
            "Pred_s": pred_secs
        }).assign(
            PredictedTime=lambda df: df["Pred_s"].apply(_format_hms),
            Driver=lambda df: df["DriverCode"].map(name_map),
            Team=lambda df: df["DriverCode"].map(team_map),
            Pos=lambda df: df["Pred_s"].rank(method="first").astype(int)
        ).sort_values("Pred_s").reset_index(drop=True)

        return {
            "gp_name": gp,
            "upcoming_date": race["Date"].isoformat(),
            "mae": f"{mae:.3f}",
            "preds": preds_df[["Pos", "Driver", "Team", "PredictedTime"]].to_dict("records"),
            "error": None,
        }

@lru_cache(maxsize=64)
def _fetch_times(year: int, gp: str, session_type: str) -> pd.Series:
    session = fastf1.get_session(year, gp, session_type)
    session.load(laps=True, telemetry=False, weather=False, messages=False)
    laps = session.laps.dropna(subset=["Driver", "LapTime"])
    grouped = laps.groupby("Driver")["LapTime"]
    aggregated = grouped.agg("min" if session_type == "Q" else "sum")
    return aggregated.dt.total_seconds()

def _map_abbr_to_fullname(session) -> dict[str, str]:
    return {
        info["Abbreviation"]: info["FullName"]
        for drv in session.drivers
        if (info := session.get_driver(drv)) is not None
    }

def _map_driver_to_team(session) -> dict[str, str]:
    return (
        session.laps.dropna(subset=["Driver", "Team"])
        .drop_duplicates("Driver")
        .set_index("Driver")["Team"]
        .to_dict()
    )

def _format_hms(seconds: float) -> str:
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{int(h)}:{int(m):02d}:{s:06.3f}"

if __name__ == "__main__":
    app = create_app()
    app.run(debug=True)