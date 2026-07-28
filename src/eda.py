"""
eda.py
------
Stage: Exploratory Data Analysis.
Generates and saves plots that describe the dataset's structure and
its relationship with the target variable (Price).
"""

import os
import matplotlib
matplotlib.use("Agg")  # headless rendering, safe for scripts
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from src.config import FIGURES_DIR
from src.utils import get_logger, ensure_dir

logger = get_logger(__name__)
sns.set_theme(style="whitegrid")


class ExploratoryDataAnalyzer:
    """Produces and saves standard EDA visualizations."""

    def __init__(self, df: pd.DataFrame, output_dir: str = FIGURES_DIR):
        self.df = df
        self.output_dir = output_dir
        ensure_dir(self.output_dir)

    def _save(self, fig, filename: str) -> None:
        path = os.path.join(self.output_dir, filename)
        fig.savefig(path, bbox_inches="tight", dpi=120)
        plt.close(fig)
        logger.info(f"Saved figure: {path}")

    # ---------------- Distribution plots ----------------

    def plot_price_distribution(self) -> None:
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.histplot(self.df["Price"], kde=True, ax=ax, color="steelblue")
        ax.set_title("Distribution of Flight Prices")
        ax.set_xlabel("Price")
        self._save(fig, "price_distribution.png")

    # ---------------- Bar charts ----------------

    def plot_flight_count_by_airline(self) -> None:
        """Bar chart: how many flights each airline has in the dataset."""
        fig, ax = plt.subplots(figsize=(10, 6))
        counts = self.df["Airline"].value_counts()
        sns.barplot(x=counts.index, y=counts.values, hue=counts.index,
                    ax=ax, palette="Blues_d", legend=False)
        ax.set_title("Number of Flights per Airline")
        ax.set_ylabel("Number of Flights")
        ax.tick_params(axis="x", rotation=75)
        self._save(fig, "flight_count_by_airline.png")

    def plot_average_price_by_airline(self) -> None:
        """Bar chart: average price per airline."""
        fig, ax = plt.subplots(figsize=(10, 6))
        avg_price = self.df.groupby("Airline")["Price"].mean().sort_values(ascending=False)
        sns.barplot(x=avg_price.index, y=avg_price.values, hue=avg_price.index,
                    ax=ax, palette="Oranges_d", legend=False)
        ax.set_title("Average Price by Airline")
        ax.set_ylabel("Average Price")
        ax.tick_params(axis="x", rotation=75)
        self._save(fig, "average_price_by_airline.png")

    def plot_average_price_by_stops(self) -> None:
        """Bar chart: average price per number of stops."""
        fig, ax = plt.subplots(figsize=(7, 5))
        avg_price = self.df.groupby("Total_Stops")["Price"].mean().sort_index()
        sns.barplot(x=avg_price.index.astype(str), y=avg_price.values, hue=avg_price.index.astype(str),
                    ax=ax, palette="Greens_d", legend=False)
        ax.set_title("Average Price by Number of Stops")
        ax.set_xlabel("Total Stops")
        ax.set_ylabel("Average Price")
        self._save(fig, "average_price_by_stops.png")

    def plot_flight_count_by_source(self) -> None:
        """Bar chart: number of flights departing from each source city."""
        fig, ax = plt.subplots(figsize=(8, 5))
        counts = self.df["Source"].value_counts()
        sns.barplot(x=counts.index, y=counts.values, hue=counts.index,
                    ax=ax, palette="Purples_d", legend=False)
        ax.set_title("Number of Flights by Source City")
        ax.set_ylabel("Number of Flights")
        self._save(fig, "flight_count_by_source.png")

    # ---------------- Box plots ----------------

    def plot_price_by_airline_boxplot(self) -> None:
        fig, ax = plt.subplots(figsize=(10, 6))
        order = self.df.groupby("Airline")["Price"].median().sort_values(ascending=False).index
        sns.boxplot(data=self.df, x="Airline", y="Price", order=order, ax=ax)
        ax.set_title("Price Distribution by Airline")
        ax.tick_params(axis="x", rotation=75)
        self._save(fig, "price_by_airline_boxplot.png")

    def plot_price_by_route_boxplot(self) -> None:
        fig, ax = plt.subplots(figsize=(10, 6))
        order = self.df.groupby("Route")["Price"].median().sort_values(ascending=False).index
        sns.boxplot(data=self.df, x="Route", y="Price", order=order, ax=ax)
        ax.set_title("Price Distribution by Route")
        ax.tick_params(axis="x", rotation=75)
        self._save(fig, "price_by_route_boxplot.png")

    # ---------------- Relationship plots ----------------

    def plot_price_vs_duration(self) -> None:
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.scatterplot(data=self.df, x="Duration_Minutes", y="Price", hue="Total_Stops",
                         alpha=0.6, ax=ax, palette="viridis")
        ax.set_title("Price vs. Flight Duration")
        self._save(fig, "price_vs_duration.png")

    def plot_correlation_heatmap(self) -> None:
        numeric_df = self.df.select_dtypes(include="number")
        fig, ax = plt.subplots(figsize=(9, 7))
        sns.heatmap(numeric_df.corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
        ax.set_title("Correlation Heatmap (Numeric Features)")
        self._save(fig, "correlation_heatmap.png")

    # ---------------- Run everything ----------------

    def run_all(self) -> None:
        logger.info("----- EXPLORATORY DATA ANALYSIS -----")
        self.plot_price_distribution()
        self.plot_flight_count_by_airline()
        self.plot_average_price_by_airline()
        self.plot_average_price_by_stops()
        self.plot_flight_count_by_source()
        self.plot_price_by_airline_boxplot()
        self.plot_price_by_route_boxplot()
        self.plot_price_vs_duration()
        self.plot_correlation_heatmap()