
# dataset.py

import pandas as pd
import numpy as np


class Dataset:

    def __init__(self, file_path):

        self.file_path = file_path
        self.data = None

    # ==========================================
    # LOAD DATA
    # ==========================================

    def load_data(self):

        self.data = pd.read_csv(self.file_path)

        print("Dataset Loaded Successfully")
        print(self.data.head())

    # ==========================================
    # HANDLE MISSING VALUES
    # ==========================================

    def handle_missing(self):

        numeric_cols = self.data.select_dtypes(
            include=["number"]
        ).columns

        self.data[numeric_cols] = self.data[
            numeric_cols
        ].fillna(
            self.data[numeric_cols].median()
        )

        categorical_cols = self.data.select_dtypes(
            include=["object", "string"]
        ).columns

        for col in categorical_cols:

            self.data[col] = self.data[col].fillna(
                self.data[col].mode()[0]
            )

        print("Missing Values Handled")

    # ==========================================
    # HANDLE OUTLIERS (IQR CLIPPING)
    # ==========================================

    def handle_outliers(self):

        numeric_cols = self.data.select_dtypes(
            include=["number"]
        ).columns

        for col in numeric_cols:

            q1 = self.data[col].quantile(0.25)
            q3 = self.data[col].quantile(0.75)

            iqr = q3 - q1

            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr

            self.data[col] = self.data[col].clip(
                lower=lower,
                upper=upper
            )

        print("Outliers Handled")

    # ==========================================
    # FEATURE ENGINEERING
    # ==========================================

    def feature_engineering(self):

        # Combined Pollution Score

        self.data["Pollution_Score"] = (

            self.data[
                [
                    "Air_Pollution_Index",
                    "Water_Pollution_Index",
                    "Soil_Pollution_Index"
                ]
            ].mean(axis=1)

        )

        # Energy Consumption Feature

        self.data[
            "Energy_Consumption_Per_Capita_Feature"
        ] = self.data[
            "Energy_Consumption_Per_Capita (in MWh)"
        ]

        # Pollution Trend

        self.data = self.data.sort_values(
            ["Country", "Year"]
        )

        self.data["Pollution_Trend"] = (

            self.data.groupby("Country")[
                "Pollution_Score"
            ].diff()

        )

        self.data["Pollution_Trend"] = (

            self.data["Pollution_Trend"].fillna(0)

        )

        # Pollution Categories
        # qcut avoids NaN caused by fixed bins

        self.data["Air_Level"] = pd.qcut(

            self.data["Air_Pollution_Index"],

            q=3,

            labels=[
                "Air_Low",
                "Air_Medium",
                "Air_High"
            ]

        )

        self.data["Water_Level"] = pd.qcut(

            self.data["Water_Pollution_Index"],

            q=3,

            labels=[
                "Water_Low",
                "Water_Medium",
                "Water_High"
            ]

        )

        self.data["Pollution_Level"] = pd.qcut(

            self.data["Pollution_Score"],

            q=3,

            labels=[
                "Pollution_Low",
                "Pollution_Medium",
                "Pollution_High"
            ]

        )

        print("Feature Engineering Completed")

    # ==========================================
    # APRIORI TRANSACTION DATA
    # ==========================================

    def get_apriori_data(self):

        df = self.data.copy()

        recovery_threshold = df[
            "Energy_Recovered (in GWh)"
        ].median()

        renewable_threshold = df[
            "Renewable_Energy (%)"
        ].median()

        consumption_threshold = df[
            "Energy_Consumption_Per_Capita_Feature"
        ].median()

        transactions = pd.DataFrame()

        transactions["Air"] = (
            df["Air_Level"].astype(str)
        )

        transactions["Water"] = (
            df["Water_Level"].astype(str)
        )

        transactions["Pollution"] = (
            df["Pollution_Level"].astype(str)
        )

        transactions["Recovery"] = np.where(

            df["Energy_Recovered (in GWh)"]
            >= recovery_threshold,

            "Recovery_High",

            "Recovery_Low"

        )

        transactions["Renewable"] = np.where(

            df["Renewable_Energy (%)"]
            >= renewable_threshold,

            "Renewable_High",

            "Renewable_Low"

        )

        transactions["Consumption"] = np.where(

            df[
                "Energy_Consumption_Per_Capita_Feature"
            ] >= consumption_threshold,

            "Consumption_High",

            "Consumption_Low"

        )

        # Remove any missing values

        transactions = transactions.fillna(
            "Unknown"
        )

        return transactions

    # ==========================================
    # COMPLETE PIPELINE
    # ==========================================

    def preprocess(self):

        self.load_data()

        self.handle_missing()

        self.handle_outliers()

        self.feature_engineering()

        print("Preprocessing Completed Successfully")

