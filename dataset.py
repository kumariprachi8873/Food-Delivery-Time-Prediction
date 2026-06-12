
# =====================================
# dataset.py
# =====================================

import pandas as pd
import numpy as np

from sklearn.preprocessing import (
    StandardScaler,
    LabelEncoder
)

from sklearn.model_selection import (
    train_test_split
)

from math import radians, cos, sin, asin, sqrt


class Dataset:

    # =====================================
    # INITIALIZE
    # =====================================

    def __init__(self, file_path):

        self.file_path = file_path

        self.data = None

        self.scaler = StandardScaler()

        self.label_encoders = {}

    # =====================================
    # LOAD DATA
    # =====================================

    def load_data(self):

        self.data = pd.read_csv(
            self.file_path
        )

        print("\nDataset Loaded Successfully")

        print(
            "Dataset Shape:",
            self.data.shape
        )

        print(
            "\nColumns:\n",
            self.data.columns
        )

    # =====================================
    # HANDLE MISSING VALUES
    # =====================================

    def handle_missing_values(self):

        num_cols = self.data.select_dtypes(
            include=['int64', 'float64']
        ).columns

        for col in num_cols:

            self.data[col] = (

                self.data[col]
                .fillna(
                    self.data[col].median()
                )
            )

        cat_cols = self.data.select_dtypes(
            include=['object']
        ).columns

        for col in cat_cols:

            self.data[col] = (

                self.data[col]
                .fillna(
                    self.data[col].mode()[0]
                )
            )

        print("\nMissing Values Handled")

    # =====================================
    # REMOVE DUPLICATES
    # =====================================

    def remove_duplicates(self):

        before = self.data.shape[0]

        self.data.drop_duplicates(
            inplace=True
        )

        after = self.data.shape[0]

        print(
            f"\nRemoved {before - after} Duplicate Rows"
        )

    # =====================================
    # HANDLE OUTLIERS
    # =====================================

    def handle_outliers(self):

        num_cols = self.data.select_dtypes(
            include=['int64', 'float64']
        ).columns

        for col in num_cols:

            Q1 = self.data[col].quantile(0.25)

            Q3 = self.data[col].quantile(0.75)

            IQR = Q3 - Q1

            lower = Q1 - (1.5 * IQR)

            upper = Q3 + (1.5 * IQR)

            self.data[col] = np.clip(
                self.data[col],
                lower,
                upper
            )

        print("\nOutliers Handled")

    # =====================================
    # HAVERSINE DISTANCE
    # =====================================

    def haversine_distance(

        self,
        lat1,
        lon1,
        lat2,
        lon2

    ):

        lon1, lat1, lon2, lat2 = map(

            radians,

            [lon1, lat1, lon2, lat2]
        )

        dlon = lon2 - lon1

        dlat = lat2 - lat1

        a = (

            sin(dlat / 2) ** 2

            + cos(lat1)
            * cos(lat2)
            * sin(dlon / 2) ** 2
        )

        c = 2 * asin(sqrt(a))

        r = 6371

        return c * r

    # =====================================
    # FEATURE ENGINEERING
    # =====================================

    def feature_engineering(self):

        # =====================================
        # DISTANCE FEATURE
        # =====================================

        required_columns = [

            'Restaurant_Latitude',
            'Restaurant_Longitude',
            'Customer_Latitude',
            'Customer_Longitude'
        ]

        if all(
            col in self.data.columns
            for col in required_columns
        ):

            self.data['Distance_km'] = (

                self.data.apply(

                    lambda row:

                    self.haversine_distance(

                        row['Restaurant_Latitude'],
                        row['Restaurant_Longitude'],
                        row['Customer_Latitude'],
                        row['Customer_Longitude']

                    ),

                    axis=1
                )
            )

            print(
                "\nDistance Feature Created"
            )

        # =====================================
        # TIME-BASED FEATURES
        # =====================================

        if 'Order_Time' in self.data.columns:

            try:

                self.data['Order_Hour'] = (

                    pd.to_datetime(
                        self.data['Order_Time']
                    ).dt.hour
                )

                # Rush Hour Feature

                self.data['Rush_Hour'] = (

                    self.data['Order_Hour']
                    .apply(

                        lambda x:

                        1 if (
                            7 <= x <= 10
                            or 17 <= x <= 21
                        )

                        else 0
                    )
                )

                # Time of Day

                def get_time_of_day(hour):

                    if 5 <= hour < 12:
                        return 'Morning'

                    elif 12 <= hour < 17:
                        return 'Afternoon'

                    elif 17 <= hour < 22:
                        return 'Evening'

                    else:
                        return 'Night'

                self.data['Time_of_Day'] = (

                    self.data['Order_Hour']
                    .apply(get_time_of_day)
                )

                print(
                    "\nTime-Based Features Created"
                )

            except:

                print(
                    "\nOrder_Time Format Issue"
                )

        # =====================================
        # WEATHER FEATURES
        # =====================================

        # Temperature Feature

        if 'Temperature' in self.data.columns:

            self.data['Temp_Category'] = (

                self.data['Temperature']
                .apply(

                    lambda x:

                    'Cold' if x < 15

                    else 'Moderate' if x < 30

                    else 'Hot'
                )
            )

            print(
                "\nTemperature Features Created"
            )

        # Humidity Feature

        if 'Humidity' in self.data.columns:

            self.data['Humidity_Level'] = (

                self.data['Humidity']
                .apply(

                    lambda x:

                    'Low' if x < 40

                    else 'Medium' if x < 70

                    else 'High'
                )
            )

            print(
                "\nHumidity Features Created"
            )

        # Weather Severity

        weather_cols = [

            'Weather_Conditions',
            'Temperature',
            'Humidity'
        ]

        if all(
            col in self.data.columns
            for col in weather_cols
        ):

            def weather_severity(row):

                score = 0

                if str(
                    row['Weather_Conditions']
                ).lower() in [

                    'rainy',
                    'stormy',
                    'snowy'
                ]:

                    score += 2

                if row['Temperature'] > 35:

                    score += 1

                if row['Humidity'] > 80:

                    score += 1

                return score

            self.data['Weather_Severity'] = (

                self.data.apply(
                    weather_severity,
                    axis=1
                )
            )

            print(
                "\nWeather Severity Feature Created"
            )

        print(
            "\nFeature Engineering Completed"
        )

    # =====================================
    # CREATE TARGET VARIABLE
    # =====================================

    def create_target_variable(self):

        possible_columns = [

            'Delivery_Time',
            'Delivery_Time_min',
            'Time_taken(min)',
            'Time_Taken',
            'DeliveryDuration'
        ]

        target_col = None

        for col in possible_columns:

            if col in self.data.columns:

                target_col = col

                break

        if target_col is None:

            print(
                "\nAvailable Columns:\n",
                self.data.columns
            )

            raise ValueError(
                "\nNo Delivery Time Column Found"
            )

        threshold = (

            self.data[target_col]
            .median()
        )

        # Fast = 0
        # Delayed = 1

        self.data['Delivery_Status'] = (

            self.data[target_col]
            .apply(

                lambda x:

                1 if x > threshold

                else 0
            )
        )

        print(
            f"\nTarget Variable Created using '{target_col}'"
        )

    # =====================================
    # ENCODE CATEGORICAL FEATURES
    # =====================================

    def encode_features(self):

        categorical_cols = self.data.select_dtypes(
            include=['object']
        ).columns

        for col in categorical_cols:

            encoder = LabelEncoder()

            self.data[col] = (

                encoder.fit_transform(
                    self.data[col].astype(str)
                )
            )

            self.label_encoders[col] = encoder

        print(
            "\nCategorical Features Encoded"
        )

    # =====================================
    # SCALE FEATURES
    # =====================================

    def scale_features(self):

        scale_cols = self.data.select_dtypes(
            include=['int64', 'float64']
        ).columns

        scale_cols = [

            col for col in scale_cols

            if col != 'Delivery_Status'
        ]

        self.data[scale_cols] = (

            self.scaler.fit_transform(
                self.data[scale_cols]
            )
        )

        print(
            "\nFeature Scaling Completed"
        )

    # =====================================
    # SPLIT DATA
    # =====================================

    def split_data(self):

        X = self.data.drop(
            columns=['Delivery_Status']
        )

        y = self.data['Delivery_Status']

        X_train, X_test, y_train, y_test = (

            train_test_split(

                X,
                y,

                test_size=0.2,

                random_state=42,

                stratify=y
            )
        )

        return (
            X_train,
            X_test,
            y_train,
            y_test
        )

    # =====================================
    # COMPLETE PREPROCESSING PIPELINE
    # =====================================

    def preprocess(self):

        self.load_data()

        self.handle_missing_values()

        self.remove_duplicates()

        self.handle_outliers()

        self.feature_engineering()

        self.create_target_variable()

        self.encode_features()

        self.scale_features()

        return self.split_data()

