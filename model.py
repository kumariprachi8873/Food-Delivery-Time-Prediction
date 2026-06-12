
# model.py

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules


class AprioriModel:

    def __init__(
        self,
        min_support=0.20,
        min_confidence=0.60
    ):

        self.min_support = min_support
        self.min_confidence = min_confidence

        self.frequent_itemsets = None
        self.rules = None

    # ==========================================
    # CONVERT DATAFRAME TO TRANSACTIONS
    # ==========================================

    def prepare_transactions(self, transactions_df):

        transactions = []

        for _, row in transactions_df.iterrows():

            transaction = []

            for item in row:

                if pd.notna(item):

                    transaction.append(str(item))

            transactions.append(transaction)

        return transactions

    # ==========================================
    # TRAIN / TEST SPLIT
    # ==========================================

    def split_data(self, transactions):

        train, test = train_test_split(

            transactions,

            test_size=0.20,

            random_state=42,

            shuffle=True

        )

        return train, test

    # ==========================================
    # ONE HOT ENCODING
    # ==========================================

    def encode_transactions(self, transactions):

        te = TransactionEncoder()

        encoded = te.fit(transactions).transform(
            transactions
        )

        encoded_df = pd.DataFrame(

            encoded,

            columns=te.columns_

        )

        return encoded_df

    # ==========================================
    # APRIORI TRAINING
    # ==========================================

    def fit(self, train_transactions):

        encoded_df = self.encode_transactions(
            train_transactions
        )

        self.frequent_itemsets = apriori(

            encoded_df,

            min_support=self.min_support,

            use_colnames=True

        )

        self.rules = association_rules(

            self.frequent_itemsets,

            metric="confidence",

            min_threshold=self.min_confidence

        )

        self.rules = self.rules.sort_values(

            by="lift",

            ascending=False

        )

        print("\n========== Frequent Itemsets ==========")

        print(self.frequent_itemsets)

        print("\n========== Association Rules ==========")

        print(

            self.rules[

                [

                    "antecedents",

                    "consequents",

                    "support",

                    "confidence",

                    "lift"

                ]

            ]

        )

    # ==========================================
    # EVALUATION
    # ==========================================

    def evaluate(self):

        if self.rules.empty:

            print("No association rules found.")

            return

        print("\n========== Evaluation ==========")

        print(

            "Average Support :",

            round(

                self.rules["support"].mean(),

                4

            )

        )

        print(

            "Average Confidence :",

            round(

                self.rules["confidence"].mean(),

                4

            )

        )

        print(

            "Average Lift :",

            round(

                self.rules["lift"].mean(),

                4

            )

        )

    # ==========================================
    # SIMPLE VALIDATION
    # ==========================================

    def validate(self, test_transactions):

        if self.rules.empty:

            return

        matches = 0

        for transaction in test_transactions:

            t = set(transaction)

            for _, rule in self.rules.iterrows():

                antecedent = set(
                    rule["antecedents"]
                )

                consequent = set(
                    rule["consequents"]
                )

                if antecedent.issubset(t):

                    if consequent.issubset(t):

                        matches += 1

                    break

        accuracy = matches / len(test_transactions)

        print(

            "\nRule Match Accuracy :",

            round(accuracy, 4)

        )

    # ==========================================
    # BAR GRAPH OF SUPPORT
    # ==========================================

    def plot_support(self):

        if self.frequent_itemsets.empty:

            return

        plt.figure(figsize=(10, 5))

        plt.bar(

            range(

                len(self.frequent_itemsets)

            ),

            self.frequent_itemsets["support"]

        )

        plt.xlabel("Frequent Itemsets")

        plt.ylabel("Support")

        plt.title(

            "Frequent Itemsets Support"

        )

        plt.show()

    # ==========================================
    # SCATTER PLOT
    # ==========================================

    def plot_rules(self):

        if self.rules.empty:

            return

        plt.figure(figsize=(8, 6))

        plt.scatter(

            self.rules["support"],

            self.rules["confidence"]

        )

        plt.xlabel("Support")

        plt.ylabel("Confidence")

        plt.title(

            "Association Rules"

        )

        plt.show()

    # ==========================================
    # COMPLETE PIPELINE
    # ==========================================

    def run(self, transactions_df):

        transactions = self.prepare_transactions(
            transactions_df
        )

        train, test = self.split_data(
            transactions
        )

        self.fit(train)

        self.evaluate()

        self.validate(test)

        self.plot_support()

        self.plot_rules()

