
# main.py

from dataset import Dataset
from model import AprioriModel


def main():

    # ==========================================
    # DATASET PATH
    # ==========================================

    file_path = "Global_Pollution_Analysis.csv"

    # ==========================================
    # LOAD & PREPROCESS DATA
    # ==========================================

    dataset = Dataset(file_path)

    dataset.preprocess()

    # ==========================================
    # PREPARE TRANSACTIONS FOR APRIORI
    # ==========================================

    transactions_df = dataset.get_apriori_data()

    print("\n===================================")
    print("Sample Transaction Data")
    print("===================================")
    print(transactions_df.head())

    # ==========================================
    # INITIALIZE MODEL
    # ==========================================

    model = AprioriModel(

        min_support=0.20,

        min_confidence=0.60

    )

    # ==========================================
    # RUN APRIORI PIPELINE
    # ==========================================

    model.run(transactions_df)

    # ==========================================
    # DISPLAY FINAL RULES
    # ==========================================

    print("\n===================================")
    print("FINAL ASSOCIATION RULES")
    print("===================================")

    if model.rules is not None and not model.rules.empty:

        display_columns = [

            "antecedents",
            "consequents",
            "support",
            "confidence",
            "lift"

        ]

        print(model.rules[display_columns])

    else:

        print(
            "No association rules found.\n"
            "Try lowering min_support or "
            "min_confidence."
        )


# ==========================================
# DRIVER CODE
# ==========================================

if __name__ == "__main__":

    main()

