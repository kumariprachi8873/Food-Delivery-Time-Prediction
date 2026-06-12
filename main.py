
# =====================================
# main.py
# FOOD DELIVERY TIME PREDICTION
# CNN MODEL
# =====================================

from dataset import Dataset

from model import CNNModel


def main():

    print(
        "\n====================================="
    )

    print(
        " FOOD DELIVERY CNN PREDICTION SYSTEM "
    )

    print(
        "=====================================\n"
    )

    # =====================================
    # DATASET LOADING + PREPROCESSING
    # =====================================

    dataset = Dataset(

        "Food_Delivery_Time_Prediction.csv"
    )

    (
        X_train,
        X_test,
        y_train,
        y_test

    ) = dataset.preprocess()

    print(
        "\nDataset Preprocessing Completed"
    )

    print(
        "\nTraining Shape:",
        X_train.shape
    )

    print(
        "Testing Shape:",
        X_test.shape
    )

    # =====================================
    # CNN MODEL
    # =====================================

    cnn_model = CNNModel()

    # =====================================
    # RUN COMPLETE PIPELINE
    # =====================================

    cnn_model.run_pipeline(

        X_train,
        X_test,
        y_train,
        y_test
    )

    # =====================================
    # CROSS VALIDATION
    # =====================================

    print(
        "\n====================================="
    )

    print(
        " CROSS VALIDATION "
    )

    print(
        "=====================================\n"
    )

    # Full dataset for K-Fold

    X = dataset.data.drop(
        columns=['Delivery_Status']
    )

    y = dataset.data['Delivery_Status']

    cnn_model.cross_validation(
        X,
        y,
        folds=5
    )

    print(
        "\n====================================="
    )

    print(
        " PROJECT COMPLETED SUCCESSFULLY "
    )

    print(
        "=====================================\n"
    )


# =====================================
# DRIVER CODE
# =====================================

if __name__ == "__main__":

    main()

