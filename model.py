
# =====================================
# model.py
# CNN MODEL FOR FOOD DELIVERY PREDICTION
# =====================================

import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (

    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc

)

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import (

    KFold
)

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import (

    Conv1D,
    MaxPooling1D,
    Flatten,
    Dense,
    Dropout,
    BatchNormalization

)

from tensorflow.keras.optimizers import Adam

from tensorflow.keras.callbacks import EarlyStopping


class CNNModel:

    # =====================================
    # INITIALIZE
    # =====================================

    def __init__(self):

        self.model = None

    # =====================================
    # RESHAPE DATA FOR CNN
    # =====================================

    def reshape_data(

        self,
        X_train,
        X_test

    ):

        X_train = np.array(X_train)

        X_test = np.array(X_test)

        X_train = X_train.reshape(

            X_train.shape[0],
            X_train.shape[1],
            1
        )

        X_test = X_test.reshape(

            X_test.shape[0],
            X_test.shape[1],
            1
        )

        return X_train, X_test

    # =====================================
    # BUILD CNN MODEL
    # =====================================

    def build_model(

        self,
        input_shape,
        filters_1=64,
        filters_2=128,
        kernel_size=3,
        learning_rate=0.001

    ):

        self.model = Sequential()

        # =====================================
        # FIRST CNN LAYER
        # =====================================

        self.model.add(

            Conv1D(

                filters=filters_1,

                kernel_size=kernel_size,

                activation='relu',

                input_shape=input_shape
            )
        )

        self.model.add(
            BatchNormalization()
        )

        self.model.add(
            MaxPooling1D(pool_size=2)
        )

        self.model.add(
            Dropout(0.3)
        )

        # =====================================
        # SECOND CNN LAYER
        # =====================================

        self.model.add(

            Conv1D(

                filters=filters_2,

                kernel_size=kernel_size,

                activation='relu'
            )
        )

        self.model.add(
            BatchNormalization()
        )

        self.model.add(
            MaxPooling1D(pool_size=2)
        )

        self.model.add(
            Dropout(0.3)
        )

        # =====================================
        # FLATTEN
        # =====================================

        self.model.add(
            Flatten()
        )

        # =====================================
        # DENSE LAYERS
        # =====================================

        self.model.add(
            Dense(128, activation='relu')
        )

        self.model.add(
            Dropout(0.4)
        )

        self.model.add(
            Dense(64, activation='relu')
        )

        # =====================================
        # OUTPUT LAYER
        # =====================================

        self.model.add(
            Dense(1, activation='sigmoid')
        )

        # =====================================
        # COMPILE MODEL
        # =====================================

        optimizer = Adam(
            learning_rate=learning_rate
        )

        self.model.compile(

            optimizer=optimizer,

            loss='binary_crossentropy',

            metrics=['accuracy']
        )

        print("\nCNN Model Built Successfully")

    # =====================================
    # TRAIN MODEL
    # =====================================

    def train_model(

        self,
        X_train,
        y_train,
        epochs=30,
        batch_size=16

    ):

        early_stop = EarlyStopping(

            monitor='val_loss',

            patience=5,

            restore_best_weights=True
        )

        history = self.model.fit(

            X_train,
            y_train,

            validation_split=0.2,

            epochs=epochs,

            batch_size=batch_size,

            callbacks=[early_stop],

            verbose=1
        )

        return history

    # =====================================
    # EVALUATE MODEL
    # =====================================

    def evaluate_model(

        self,
        X_test,
        y_test

    ):

        predictions_prob = self.model.predict(
            X_test
        )

        predictions = (

            predictions_prob > 0.5
        ).astype(int)

        # =====================================
        # METRICS
        # =====================================

        accuracy = accuracy_score(
            y_test,
            predictions
        )

        precision = precision_score(
            y_test,
            predictions
        )

        recall = recall_score(
            y_test,
            predictions
        )

        f1 = f1_score(
            y_test,
            predictions
        )

        print("\n===== CNN RESULTS =====")

        print(f"Accuracy : {accuracy}")

        print(f"Precision : {precision}")

        print(f"Recall : {recall}")

        print(f"F1 Score : {f1}")

        print(

            "\nClassification Report:\n",

            classification_report(
                y_test,
                predictions
            )
        )

        print(

            "\nConfusion Matrix:\n",

            confusion_matrix(
                y_test,
                predictions
            )
        )

        # =====================================
        # ROC CURVE
        # =====================================

        fpr, tpr, thresholds = roc_curve(

            y_test,
            predictions_prob
        )

        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=(7, 5))

        plt.plot(
            fpr,
            tpr,
            label=f'AUC = {roc_auc:.2f}'
        )

        plt.plot(
            [0, 1],
            [0, 1],
            linestyle='--'
        )

        plt.xlabel('False Positive Rate')

        plt.ylabel('True Positive Rate')

        plt.title('ROC Curve')

        plt.legend()

        plt.show()

    # =====================================
    # LOGISTIC REGRESSION COMPARISON
    # =====================================

    def logistic_regression_baseline(

        self,
        X_train,
        X_test,
        y_train,
        y_test

    ):

        # CNN to 2D reshape

        X_train_lr = X_train.reshape(

            X_train.shape[0],
            X_train.shape[1]
        )

        X_test_lr = X_test.reshape(

            X_test.shape[0],
            X_test.shape[1]
        )

        model = LogisticRegression()

        model.fit(
            X_train_lr,
            y_train
        )

        predictions = model.predict(
            X_test_lr
        )

        accuracy = accuracy_score(
            y_test,
            predictions
        )

        print(
            "\n===== LOGISTIC REGRESSION ====="
        )

        print(
            f"Accuracy : {accuracy}"
        )

        print(

            "\nClassification Report:\n",

            classification_report(
                y_test,
                predictions
            )
        )

    # =====================================
    # K-FOLD CROSS VALIDATION
    # =====================================

    def cross_validation(

        self,
        X,
        y,
        folds=5

    ):

        X = np.array(X)

        y = np.array(y)

        kf = KFold(

            n_splits=folds,

            shuffle=True,

            random_state=42
        )

        fold = 1

        accuracy_scores = []

        for train_index, test_index in kf.split(X):

            print(
                f"\n===== Fold {fold} ====="
            )

            X_train = X[train_index]

            X_test = X[test_index]

            y_train = y[train_index]

            y_test = y[test_index]

            # Reshape

            X_train = X_train.reshape(

                X_train.shape[0],
                X_train.shape[1],
                1
            )

            X_test = X_test.reshape(

                X_test.shape[0],
                X_test.shape[1],
                1
            )

            # Build Model

            self.build_model(

                (
                    X_train.shape[1],
                    1
                )
            )

            # Train

            self.train_model(

                X_train,
                y_train,
                epochs=15
            )

            # Predict

            predictions = (

                self.model.predict(
                    X_test
                ) > 0.5

            ).astype(int)

            accuracy = accuracy_score(

                y_test,
                predictions
            )

            accuracy_scores.append(
                accuracy
            )

            print(
                f"Fold Accuracy: {accuracy}"
            )

            fold += 1

        print(

            "\nAverage Cross Validation Accuracy:",

            np.mean(accuracy_scores)
        )

    # =====================================
    # HYPERPARAMETER TUNING
    # =====================================

    def hyperparameter_tuning(

        self,
        X_train,
        X_test,
        y_train,
        y_test

    ):

        filter_options = [32, 64]

        kernel_options = [2, 3]

        learning_rates = [

            0.001,
            0.0001
        ]

        best_accuracy = 0

        best_params = {}

        for filters in filter_options:

            for kernel in kernel_options:

                for lr in learning_rates:

                    print(

                        f"\nTesting -> Filters:{filters}, Kernel:{kernel}, LR:{lr}"
                    )

                    self.build_model(

                        (
                            X_train.shape[1],
                            1
                        ),

                        filters_1=filters,

                        filters_2=filters * 2,

                        kernel_size=kernel,

                        learning_rate=lr
                    )

                    self.train_model(

                        X_train,
                        y_train,
                        epochs=10
                    )

                    predictions = (

                        self.model.predict(
                            X_test
                        ) > 0.5

                    ).astype(int)

                    accuracy = accuracy_score(

                        y_test,
                        predictions
                    )

                    print(
                        f"Accuracy: {accuracy}"
                    )

                    if accuracy > best_accuracy:

                        best_accuracy = accuracy

                        best_params = {

                            'filters': filters,

                            'kernel_size': kernel,

                            'learning_rate': lr
                        }

        print(
            "\n===== BEST PARAMETERS ====="
        )

        print(best_params)

        print(
            f"Best Accuracy: {best_accuracy}"
        )

    # =====================================
    # COMPLETE PIPELINE
    # =====================================

    def run_pipeline(

        self,
        X_train,
        X_test,
        y_train,
        y_test

    ):

        # =====================================
        # RESHAPE
        # =====================================

        X_train, X_test = self.reshape_data(

            X_train,
            X_test
        )

        # =====================================
        # BUILD MODEL
        # =====================================

        self.build_model(

            (
                X_train.shape[1],
                1
            )
        )

        # =====================================
        # TRAIN MODEL
        # =====================================

        self.train_model(

            X_train,
            y_train
        )

        # =====================================
        # EVALUATE
        # =====================================

        self.evaluate_model(

            X_test,
            y_test
        )

        # =====================================
        # LOGISTIC REGRESSION COMPARISON
        # =====================================

        self.logistic_regression_baseline(

            X_train,
            X_test,
            y_train,
            y_test
        )

        # =====================================
        # HYPERPARAMETER TUNING
        # =====================================

        self.hyperparameter_tuning(

            X_train,
            X_test,
            y_train,
            y_test
        )

