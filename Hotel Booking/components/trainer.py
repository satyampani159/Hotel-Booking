from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from entity.config_entity import TrainingConfig
from utils.helpers import save_model, ensure_dir
from logger.log_config import get_logger
from exception.custom_exception import CustomException
import os

logger = get_logger("trainer")


def lasso_feature_selection(X, y, alpha=0.005):
    sel = SelectFromModel(Lasso(alpha=alpha))
    sel.fit(X, y)
    support = sel.get_support()
    selected = X.columns[support]
    return list(selected)


class Trainer:
    def __init__(self, config: TrainingConfig):
        self.config = config

    def train(self, X, y):
        try:
            from components.visualizations import plot_confusion_matrix
            from components.output_reports import (
                print_feature_importance, 
                print_model_training_summary,
                print_cross_validation_summary
            )
            
            # feature selection
            logger.info("Running Lasso for feature selection")
            try:
                selected = lasso_feature_selection(X, y)
            except Exception:
                # fallback: keep all if Lasso fails
                selected = list(X.columns)
            
            X_sel = X[selected]
            
            # Print selected features
            print_feature_importance(selected, "SELECTED FEATURES (Lasso)")

            # split
            X_train, X_test, y_train, y_test = train_test_split(
                X_sel, y, test_size=self.config.test_size, random_state=self.config.random_state
            )
            
            print(f"\n{'='*80}")
            print("TRAIN-TEST SPLIT")
            print(f"{'='*80}")
            print(f"Training set shape: {X_train.shape}")
            print(f"Test set shape: {X_test.shape}")
            print(f"Training/Test ratio: {len(X_train)}/{len(X_test)}")
            print("")

            # train logistic regression
            model = LogisticRegression(max_iter=1000)
            model.fit(X_train, y_train)
            
            print(f"Model trained: LogisticRegression(max_iter=1000)")
            print(f"Model coefficients shape: {model.coef_.shape}")
            print(f"Intercept: {model.intercept_}")
            print("")

            preds = model.predict(X_test)
            acc = accuracy_score(y_test, preds)
            cm = confusion_matrix(y_test, preds)
            
            # Print training summary
            print_model_training_summary(X_train.shape, X_test.shape, "LogisticRegression", acc, cm)

            # Cross-validation
            try:
                from sklearn.model_selection import cross_val_score
                cv_scores = cross_val_score(model, X_sel, y, cv=10)
                print_cross_validation_summary(cv_scores, cv_scores.mean(), cv_scores.std())
            except Exception as e:
                logger.warning(f"Could not perform cross-validation: {e}")
                cv_scores = None

            # save model
            model_dir = self.config.model_dir
            ensure_dir(model_dir)
            model_path = os.path.join(model_dir, self.config.model_name)
            save_model(model, model_path)
            logger.info(f"Model saved at {model_path}")
            
            # generate confusion matrix plot
            try:
                plot_confusion_matrix(cm)
            except Exception as e:
                logger.warning(f"Could not generate confusion matrix plot: {e}")

            return {"accuracy": acc, "confusion_matrix": cm, "model_path": model_path, "cv_scores": cv_scores}
        except Exception as e:
            raise CustomException("Error during training", e)
