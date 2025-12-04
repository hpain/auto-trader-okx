import os
import logging
import pandas as pd
import shutil
import json
import numpy as np
from datetime import datetime
from joblib import load

# 导入项目模块
from data.binance import get_klines_bian
from exchange.okx_exchange import OKXExchange
from features.feature_engineering import generate_features, make_supervised
from models.evolution import train_evolve
from utils.backtest import run_backtest

# 配置日志
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def _calculate_sharpe(returns: pd.Series, periods_per_year: int = 365*24) -> float:
    """根据回报序列计算年化夏普比率"""
    if returns.std() == 0 or len(returns.loc[returns != 0]) < 10:
        return 0.0
    annualization_factor = np.sqrt(periods_per_year)
    sharpe = (returns.mean() / returns.std()) * annualization_factor
    return sharpe if np.isfinite(sharpe) else 0.0

def validate_model(
    new_model_dir: str,
    prod_model_dir: str,
    validation_df: pd.DataFrame,
    feature_cols: list[str],
    improvement_threshold: float = 0.10
) -> bool:
    """
    验证新模型是否显著优于生产模型。
    """
    logging.info("--- [Validation] --- ")

    # 定义模型和元数据路径
    new_model_path = os.path.join(new_model_dir, "best_model.pkl")
    new_meta_path = os.path.join(new_model_dir, "metadata.json")
    prod_model_path = os.path.join(prod_model_dir, "best_model.pkl")
    prod_meta_path = os.path.join(prod_model_dir, "metadata.json")

    # 检查生产模型是否存在
    if not os.path.exists(prod_model_path) or not os.path.exists(prod_meta_path):
        logging.warning("Production model not found. Approving new model by default.")
        return True

    try:
        # 加载模型和元数据
        new_model = load(new_model_path)
        prod_model = load(prod_model_path)
        with open(new_meta_path, 'r') as f: new_meta = json.load(f)
        with open(prod_meta_path, 'r') as f: prod_meta = json.load(f)

        # 准备验证数据
        # 确保验证集中的特征与模型训练时一致
        X_val = validation_df[feature_cols]

        # 评估新模型
        logging.info("Evaluating new model...")
        new_preds = pd.Series(new_model.predict(X_val), index=X_val.index)
        new_probs = new_model.predict_proba(X_val)[:, 1]
        _, _, _, _, new_returns = run_backtest(
            predictions=new_preds,
            probabilities=new_probs,
            test_data=validation_df,
            confidence_threshold=new_meta['best_params']['confidence_threshold']
        )
        new_sharpe = _calculate_sharpe(new_returns)
        logging.info(f"New model Sharpe Ratio on validation set: {new_sharpe:.4f}")

        # 评估生产模型
        logging.info("Evaluating production model...")
        # 旧模型可能使用不同的特征集，但在这里我们强制其在相同的新特征集上评估
        # 这是一个简化处理，更复杂的场景可能需要保存旧模型的特征列表
        # A potential issue is that the production model might not have been trained on all features in `feature_cols`.
        # We will try to predict, but if it fails, we assume the new model is better.
        try:
            prod_preds = pd.Series(prod_model.predict(X_val), index=X_val.index)
            prod_probs = prod_model.predict_proba(X_val)[:, 1]
            _, _, _, _, prod_returns = run_backtest(
                predictions=prod_preds,
                probabilities=prod_probs,
                test_data=validation_df,
                confidence_threshold=prod_meta['best_params']['confidence_threshold']
            )
            prod_sharpe = _calculate_sharpe(prod_returns)
            logging.info(f"Production model Sharpe Ratio on validation set: {prod_sharpe:.4f}")
        except Exception as e:
            logging.warning(f"Could not evaluate production model on new feature set: {e}. Assuming new model is better.")
            prod_sharpe = -np.inf


        # 决策
        is_better = new_sharpe > prod_sharpe * (1 + improvement_threshold)
        if is_better:
            logging.info(f"New model is significantly better (>{improvement_threshold*100}% improvement). Approval GRANTED.")
        else:
            logging.info("New model is not significantly better. Approval DENIED.")

        return is_better

    except Exception as e:
        logging.error(f"An error occurred during validation: {e}", exc_info=True)
        return False # 出错时，不批准新模型

def run_training_pipeline(n_trials: int = 50, data_years: float = 2.0, validation_split: float = 0.15):
    """
    执行完整的“训练-验证-部署”流水线。
    """
    # --- 训练锁机制 ---
    base_model_dir = "models"
    lock_file_path = os.path.join(base_model_dir, "training.lock")

    # 检查锁文件，防止重复启动
    if os.path.exists(lock_file_path):
        logging.warning(f"Training lock file found at {lock_file_path}. Another training process may be running. Aborting.")
        return {"status": "SKIPPED", "reason": "Lock file exists."}

    # 创建锁文件
    with open(lock_file_path, 'w') as f:
        f.write(json.dumps({"pid": os.getpid(), "start_time": datetime.now().isoformat()}))

    logging.info("===== Starting Model Training Pipeline =====")

    # 1. 设置目录
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    staging_dir = os.path.join(base_model_dir, f"staging_{timestamp}")
    os.makedirs(staging_dir, exist_ok=True)
    logging.info(f"Staging directory created at: {staging_dir}")

    try:
        # 2. 加载和准备数据
        logging.info(f"Loading {data_years} years of data...")
        client = OKXExchange(sandbox=True)
        raw_data = get_klines_bian(client, symbol="BTC-USDT", interval="1H", years=data_years)
        if raw_data is None or raw_data.empty:
            raise ValueError("Failed to load data.")

        logging.info("Generating features and labels...")
        feature_data = generate_features(raw_data)
        supervised_data = make_supervised(feature_data, horizon=1, threshold=0.005)

        # 动态确定特征列
        feature_cols = [col for col in supervised_data.columns if col not in ['y', 'future_high', 'future_low', 'future_close', 'future_ret']]
        logging.info(f"Using {len(feature_cols)} features for training.")


        # 分割训练/验证集
        val_size = int(len(supervised_data) * validation_split)
        train_df = supervised_data.iloc[:-val_size]
        validation_df = supervised_data.iloc[-val_size:]
        logging.info(f"Data split: {len(train_df)} training samples, {len(validation_df)} validation samples.")

        # 3. 执行模型演进和训练
        logging.info(f"Starting model evolution with {n_trials} trials on training set...")
        train_evolve(
            data=train_df,
            feature_cols=feature_cols, # 使用动态特征列表
            out_dir=staging_dir,
            n_trials=n_trials,
            model_list=[], profit_threshold=0.0, confidence_threshold=0.5,
            stop_loss_pct=0.05, take_profit_pct=0.10, max_drawdown_limit=0.2,
            success_rate_threshold=0.5, interval="1H"
        )
        logging.info("Model evolution finished.")

        # 4. 验证新模型
        is_new_model_better = validate_model(
            staging_dir, base_model_dir, validation_df, feature_cols=feature_cols
        )

        # 5. 部署模型
        if is_new_model_better:
            logging.info("Deploying new model to production...")
            backup_timestamp = datetime.now().strftime("%Y%m%d%H%M%S.bak")
            files_to_deploy = ["best_model.pkl", "metadata.json", "feature_importance.png", "trials_summary.csv", "best_trial_details.json"]

            for filename in files_to_deploy:
                prod_path = os.path.join(base_model_dir, filename)
                staging_path = os.path.join(staging_dir, filename)

                if os.path.exists(staging_path):
                    if os.path.exists(prod_path):
                        backup_path = f"{prod_path}.{backup_timestamp}"
                        logging.info(f"Backing up {prod_path} to {backup_path}")
                        shutil.move(prod_path, backup_path)

                    logging.info(f"Deploying {staging_path} to {prod_path}")
                    shutil.move(staging_path, prod_path)

            logging.info("Deployment successful.")

            # 创建标志文件，通知主进程模型已更新
            flag_path = os.path.join(base_model_dir, 'model_update_complete.flag')
            with open(flag_path, 'w') as f:
                f.write(datetime.now().isoformat())
            logging.info(f"Created model update flag file at: {flag_path}")

        else:
            logging.info("New model did not pass validation. Discarding.")

    except Exception as e:
        logging.error(f"An error occurred in the training pipeline: {e}", exc_info=True)
        return {"status": "FAILED", "error": str(e)}
    finally:
        # 6. 清理临时目录
        if os.path.exists(staging_dir):
            logging.info(f"Cleaning up staging directory: {staging_dir}")
            shutil.rmtree(staging_dir)
        
        # --- 释放训练锁 ---
        if os.path.exists(lock_file_path):
            os.remove(lock_file_path)
            logging.info("Training lock file removed.")

    logging.info("===== Model Training Pipeline Finished =====")
    return {"status": "SUCCESS"}

if __name__ == '__main__':
    run_training_pipeline(n_trials=10)
