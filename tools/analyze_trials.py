import pandas as pd
import json

def analyze_trials(file_path):
    """
    Analyzes the trial summary CSV file and prints a summary of the results.

    Args:
        file_path (str): The path to the trials_summary.csv file.
    """
    try:
        df = pd.read_csv(file_path)

        print("--- Trials Summary Analysis ---")
        print(f"Total trials: {len(df)}")

        # --- Sharpe Ratio Analysis ---
        print("\n--- Overall Sharpe Ratio ---")
        print(df['overall_sharpe'].describe())

        best_sharpe_trial = df.loc[df['overall_sharpe'].idxmax()]
        print(f"\nBest Sharpe Ratio Trial: #{best_sharpe_trial['trial_number']}")
        print(f"  - Overall Sharpe: {best_sharpe_trial['overall_sharpe']:.4f}")
        print(f"  - Stability Score: {best_sharpe_trial['stability_score']:.4f}")
        
        # --- Stability Score Analysis ---
        print("\n--- Stability Score ---")
        print(df['stability_score'].describe())
        
        # --- Zero Sharpe Trials ---
        zero_sharpe_trials = df[df['overall_sharpe'] == 0]
        print(f"\nNumber of trials with Zero Sharpe Ratio: {len(zero_sharpe_trials)} ({len(zero_sharpe_trials)/len(df):.2%})")

        # --- Negative Sharpe Trials ---
        negative_sharpe_trials = df[df['overall_sharpe'] < 0]
        print(f"Number of trials with Negative Sharpe Ratio: {len(negative_sharpe_trials)} ({len(negative_sharpe_trials)/len(df):.2%})")

        # --- Positive Sharpe Trials ---
        positive_sharpe_trials = df[df['overall_sharpe'] > 0]
        print(f"Number of trials with Positive Sharpe Ratio: {len(positive_sharpe_trials)} ({len(positive_sharpe_trials)/len(df):.2%})")


    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    analyze_trials("e:/pycode/auto-trader-okx/models/trials_summary.csv")
