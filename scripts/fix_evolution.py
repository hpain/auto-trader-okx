import os

file_path = r'e:\pycode\auto-trader-okx\research\improved_evolution.py'
new_logic = r'''    # --- Evolutionary Factor Mining (Optional) ---
    if args.mining_generations > 0:
        logging.info(f"--- Starting Evolutionary Factor Mining (Generations: {args.mining_generations}) ---")
        try:
            from research.factor_mining import FactorMiner
            
            # Prepare Target for Mining (Forward Return)
            logging.info("Preparing data for factor mining...")
            mining_df = dfm.copy()
            # Predict next bar return
            mining_df['target'] = mining_df['close'].shift(-1) / mining_df['close'] - 1
            mining_df = mining_df.dropna()
            
            # Select features to evolve from (exclude non-feature columns)
            exclude_cols = ['target', 'y', 'future_ret', 'future_high', 'future_low', 'future_close', 'date', 'open', 'high', 'low', 'close', 'volume', 'time', 'day']
            feature_cols = [c for c in mining_df.columns if c not in exclude_cols and not c.startswith('onchain_')]
            
            # Initial Run
            logging.info(f"Mining on {len(mining_df)} rows with {len(feature_cols)} base features.")
            
            miner = FactorMiner(
                generations=args.mining_generations, 
                population_size=max(500, args.trials * 5), 
                n_components=10,
                random_state=42
            )
            
            miner.fit(mining_df[feature_cols], mining_df['target'], feature_names=feature_cols)
            miner.extract_best_factors(feature_names=feature_cols)
            
            logging.info("Reloading features to include newly mined factors...")
            # Re-run generate_features to pick up the new JSON automatically
            dfm = generate_features(dfp, news_csv_path=news_csv_path, feature_dfs=feature_dfs, derivatives_dfs=derivatives_dfs, onchain_dfs=onchain_dfs)
            
        except ImportError:
            logging.error("Failed to import FactorMiner. Is 'gplearn' installed? Run 'pip install gplearn'.")
        except Exception as e:
            logging.error(f"Factor Mining failed: {e}", exc_info=True)
'''

with open(file_path, 'r', encoding='utf-8') as f:
    lines = f.readlines()

start_idx = -1
end_idx = -1

# Find the start: first occurrence of "# --- Mined Feature Integration ---"
for i, line in enumerate(lines):
    if "# --- Mined Feature Integration ---" in line:
        start_idx = i
        break

# Find the end: specifically the duplicate exception block or just safely after the block
# The block ends before "except Exception as e:" for the OUTER Loop? 
# No, let's look for the specific error message line: 'logging.warning(f"Error during mined feature integration: {e}")'
# valid end is the LAST occurrence of that line + 1 (because of the duplicate)

for i in range(len(lines) - 1, -1, -1):
    if 'logging.warning(f"Error during mined feature integration: {e}")' in lines[i]:
        end_idx = i + 1 # Include this line to start AFTER it? No, we want to replace IT too.
        # But wait, we want to replace inclusive.
        # So slice lines[:start_idx] + new + lines[end_idx:]
        end_idx = i + 1 
        break

if start_idx != -1 and end_idx != -1:
    print(f"Replacing lines {start_idx} to {end_idx}")
    new_lines = lines[:start_idx] + [new_logic + '\n'] + lines[end_idx:]
    
    with open(file_path, 'w', encoding='utf-8') as f:
        f.writelines(new_lines)
    print("Success!")
else:
    print("Failed to find block.")
    print(f"Start: {start_idx}, End: {end_idx}")
