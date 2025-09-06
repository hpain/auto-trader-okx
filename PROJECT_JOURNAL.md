# Project Journal & Continuity Log

This document serves as a persistent memory for the Gemini agent to ensure project continuity across sessions. It tracks goals, status, key decisions, and future plans.

##  प्रोजेक्ट लक्ष्य

Develop a Python-based automated trading bot. The bot uses a machine learning model (LightGBM) trained on historical data from Binance to generate trading signals, and executes those trades on the OKX exchange. The system must be robust, configurable, and well-documented.

## Codebase Notes

- **Data Strategy:** The project cleverly uses two data sources. Historical data for model training is sourced from Binance (`data/binance.py`) due to its depth. Live data for trading execution is fetched from OKX (`data/okx.py`).
- **Trading Logic:** The core trading logic is in `trader/executor.py`. It operates on a primary ML-based signal but includes a fallback to a simple Moving Average (MA) strategy if the model or its prediction fails.
- **Model Evolution:** Model training and hyperparameter optimization are handled by `research/evolve.py`, which uses the Optuna library.
- **Configuration:** Project settings, including API keys and file paths, are centralized in `config/settings.yaml` and loaded by `config/__init__.py`. The config module loads the file immediately upon import, which can cause issues in test environments.
- **Backtesting Files:** There are two distinct backtesting files: `utils/backtest.py` is used for the detailed, stop-loss-based evaluation during model optimization. `models/backtest.py` is a simpler, vectorized backtester for quick assessments.
- **Testing Strategy:** Unit tests are built with `pytest`. Due to the eager-loading `config` module, tests for any module that imports `config` (directly or indirectly) require a special setup. The standard pattern is to use `unittest.mock.patch.dict` to mock configuration dictionaries (e.g., `executor.config`) directly within test functions or fixtures. This targeted approach avoids globally mocking file I/O, which can break third-party libraries that also read files on import (e.g., `matplotlib`).

## Session History (Log)

### 2025-09-06

- **Initial Code Review:** Performed a comprehensive review of the entire codebase. Identified strengths (good structure, data strategy, fallback mechanism) and areas for improvement.
- **Refactoring (Code Deduplication):** 
    - Investigated `utils/data_normalization.py` and `features/data_normalization.py`. Found the latter to be empty and safely deleted it.
    - Compared `utils/backtest.py` and `models/backtest.py` and concluded they serve different purposes and should not be merged.
- **Refactoring (Hardcoded Paths):**
    - Centralized file paths into `config/settings.yaml` under a `paths` key.
    - Modified `trader/executor.py`, `research/evolve.py`, and `data/binance.py` to use the new configuration paths.
- **Documentation:**
    - Significantly enhanced the `README.md` to include a project description, setup instructions, usage examples for training and trading, and a full explanation of the `evolve.py` script's command-line arguments.
    - Created a `config/settings.yaml.example` file to improve management of secrets.
- **Exception Handling Strategy:**
    - Discussed the pros and cons of broad vs. specific exception handling.
    - Agreed on a hybrid approach: catch specific, expected errors (`FileNotFoundError`) for graceful recovery, and use a logging `except Exception:` block as a safety net for unexpected errors.
    - Implemented this improved strategy in the `_load_model` function in `trader/executor.py`.
- **Refactoring (Code Cleanup):**
    - Removed the unused `--patience` command-line argument from `research/evolve.py` to improve code clarity.
- **Unit Testing Setup & Execution:**
    - Established a full test suite with `pytest`, achieving 100% test coverage for all core modules (`data_normalization`, `feature_engineering`, `build_model`, `executor`).
    - Developed robust testing patterns, including a targeted mocking strategy (`@patch.dict`) to handle the project's specific configuration loading challenges without disrupting third-party libraries.
    - Systematically debugged and resolved numerous issues across the test suite, resulting in 18 stable and passing tests.
- **Self-Correction & Learning:**
    - Encountered and overcame a persistent inability to update the project journal, leading to the development of new meta-cognitive strategies for problem-solving.
    - Established and saved two core principles for future work: 1) When stuck, seek entirely different methods rather than repeating a failing one. 2) Proactively communicate the specific goal and failing method to the user to facilitate collaborative debugging.

### 2025-09-06 (Evening)

- **Strategy Enhancement (Phase 1 - Research):**
    - Began the "Strategy Enhancement" task, with the initial goal of integrating news sentiment as a new model feature.
    - Analyzed `data/news.py` and confirmed the existence of functions for fetching, scoring, and aggregating news sentiment.
    - Verified that necessary dependencies (`vaderSentiment`, `requests`) are present in `requirements.txt`.
    - Decided to use a local CSV for development to bypass the need for a live `newsapi.org` API key.
    - Attempted to source a sample news CSV file from public datasets on Kaggle and GitHub. Faced significant challenges, including authentication requirements, failing tool calls (`web_fetch`), and 404 errors from guessed URLs.
    - Concluded the session after multiple unsuccessful attempts to download a specific data file. The next step will be to continue the search for a suitable sample news CSV from a new source.

## Current Status

Work has begun on the **Strategy Enhancement** task. The initial analysis of the existing news sentiment code is complete. The project is currently blocked by the need to acquire a sample news dataset for development and testing. The next session will resume with attempts to find a new, more reliable data source.

## 后续计划

With a solid, well-tested foundation, the project is ready for the next phase. The user has prioritized the order of future tasks as follows:
1.  **Strategy Enhancement:** (In Progress) Research and add new types of trading signals or alternative strategies.
2.  **Advanced Risk Management:** Implement more sophisticated risk management rules (e.g., dynamic position sizing, stop-loss/take-profit orders).
3.  **Live Deployment & Monitoring:** Develop scripts and procedures for deploying the bot in a live environment, including robust monitoring and alerting.
4.  **Code Refinement:** Perform a final code review and refactoring pass to improve clarity and maintainability.