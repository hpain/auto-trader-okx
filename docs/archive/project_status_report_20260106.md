# Project Status Report
**Date**: 2026-01-06
**Status**: LIVE (Dual-Process Architecture)

## 1. Executive Summary
We have successfully transitioned the project from a research-heavy, unstable monolith to a robust **Dual-Process Architecture**.
- **Process A (Structural Alpha)**: A low-risk Funding Arbitrage Bot (`auto-trader-arb`) is active on ETH/USDT Swap.
- **Process B (Predictive Alpha)**: A high-risk ML Bot (`auto-trader-ml`) is running in parallel on BTC/USDT Spot (Mock Mode) for data collection.

Both bots are stable, verified, and operating on the VPS with resource limits enforcement.

## 2. Recent Accomplishments (Phase 2 & 3-1)
### Phase 2: Predictive Alpha (Complete)
- **Model Upgrade**: Deployed Transformer V2 (139 inputs, 120-window sequence).
- **Stability Fix**: Resolved chronic OOM (Exit 137) crashes by limiting Docker memory to 2GB.
- **Verification**: Validated correct model loading and inference dimensions on the live server.

### Phase 3-1: Optimization (Complete)
- **Dynamic Logic**: Implemented "Breakeven" logic for Arb Bot.
    - Instead of a hardcoded 0.15% threshold, the bot calculates: `Entry = Cost / (3 * RecoveryDays)`.
    - Current settings (0.3% Cost, 5 Days) result in a smart entry threshold of ~0.02%.
- **Live Verification**:
    - Resolved `KeyError: 'ContainerConfig'` via full Docker reset.
    - Resolved `TypeError` (Missing Abstract Methods) via code sync and cache clearing on VPS.
    - Confirmed via logs: `Using DYNAMIC PROFIT CALCULATION`.

## 3. Validated Conclusions
1.  **Docker Layers Matter**: The persistent errors were caused by stale Docker layers and Python `__pycache__` referencing old code. `start.sh --build` is the robust fix.
2.  **Mathematics > Hardcoding**: The Dynamic Logic is superior because it adapts to transaction cost changes (if we lower fees via VIP tiers) without code changes.
3.  **Isolation Works**: Running two separate containers prevents one bot's crash (e.g. ML OOM) from killing the profit-generating Arb bot.

## 4. Next Steps
1.  **Monitor Performance (Passive)**: Let the Arb bot run for 5-7 days to prove stability and catch the first real arb opportunity.
2.  **Phase 3-2 (ML Refinement)**:
    - Analyze the *prediction accuracy* of the live running Transformer model.
    - If memory pressure rises again, implement feature reduction (PCA or selection).

## 5. Artifacts
Attached in this folder are the current snapshots of:
- `task.md`: Complete work checklist.
- `implementation_plan.md`: Technical design specs.
- `walkthrough.md`: Detailed system architecture and verification proofs.

## 6. Operational Context (For AI Assistant)
**Critical Infrastructure Details**
- **VPS IP**: `35.216.120.161`
- **SSH Key**: `C:\Users\hk\.ssh\hk-ed25519-key`
- **SSH Command**: `ssh -i "C:\Users\hk\.ssh\hk-ed25519-key" 35.216.120.161`
- **Remote Directory**: `/root/auto-trader-okx-lt` (accessed via `sudo -i` then `cd auto-trader-okx-lt`)
- **Git Branch**: `main_lt` (Must sync this branch to VPS)

**Standard Operating Procedures**
- **Safe Deployment**: Always use `bash ./deploy/start.sh --build` locally or on VPS to ensure Docker cache is cleared.
- **Log Monitoring**: `docker-compose logs --tail=50 -f auto-trader-arb` (or `auto-trader-ml`).
- **Code Locations**:
    - **Strategy Logic**: `strategies/funding_arb.py` (The logic verified in P3-1).
    - **Runner Script**: `run_live_simple.py` (The minimal runner).
    - **Config**: `config/settings.yaml` (Shared settings).

**Known Quirks**
- **Docker Cache**: VPS has sticky Python bytecode cache. If code changes don't reflect, delete `__pycache__` and use `--build`.
- **Memory Limit**: `auto-trader-ml` MUST have `limits: memory: 2G` in `docker-compose.yml` or it will OOM crash the server.

