# Session Context: Docker Fix & Gemma 3 Model Migration
**Date**: 2026-01-30
**Topic**: Troubleshooting Deployment, SDK Upgrade, and LLM Model Selection

## 1. Docker Deployment Issues
### Problem
- **Error**: `KeyError: 'ContainerConfig'` during `docker-compose up`.
- **Cause**: Incompatibility between Docker Compose v1 (Legacy) and images built with Docker BuildKit.

### Solution
- Modified `deploy/start.sh` to force legacy build mode when V2 is not detected.
- **Env Vars Added**:
  ```bash
  export DOCKER_BUILDKIT=0
  export COMPOSE_DOCKER_CLI_BUILD=0
  ```
- **Status**: Resolved. Users can deploy using `./deploy/start.sh --build`.

---

## 2. LLM SDK Upgrade
### Problem
- Old SDK `google.generativeai` is deprecated and flagged warnings.

### Solution
- Upgraded to `google.genai` (Native Python SDK).
- **Files Updated**:
  - `requirements.txt`: Added `google-genai`.
  - `utils/llm_client_genai.py`: Rewrote `GeminiClient` to use the new `genai.Client`.

---

## 3. Rate Limit & Model Availability Crisis
### Issue Roadmap
1. **Gemini Flash Latest**: User reported `429 Resource Exhausted` with `limit: 20` (RPD).
   - *Analysis*: Experimental/Preview models have extremely low quotas.
2. **Gemini 2.0 Flash**: Attempted switch, but got `429 limit: 0`.
   - *Analysis*: Model unavailable for the specific API Key/Project.
3. **Gemini 1.5 Flash**: Attempted switch, got `404 Not Found`.
   - *Analysis*: Model naming or access issue.
4. **Discovery**: Ran `scripts/check_models.py` and found **Gemma 3 27B IT**.
   - *Quota*: **14,400 Requests Per Day** (vs 20 for Gemini).

### Final Decision in .env
```bash
LLM_MODEL=gemma-3-27b-it
```

---

## 4. Gemma 3 Compatibility Fixes
### Problem 1: System Instruction (400 Error)
- **Error**: `Developer instruction is not enabled for models/gemma-3-27b-it`.
- **Cause**: Gemma models do not support the separate `system_instruction` API parameter.
- **Fix**: In `llm_client_genai.py`, manually detected `gemma` in model name and prepended the system prompt to the user message.
  ```python
  if "gemma" in self.model_name.lower():
      final_contents = f"System: {system_prompt}\n\nUser: {prompt}"
  ```

### Problem 2: Client Routing
- **Bug**: `llm_supervisor.py` was checking `if 'gemini' in model_name`.
- **Result**: `gemma` models fell back to the old OpenAI-compatible client, causing failures.
- **Fix**: Updated condition to `if 'gemini' in ... or 'gemma' in ...`.

---

## 5. Verification
- **Logs Confirmed Success**:
  ```
  INFO - ✅ LLM Update: Risk=0.7, Bias=short
  ```
- **Capability Assessment**:
  - Gemma 3 27B is capable of JSON instruction following and reasoning.
  - Suitable for market sentiment analysis and macro data processing.
  - High quota makes it ideal for high-frequency bots (e.g., Polymarket, Scalping).

---

## 6. Next Steps / Recommendations
- **Polymarket Bot**: Feasible using Gemma 3 for news scanning/RAG.
- **Macro Analysis**: Can setup a workflow to ingest Calendar/News -> Gemma 3 -> Market Direction Prediction.
