import os
import logging
import time
import json
from google import genai
from google.genai import types

class GeminiClient:
    """
    Client for Google's Generative AI SDK (new google.genai package).
    Replaces the deprecated google.generativeai package.
    """
    def __init__(self, api_key=None, model="gemini-2.0-flash"):
        self.api_key = api_key or os.getenv("LLM_API_KEY")
        self.model_name = model
        self.logger = logging.getLogger("GeminiClient")
        
        if not self.api_key:
            self.logger.warning("LLM API Key not found. Please set LLM_API_KEY env var.")
            self.client = None
        else:
            self.client = genai.Client(api_key=self.api_key)

    def query(self, prompt, system_prompt="You are a helpful assistant.", temperature=0.7, max_retries=3):
        """
        Send request to Gemini using the new SDK.
        """
        if not self.client:
            self.logger.error("Client not initialized (missing API key)")
            return None
            
        for attempt in range(max_retries):
            try:
                # Gemma 3 models do not support 'system_instruction' param
                # We must manually prepend it to the prompt
                is_gemma = "gemma" in self.model_name.lower()
                
                final_contents = prompt
                config_params = {"temperature": temperature}

                if is_gemma:
                    # Manually prepend system prompt
                    if system_prompt:
                         final_contents = f"System: {system_prompt}\n\nUser: {prompt}"
                else:
                    # Use native system_instruction for Gemini models
                    config_params["system_instruction"] = system_prompt

                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=final_contents,
                    config=types.GenerateContentConfig(**config_params)
                )
                return response.text
            except Exception as e:
                error_str = str(e)
                self.logger.error(f"Gemini Request failed (Attempt {attempt+1}): {error_str}")
                
                # Intelligent Retry for Rate Limits (429)
                import re
                if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
                    # Try to find "Please retry in X s"
                    match = re.search(r"retry in ([\d\.]+)s", error_str)
                    if match:
                        wait_time = float(match.group(1)) + 1.0 # Add 1s buffer
                        self.logger.warning(f"Rate Limit Hit. Sleeping for {wait_time:.1f}s...")
                        time.sleep(wait_time)
                        continue
                
                time.sleep(2 * (attempt + 1))
                
        return None

    def query_json(self, prompt, system_prompt):
        """
        Request JSON output specifically.
        """
        # Force JSON instruction validation
        if "JSON" not in system_prompt and "json" not in system_prompt:
            system_prompt += "\nRespond in strictly valid JSON format."

        content = self.query(prompt, system_prompt, temperature=0.1)
        if not content:
            return None
            
        # Clean markdown code blocks
        cleaned_content = content.replace("```json", "").replace("```", "").strip()
        
        try:
            return json.loads(cleaned_content)
        except json.JSONDecodeError:
            self.logger.error(f"Failed to parse JSON from Gemini response: {cleaned_content[:200]}...")
            return None
