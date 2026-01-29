import os
import logging
import time
import json
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold

class GeminiClient:
    """
    Client for Google's Generative AI SDK (native).
    Replacing the REST-based LLMClient for better stability with Gemini.
    """
    def __init__(self, api_key=None, model="gemini-1.5-flash"):
        self.api_key = api_key or os.getenv("LLM_API_KEY")
        self.model_name = model
        self.logger = logging.getLogger("GeminiClient")
        
        if not self.api_key:
            self.logger.warning("LLM API Key not found. Please set LLM_API_KEY env var.")
        else:
            genai.configure(api_key=self.api_key)
            
        self.model = genai.GenerativeModel(self.model_name)

    def query(self, prompt, system_prompt="You are a helpful assistant.", temperature=0.7, max_retries=3):
        """
        Send request to Gemini.
        Note: System instructions are better passed in init but for dynamic compat we prepend them.
        """
        combined_prompt = f"{system_prompt}\n\n{prompt}"
        
        for attempt in range(max_retries):
            try:
                response = self.model.generate_content(
                    combined_prompt,
                    generation_config=genai.types.GenerationConfig(
                        temperature=temperature
                    ),
                    safety_settings={
                        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
                    }
                )
                return response.text
            except Exception as e:
                self.logger.error(f"Gemini Request failed (Attempt {attempt+1}): {e}")
                time.sleep(2 * (attempt + 1))
                
        return None

    def query_json(self, prompt, system_prompt):
        """
        Request JSON output specifically.
        Gemini 1.5 supports native JSON mode, but we'll use text prompting for compatibility with older patterns first.
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
