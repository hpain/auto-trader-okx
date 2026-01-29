import os
import requests
import json
import logging
import time

class LLMClient:
    """
    通用 LLM 客户端，适配 OpenAI 接囗格式 (DeepSeek/ChatGPT)。
    """
    def __init__(self, api_key=None, base_url="https://api.deepseek.com", model="deepseek-chat"):
        self.api_key = api_key or os.getenv("LLM_API_KEY")
        self.base_url = base_url
        self.model = model
        self.logger = logging.getLogger("LLMClient")
        
        if not self.api_key:
            self.logger.warning("LLM API Key not found. Please set LLM_API_KEY env var or pass in init.")

    def query(self, prompt, system_prompt="You are a helpful assistant.", temperature=0.7, max_retries=3):
        """
        发送请求到 LLM。
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            "temperature": temperature,
            "stream": False
        }
        
        for attempt in range(max_retries):
            try:
                # Assuming standard OpenAI chat completion endpoint
                url = f"{self.base_url}/v1/chat/completions" if not self.base_url.endswith("/chat/completions") else self.base_url
                
                response = requests.post(url, headers=headers, json=data, timeout=60)
                
                if response.status_code == 200:
                    result = response.json()
                    content = result['choices'][0]['message']['content']
                    return content
                else:
                    self.logger.error(f"LLM Request failed (Attempt {attempt+1}): {response.status_code} - {response.text}")
                    
            except Exception as e:
                self.logger.error(f"LLM Connection error (Attempt {attempt+1}): {e}")
            
            time.sleep(2 * (attempt + 1)) # Backoff
            
        return None

    def query_json(self, prompt, system_prompt):
        """
        请求并强制解析 JSON 格式
        """
        content = self.query(prompt, system_prompt, temperature=0.1) # Low temp for JSON
        if not content:
            return None
            
        # Clean markdown code blocks if present
        cleaned_content = content.replace("```json", "").replace("```", "").strip()
        
        try:
            return json.loads(cleaned_content)
        except json.JSONDecodeError:
            self.logger.error(f"Failed to parse JSON from LLM response: {content}")
            return None
