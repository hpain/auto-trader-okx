import os
import logging
import requests
import json
from typing import Optional

class Notifier:
    """
    Generic Notification Manager.
    Currently supports: PushPlus (WeChat)
    """
    def __init__(self):
        self.logger = logging.getLogger("Notifier")
        self.pushplus_token = os.getenv("PUSHPLUS_TOKEN", "")
        
    def send(self, title: str, content: str, template: str = "markdown"):
        """
        Send notification via configured channels.
        """
        if self.pushplus_token:
            self._send_pushplus(title, content, template)
        else:
            self.logger.warning("No notification tokens configured (PUSHPLUS_TOKEN). Notification skipped.")

    def _send_pushplus(self, title: str, content: str, template: str):
        """
        Send message via PushPlus (http://www.pushplus.plus/)
        """
        url = "http://www.pushplus.plus/send"
        
        # PushPlus expects 'content' to be the body.
        # If template is markdown, we should ensure content is formatted correctly.
        
        payload = {
            "token": self.pushplus_token,
            "title": title,
            "content": content,
            "template": template
        }
        
        try:
            response = requests.post(url, json=payload, timeout=10)
            resp_json = response.json()
            
            if resp_json.get("code") == 200:
                self.logger.info(f"✅ Notification sent via PushPlus: {title}")
            else:
                self.logger.error(f"❌ PushPlus failed: {resp_json}")
                
        except Exception as e:
            self.logger.error(f"❌ Failed to send PushPlus notification: {e}")

# Global instance
_notifier_instance = None

def get_notifier():
    global _notifier_instance
    if _notifier_instance is None:
        _notifier_instance = Notifier()
    return _notifier_instance
