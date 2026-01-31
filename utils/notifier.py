import os
import logging
import requests
import json
from typing import Optional

class Notifier:
    """
    Generic Notification Manager.
    Supports: WeCom (Enterprise WeChat), PushPlus
    """
    def __init__(self):
        self.logger = logging.getLogger("Notifier")
        self.pushplus_token = os.getenv("PUSHPLUS_TOKEN", "")
        
        # WeCom Config
        self.wecom_cid = os.getenv("WECOM_CORPID", "")
        self.wecom_secret = os.getenv("WECOM_SECRET", "")
        self.wecom_aid = os.getenv("WECOM_AGENTID", "")
        
    def send(self, title: str, content: str, template: str = "markdown"):
        """
        Send notification via configured channels. Priority: WeCom > PushPlus.
        """
        if self.wecom_cid and self.wecom_secret and self.wecom_aid:
            self._send_wecom(title, content)
        elif self.pushplus_token:
            self._send_pushplus(title, content, template)
        else:
            self.logger.warning("No notification tokens configured (WeCom/PushPlus). Notification skipped.")

    def _send_wecom(self, title: str, content: str):
        """
        Send message via Enterprise WeChat (WeCom)
        Docs: https://developer.work.weixin.qq.com/document/path/90236
        """
        try:
            # 1. Get Access Token
            token_url = f"https://qyapi.weixin.qq.com/cgi-bin/gettoken?corpid={self.wecom_cid}&corpsecret={self.wecom_secret}"
            r = requests.get(token_url, timeout=10)
            token_data = r.json()
            
            if token_data.get('errcode') != 0:
                self.logger.error(f"❌ WeCom Token Error: {token_data}")
                return
                
            access_token = token_data.get('access_token')
            
            # 2. Send Message (Markdown)
            send_url = f"https://qyapi.weixin.qq.com/cgi-bin/message/send?access_token={access_token}"
            
            # WeCom Markdown Format: Only supports bold, italic, link, code.
            # Convert title to bold header
            md_content = f"# {title}\n\n{content}"
            
            payload = {
                "touser": "@all",
                "msgtype": "markdown",
                "agentid": self.wecom_aid,
                "markdown": {
                    "content": md_content
                }
            }
            
            r = requests.post(send_url, json=payload, timeout=10)
            resp = r.json()
            
            if resp.get('errcode') == 0:
                self.logger.info(f"✅ Notification sent via WeCom: {title}")
            else:
                self.logger.error(f"❌ WeCom Send Error: {resp}")
                
        except Exception as e:
            self.logger.error(f"❌ Failed to send WeCom notification: {e}")

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
