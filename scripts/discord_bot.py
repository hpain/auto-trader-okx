"""
Discord Bot for Auto-Trader
============================
Responds to messages in #trading channel, provides:
- Market status and briefings
- Server health and resource status
- Bot activity (ML positions, Arb trades)
- Free-form LLM conversations with full context

Only responds to messages from the authorized user (DISCORD_OWNER_ID).
"""
import os
import sys
import json
import logging
import asyncio
import time
import re
import io
from datetime import datetime, timezone
from pathlib import Path

import discord
from discord.ext import commands
from dotenv import load_dotenv

# Optional dependency
try:
    import psutil
except ImportError:
    psutil = None

# Ensure project root is in path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.llm_client_genai import GeminiClient

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("DiscordBot")

# Config
DISCORD_BOT_TOKEN = os.getenv("Discord_Bot_Token") or os.getenv("DISCORD_BOT_TOKEN")
DISCORD_OWNER_ID = os.getenv("DISCORD_OWNER_ID")
TRADING_CHANNEL_NAME = os.getenv("DISCORD_CHANNEL_NAME", "trading")
LLM_MODEL = os.getenv("LLM_MODEL", "gemma-3-27b-it")

# Data paths
MARKET_CONTEXT_FILE = "data/market_context.json"
BRIEFING_FILE = "data/latest_briefing.md"
ML_LOG_FILE = "logs/trading_cycles.log"
ARB_LOG_FILE = "logs/simple_funding_arb.log"
SUPERVISOR_LOG_FILE = "logs/llm.log"

# Intents
intents = discord.Intents.default()
intents.message_content = True

bot = commands.Bot(command_prefix="!", intents=intents)

# LLM Client
llm_client = None

# --- Data Loading Utilities ---

def load_json_file(path):
    try:
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load {path}: {e}")
    return None

def load_text_file(path, max_chars=2000):
    try:
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read()
                if len(content) > max_chars:
                    return content[-max_chars:] # Get latest part
                return content
    except Exception as e:
        logger.error(f"Failed to load {path}: {e}")
    return None

def get_file_health(path, stale_seconds=3600):
    """Check if a file has been updated recently."""
    try:
        if os.path.exists(path):
            mtime = os.path.getmtime(path)
            age = time.time() - mtime
            status = "✅ Running" if age < stale_seconds else "⚠️ Stale"
            last_active = datetime.fromtimestamp(mtime).strftime('%H:%M:%S')
            return status, f"{int(age)}s ago ({last_active})"
    except:
        pass
    return "❌ Offline", "N/A"

def get_latest_log_line(path, pattern=None):
    """Read the last line of a file, optionally matching a pattern."""
    try:
        if os.path.exists(path):
            with open(path, 'rb') as f:
                f.seek(0, os.SEEK_END)
                size = f.tell()
                # Read last 4KB
                offset = min(size, 4096)
                f.seek(size - offset)
                lines = f.read().decode('utf-8', errors='ignore').splitlines()
                if not lines: return None
                if pattern:
                    for line in reversed(lines):
                        if pattern in line: return line
                return lines[-1]
    except Exception as e:
        logger.error(f"Log tail failed for {path}: {e}")
    return None

def get_ml_position_summary():
    """Extract position info from trading_cycles.log JSON entries."""
    line = get_latest_log_line(ML_LOG_FILE, pattern='{"cycle_id"')
    if line:
        try:
            # Extract JSON from line (sometimes logs have prefix)
            if '{' in line:
                json_str = line[line.find('{'):]
                data = json.loads(json_str)
                portfolio = data.get('portfolio', {})
                if portfolio:
                    summary = []
                    for sym, pos in portfolio.items():
                        if abs(pos) > 0.00001:
                            summary.append(f"{sym}: {pos:+.4f}")
                    return " | ".join(summary) if summary else "No open positions"
        except:
            pass
    return "Unknown/None"

def get_arb_activity_summary():
    """Extract latest activity from Arb bot log."""
    line = get_latest_log_line(ARB_LOG_FILE, pattern=' - INFO - ')
    if line:
        # Keep it short
        parts = line.split(' - INFO - ')
        return parts[-1] if len(parts) > 1 else line
    return "No recent activity"

def get_log_slice(path, count=50):
    """Get the last N lines of a log file."""
    try:
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
                return "".join(lines[-count:])
    except Exception as e:
        logger.error(f"Read log failed {path}: {e}")
    return f"Error reading log: {path}"

# --- Status Formatters ---

def format_server_embed():
    embed = discord.Embed(title="🖥️ Server & Bot Status", color=0x2ECC71, timestamp=datetime.now(timezone.utc))
    
    # System Resources
    if psutil:
        cpu = psutil.cpu_percent()
        ram = psutil.virtual_memory()
        ram_str = f"{ram.used / (1024**3):.1f}GB / {ram.total / (1024**3):.1f}GB ({ram.percent}%)"
        embed.add_field(name="Resources", value=f"CPU: {cpu}% | RAM: {ram_str}", inline=False)
    
    # Bot Health
    ml_status, ml_time = get_file_health(ML_LOG_FILE)
    arb_status, arb_time = get_file_health(ARB_LOG_FILE)
    sup_status, sup_time = get_file_health(SUPERVISOR_LOG_FILE)
    
    embed.add_field(name="ML Bot", value=f"{ml_status} ({ml_time})", inline=True)
    embed.add_field(name="Arb Bot", value=f"{arb_status} ({arb_time})", inline=True)
    embed.add_field(name="Supervisor", value=f"{sup_status} ({sup_time})", inline=True)
    
    # Activity
    embed.add_field(name="ML Positions", value=f"`{get_ml_position_summary()}`", inline=False)
    embed.add_field(name="Arb Latest", value=f"`{get_arb_activity_summary()}`", inline=False)
    
    return embed

def format_market_embed():
    ctx_data = load_json_file(MARKET_CONTEXT_FILE)
    embed = discord.Embed(title="📊 Market Overview", color=0x3498DB, timestamp=datetime.now(timezone.utc))
    
    if ctx_data:
        score = ctx_data.get('sentiment_score', 0.0)
        emoji = "🔥" if score > 0.5 else ("❄️" if score < -0.5 else "⚖️")
        
        embed.add_field(
            name=f"{emoji} Sentiment / Bias / Risk",
            value=f"**{score:+.2f}** / **{ctx_data.get('bias', 'N/A')}** / **{ctx_data.get('risk_multiplier', 'N/A')}x**",
            inline=False
        )
        embed.add_field(name="Regime", value=f"`{ctx_data.get('regime', 'unknown')}`", inline=True)
        
        upd = ctx_data.get('timestamp', 'N/A')
        if 'T' in upd: upd = upd.split('T')[1][:8]
        embed.add_field(name="Last Update", value=upd, inline=True)
        
        reasoning = ctx_data.get('reasoning', '')
        if reasoning:
            embed.add_field(name="Reasoning", value=reasoning[:1000], inline=False)
    else:
        embed.description = "⚠️ Market context unavailable."
    
    return embed

# --- LLM Context Logic ---

async def query_llm_with_full_context(user_message: str) -> str:
    """Synchronous version with comprehensive context."""
    return await asyncio.to_thread(_query_llm_sync, user_message)

def _query_llm_sync(user_message: str) -> str:
    global llm_client
    if llm_client is None:
        llm_client = GeminiClient(model=LLM_MODEL)
    
    # 1. Market Context
    ctx = load_json_file(MARKET_CONTEXT_FILE)
    # 2. Latest Briefing
    briefing = load_text_file(BRIEFING_FILE, max_chars=1500)
    # 3. Server Status
    ml_pos = get_ml_position_summary()
    arb_act = get_arb_activity_summary()
    
    context_str = f"""
[Market Context]
Regime: {ctx.get('regime', 'N/A') if ctx else 'N/A'}
Sentiment Score: {ctx.get('sentiment_score', 0.0) if ctx else 0.0}
Bias: {ctx.get('bias', 'N/A') if ctx else 'N/A'}
Reasoning: {ctx.get('reasoning', 'N/A') if ctx else 'N/A'}

[Latest Briefing Snippet]
{briefing if briefing else 'No briefing available.'}

[Bot Status]
ML Positions: {ml_pos}
Arb Activity: {arb_act}
"""

    system_prompt = f"""You are the AI trading co-pilot (Gemma 3). 
You have access to the bot's live logs, market analysis, and positions.
Respond in the language the user uses. Be concise and professional.

Current context:
{context_str}

Use the context to answer trading-related questions. 
If the user asks about the server or bot performance, use the [Bot Status] info.
Limit responses to 300 words."""
    
    try:
        response = llm_client.query(user_message, system_prompt, temperature=0.7)
        return response if response else "⚠️ Empty response from LLM."
    except Exception as e:
        logger.error(f"LLM Error: {e}")
        return f"⚠️ LLM Error: {str(e)[:100]}"

# --- Bot Commands ---

@bot.event
async def on_ready():
    logger.info(f"✅ Bot logged in as {bot.user}")
    logger.info(f"   Target channel: #{TRADING_CHANNEL_NAME}")

@bot.event
async def on_message(message):
    if message.author == bot.user: return
    if message.channel.name != TRADING_CHANNEL_NAME: return
    
    # Security: Owner only
    if DISCORD_OWNER_ID and str(message.author.id) != str(DISCORD_OWNER_ID):
        return

    content = message.content.strip().lower()

    # --- Log Retrieval (Natural Language Intent) ---
    if ('日志' in content or 'log' in content) and any(x in content for x in ['ml', 'arb', 'supervisor', 'llm']):
        # Find count
        count_match = re.search(r'(\d+)', content)
        count = int(count_match.group(1)) if count_match else 50
        count = min(count, 500) # Safety limit
        
        target_file = None
        bot_name = ""
        if 'ml' in content: 
            target_file = ML_LOG_FILE
            bot_name = "ML_Bot"
        elif 'arb' in content: 
            target_file = ARB_LOG_FILE
            bot_name = "Arb_Bot"
        elif 'sup' in content or 'llm' in content: 
            target_file = SUPERVISOR_LOG_FILE
            bot_name = "Supervisor"
            
        if target_file and os.path.exists(target_file):
            log_data = get_log_slice(target_file, count)
            # Send as file to avoid 2000 char limit
            buf = io.BytesIO(log_data.encode('utf-8'))
            file_name = f"{bot_name}_logs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            await message.channel.send(
                content=f"📈 这是 **{bot_name}** 最近的 {count} 条日志：",
                file=discord.File(buf, filename=file_name)
            )
            return
        elif target_file:
            await message.channel.send(f"⚠️ 找不到日志文件: `{target_file}`")
            return

    # --- Commands ---
    if content in ['状态', 'status', '!status', 'market']:
        await message.channel.send(embed=format_market_embed())
        return

    if content in ['服务器', 'server', '!server']:
        await message.channel.send(embed=format_server_embed())
        return

    if content in ['简报', 'briefing', '!briefing']:
        briefing = load_text_file(BRIEFING_FILE, max_chars=1900)
        if briefing:
            await message.channel.send(f"```markdown\n{briefing}\n```")
        else:
            await message.channel.send("⚠️ No briefing found.")
        return

    if content in ['帮助', 'help', '!help']:
        help_text = """**🤖 Auto-Trader Assistant**
`状态` / `status` - 市场情绪与分析
`服务器` / `server` - Bot 运行状态与持仓
`简报` / `briefing` - 最新市场深度报告
`帮助` / `help` - 显示此信息

*直接发送消息可与 AI 进行深度对话 (已注入实时上下文)*"""
        await message.channel.send(help_text)
        return

    # --- LLM Query ---
    async with message.channel.typing():
        response = await query_llm_full(message.content)
        # Handle Discord's 2000 char limit
        for i in range(0, len(response), 1900):
            await message.channel.send(response[i:i+1900])

async def query_llm_full(msg):
    return await query_llm_with_full_context(msg)

def main():
    if not DISCORD_BOT_TOKEN:
        logger.critical("❌ DISCORD_BOT_TOKEN missing!")
        sys.exit(1)
    bot.run(DISCORD_BOT_TOKEN)

if __name__ == "__main__":
    main()
