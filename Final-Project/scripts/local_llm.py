"""
Local LLM Module - Private Network AI via Ollama
=================================================
Connects to a locally running Ollama server to chat with open-source LLMs
that run entirely on your machine / private network.

No data is sent to external APIs. Everything stays local.

Setup (one-time):
    1. Install Ollama: https://ollama.com
    2. Pull a model:
       ollama pull llama3.2          # General purpose (recommended)
       ollama pull dolphin-mistral   # Uncensored/unrestricted version
       ollama pull mistral           # Fast and capable
       ollama pull qwen2.5           # Multilingual support
    3. Ollama runs as a background service on http://localhost:11434

How it works:
    Ollama runs a REST API server locally. This module sends HTTP requests
    to that server - no internet required after model download.
"""

import json
import requests
from datetime import datetime


SYSTEM_PROMPT = """You are FinanceAI, an expert financial analyst assistant running
completely locally on a private network. You have deep knowledge of:
- Stock market analysis and technical indicators
- Financial document interpretation
- Investment strategies and risk assessment
- Machine learning models for financial prediction
- Portfolio management

You provide direct, insightful financial analysis. You are not limited by
external API restrictions - you run fully locally and privately.
Always base your analysis on the data provided in the context."""


class LocalLLM:
    """
    Interface to a locally running Ollama LLM server.

    Supports streaming output (text appears word by word) and
    context injection for RAG (Retrieval Augmented Generation) -
    feeding your extracted data to the model.
    """

    # Default: try these models in order if none specified
    RECOMMENDED_MODELS = [
        "llama3.2",         # Meta's LLaMA 3.2 - excellent general purpose
        "dolphin-mistral",  # Uncensored fine-tune of Mistral
        "mistral",          # Mistral 7B - fast and capable
        "llama3",           # LLaMA 3 8B
        "qwen2.5",          # Alibaba Qwen 2.5
    ]

    def __init__(self, model=None, host="http://localhost:11434"):
        """
        Args:
            model (str): Ollama model name. If None, auto-detects available model.
            host (str): Ollama server URL. Default: http://localhost:11434
        """
        self.host = host.rstrip('/')
        self.model = model
        self.conversation_history = []
        self._available = None

    def is_available(self):
        """
        Check if Ollama server is running.

        Returns:
            bool: True if server responds
        """
        try:
            response = requests.get(f"{self.host}/api/tags", timeout=3)
            self._available = response.status_code == 200
        except (requests.ConnectionError, requests.Timeout):
            self._available = False
        return self._available

    def list_models(self):
        """
        List all models available on the local Ollama server.

        Returns:
            list: Model names installed locally
        """
        try:
            response = requests.get(f"{self.host}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get('models', [])
                return [m['name'] for m in models]
        except Exception:
            pass
        return []

    def auto_select_model(self):
        """
        Automatically pick the best available model from recommendations.

        Returns:
            str: Model name, or None if nothing available
        """
        available = self.list_models()
        if not available:
            return None

        # Try recommended models first
        for recommended in self.RECOMMENDED_MODELS:
            for installed in available:
                if recommended in installed:
                    return installed

        # Fall back to first available
        return available[0]

    def setup(self):
        """
        Initialize the LLM: check server, select model.

        Returns:
            bool: True if ready to use
        """
        if not self.is_available():
            print("\n[LLM] Ollama server not running!")
            print("      Install from: https://ollama.com")
            print("      Then run: ollama serve")
            print("      Then pull a model: ollama pull llama3.2")
            return False

        if self.model is None:
            self.model = self.auto_select_model()

        if self.model is None:
            print("\n[LLM] No models found! Pull one first:")
            print("      ollama pull llama3.2")
            print("      ollama pull dolphin-mistral   (uncensored)")
            return False

        print(f"[LLM] Ready - using model: {self.model}")
        return True

    def chat(self, user_message, context="", stream=True):
        """
        Send a message and get a response.

        Args:
            user_message (str): Your question or prompt
            context (str): Financial data/analysis to include (RAG)
            stream (bool): Stream output word-by-word (True) or wait for full response

        Returns:
            str: The model's response
        """
        if not self.model:
            if not self.setup():
                return "LLM not available."

        # Build message with optional financial context (RAG pattern)
        if context:
            full_message = (
                f"=== FINANCIAL DATA CONTEXT ===\n{context}\n"
                f"=== USER QUESTION ===\n{user_message}"
            )
        else:
            full_message = user_message

        # Add to conversation history (maintains multi-turn context)
        self.conversation_history.append({
            "role": "user",
            "content": full_message
        })

        messages = [{"role": "system", "content": SYSTEM_PROMPT}] + self.conversation_history

        payload = {
            "model": self.model,
            "messages": messages,
            "stream": stream,
            "options": {
                "temperature": 0.7,     # Creativity (0=deterministic, 1=creative)
                "top_p": 0.9,           # Nucleus sampling
                "num_predict": 1024,    # Max tokens to generate
            }
        }

        try:
            if stream:
                return self._stream_response(payload)
            else:
                return self._blocking_response(payload)
        except requests.ConnectionError:
            return "Connection lost to Ollama server."
        except Exception as e:
            return f"Error: {e}"

    def _stream_response(self, payload):
        """
        Stream the response token by token (prints as generated).

        Args:
            payload (dict): Request payload

        Returns:
            str: Complete response text
        """
        response = requests.post(
            f"{self.host}/api/chat",
            json=payload,
            stream=True,
            timeout=120
        )

        full_response = ""
        print(f"\n[{self.model}] ", end='', flush=True)

        for line in response.iter_lines():
            if line:
                try:
                    data = json.loads(line)
                    if not data.get('done', False):
                        chunk = data.get('message', {}).get('content', '')
                        print(chunk, end='', flush=True)
                        full_response += chunk
                except json.JSONDecodeError:
                    continue

        print()  # newline after streaming

        # Save to conversation history
        self.conversation_history.append({
            "role": "assistant",
            "content": full_response
        })

        return full_response

    def _blocking_response(self, payload):
        """
        Wait for complete response (no streaming).

        Returns:
            str: Complete response text
        """
        response = requests.post(
            f"{self.host}/api/chat",
            json=payload,
            timeout=120
        )
        result = response.json()
        content = result.get('message', {}).get('content', '')

        self.conversation_history.append({
            "role": "assistant",
            "content": content
        })

        return content

    def ask_about_stock(self, ticker, ml_summary, ocr_text=""):
        """
        Ask the LLM to analyze specific stock data.

        Args:
            ticker (str): Stock ticker
            ml_summary (str): ML prediction summary text
            ocr_text (str): Text extracted from document image

        Returns:
            str: LLM analysis
        """
        context = f"Stock Ticker: {ticker}\n\n{ml_summary}"
        if ocr_text:
            context += f"\n\nExtracted from document image:\n{ocr_text}"

        prompt = (
            f"Based on the ML prediction data and any document content provided, "
            f"give me a comprehensive analysis of {ticker}. Include:\n"
            f"1. What the prediction suggests about short-term movement\n"
            f"2. Key technical signals to watch\n"
            f"3. Risk factors\n"
            f"4. Your overall assessment"
        )

        return self.chat(prompt, context=context)

    def interactive_chat(self, context=""):
        """
        Start an interactive chat session with the LLM.

        Args:
            context (str): Financial context to make available to the model
        """
        print("\n" + "=" * 60)
        print(f"LOCAL LLM CHAT - {self.model}")
        print("Running on your private network | No data sent externally")
        print("Type 'quit' or 'exit' to end | 'clear' to reset history")
        print("=" * 60)

        if context:
            print("\n[Context loaded: financial data & ML results available]")

        while True:
            try:
                user_input = input("\nYou: ").strip()
            except (KeyboardInterrupt, EOFError):
                print("\n[Chat ended]")
                break

            if not user_input:
                continue

            if user_input.lower() in ('quit', 'exit', 'q'):
                print("[Chat ended]")
                break

            if user_input.lower() == 'clear':
                self.conversation_history = []
                print("[Conversation history cleared]")
                continue

            if user_input.lower() == 'history':
                print(f"[{len(self.conversation_history)} messages in history]")
                continue

            # First message uses context; subsequent messages maintain conversation
            use_context = context if not self.conversation_history else ""
            self.chat(user_input, context=use_context)

    def clear_history(self):
        """Reset the conversation history."""
        self.conversation_history = []
        print("[LLM] Conversation history cleared.")

    def get_status(self):
        """
        Get a formatted status string.

        Returns:
            str: Status info
        """
        if self.is_available():
            models = self.list_models()
            return (
                f"Ollama Server: ONLINE ({self.host})\n"
                f"Active Model:  {self.model or 'None selected'}\n"
                f"Models Available: {len(models)}\n"
                f"Installed: {', '.join(models[:5]) if models else 'None'}"
            )
        else:
            return (
                f"Ollama Server: OFFLINE\n"
                f"To start: run 'ollama serve' in terminal\n"
                f"To install: https://ollama.com"
            )
