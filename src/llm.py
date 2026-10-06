"""LLM integration: sends the classifier's findings to an LLM (Groq API) and
returns a plain-English explanation. Prompts live in prompts/prompts.yaml."""
import os
import time

import yaml
from groq import Groq


def load_prompts(path="prompts/prompts.yaml"):
    """Read all prompt templates from the YAML prompt file."""
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


class Explainer:
    """Wraps the LLM API call for the 'explainer' prompt."""

    def __init__(self, cfg, prompts):
        llm = cfg["llm"]
        self.model = llm["model"]
        self.retries = llm["max_retries"]
        self.prompt = prompts["explainer"]
        api_key = os.environ.get(llm["api_key_env"])
        # timeout: give up on a slow answer instead of freezing the app
        self.client = (Groq(api_key=api_key, timeout=llm["timeout_seconds"], max_retries=0)
                       if api_key else None)

    def explain(self, **facts):
        """Fill the prompt template with the analysis facts and ask the LLM.
        Returns the explanation text, or None if the LLM is unavailable."""
        if self.client is None:
            return None
        messages = [{"role": "system", "content": self.prompt["system"]},
                    {"role": "user", "content": self.prompt["user_template"].format(**facts)}]
        for attempt in range(self.retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model, messages=messages,
                    temperature=self.prompt["temperature"],
                    max_tokens=self.prompt["max_output_tokens"])
                text = response.choices[0].message.content
                if text:
                    return text.strip()
            except Exception as err:            # network/quota/timeout errors: wait, retry, then give up
                print(f"[LLM] call failed: {err}")
                time.sleep(2 ** attempt)        # wait 1s, 2s, 4s between tries
        return None
