"""LLM integration: sends the classifier's findings to Gemini and returns
a plain-English explanation. Prompts live in prompts/prompts.yaml."""
import os

import yaml
from google import genai
from google.genai import types


def load_prompts(path="prompts/prompts.yaml"):
    """Read all prompt templates from the YAML prompt file."""
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


class Explainer:
    """Wraps the Gemini API call for the 'explainer' prompt."""

    def __init__(self, cfg, prompts):
        self.model = cfg["llm"]["model"]
        self.retries = cfg["llm"]["max_retries"]
        self.prompt = prompts["explainer"]
        api_key = os.environ.get(cfg["llm"]["api_key_env"])
        self.client = genai.Client(api_key=api_key) if api_key else None

    def explain(self, **facts):
        """Fill the prompt template with the analysis facts and ask Gemini.
        Returns the explanation text, or None if the LLM is unavailable."""
        if self.client is None:
            return None
        config = types.GenerateContentConfig(
            system_instruction=self.prompt["system"],
            temperature=self.prompt["temperature"],
            max_output_tokens=self.prompt["max_output_tokens"],
        )
        user_msg = self.prompt["user_template"].format(**facts)
        for _ in range(self.retries):
            try:
                response = self.client.models.generate_content(
                    model=self.model, contents=user_msg, config=config)
                if response.text:
                    return response.text.strip()
            except Exception as err:            # network/quota errors: retry, then give up
                print(f"[LLM] call failed: {err}")
        return None
