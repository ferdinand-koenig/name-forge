# domain_generator.py
import ast
import json
from typing import List, Optional

import yaml

from .lib.local_llm_wrapper import LocalTransformersLLM


class DomainGenerator:
    def __init__(
        self,
        llm,
        prompt_template: Optional[str] = None,
    ):
        """
        Args:
            llm: An LLM instance (e.g., LocalTransformersLLM)
            prompt_template (str): Optional prompt template; if None, loads from YAML
        """
        self.llm = llm

        if prompt_template is not None:
            self.prompt_template = prompt_template
        else:
            # Load default prompt from YAML
            with open("prompts/prompt-1.yaml", "r") as f:
                yaml_content = yaml.safe_load(f)
            self.prompt_template = yaml_content["domain_prompt"]

    def generate(self, description: str) -> List[str]:
        """
        Generate 2–3 domain names based on a business description.

        Args:
            description (str): Business/project description

        Returns:
            List[str]: Generated domain names or ["__BLOCKED__"] if unsafe
        """
        prompt = self.prompt_template.format(description=description)
        output = self.llm(prompt)

        return self._parse_output(output)

    def _parse_output(self, output: str) -> List[str]:
        try:
            start = output.find("[")
            end = output.rfind("]") + 1
            snippet = output[start:end]
            # Try parsing as JSON
            try:
                domains = json.loads(snippet)
            except json.JSONDecodeError:
                # Fall back to Python literal parsing
                domains = ast.literal_eval(snippet)

            if isinstance(domains, list):
                if "__BLOCKED__" in domains:
                    domains = ["__BLOCKED__"] * len(domains)
                return domains
            else:
                print("⚠️ Output is not a list. Raw output:", output)
                return []
        except Exception as e:
            print(e)
            print("⚠️ Failed to parse JSON. Raw output:", output)
            return []


# -----------------------------
# Example usage
# -----------------------------
if __name__ == "__main__":
    # Example: create LLM instance (replace with your GGUF path)
    MODEL_PATH = "path/to/mistral.gguf"
    llm = LocalTransformersLLM(
        model_name=MODEL_PATH,
        max_length=50,
        do_sample=True,
        temperature=0.7,
    )

    # Instantiate the generator
    generator = DomainGenerator(llm=llm)

    # Generate domains
    description = "A sustainable eco-friendly packaging company"
    domains = generator.generate(description)
    print("Generated domains:", domains)
