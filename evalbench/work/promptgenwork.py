"""Work is the base class for all work items."""

from typing import Any
from work import Work


class SQLPromptGenWork(Work):
    """SQLPromptGenWork Generates SQL from the generator."""

    def __init__(self, generator: Any, eval_result: dict):
        self.generator = generator
        self.eval_result = eval_result

    def run(self, work_config: Any = None) -> dict:
        """Runs the work item.

        Args:
          work_config:

        Returns:

        """
        generated_prompt = None
        prompt_generator_error = None

        try:
            generated_prompt = self.generator.generate(self.eval_result["nl_prompt"])
        except Exception as e:
            prompt_generator_error = str(e)

        self.eval_result["generated_prompt"] = generated_prompt
        self.eval_result["prompt_generator_error"] = prompt_generator_error
        return self.eval_result
