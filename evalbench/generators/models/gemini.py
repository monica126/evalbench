import vertexai
import logging

from vertexai.generative_models._generative_models import (
    GenerativeModel,
    GenerationResponse,
)
from .generator import QueryGenerator


class GeminiGenerator(QueryGenerator):
    """Generator queries using Vertex model."""

    def __init__(self, core_db, querygenerator_config):
        super().__init__(core_db, querygenerator_config)
        self.name = "gemini"
        self.project_id = querygenerator_config["project_id"]
        self.location = querygenerator_config["location"]
        self.vertex_model = querygenerator_config["vertex_model"]
        self.base_prompt = querygenerator_config["base_prompt"]
        self.generation_config = None

        vertexai.init(project=self.project_id, location=self.location)
        self.model = GenerativeModel(self.vertex_model)
        self.base_prompt = self.base_prompt

    def generate(self, prompt):
        response = self.model.generate_content(
            self.base_prompt + prompt,
            generation_config=self.generation_config,
        )
        if isinstance(response, GenerationResponse):
            r = response.text
            r = r.replace("```sql", "") #required for gemini_1.0_pro, gemini_2.0_flash, gemini_2.0_pro
        return r
