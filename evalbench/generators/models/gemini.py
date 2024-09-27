import vertexai
import logging

from vertexai.generative_models._generative_models import (
    GenerativeModel,
    GenerationResponse,
)
from .generator import QueryGenerator


class GeminiGenerator(QueryGenerator):
    """Generator queries using Vertex model."""

    def __init__(self, querygenerator_config):
        super().__init__(querygenerator_config)
        self.project_id = querygenerator_config["project_id"]
        self.location = querygenerator_config["location"]
        self.vertex_model = querygenerator_config["vertex_model"]
        self.base_prompt = querygenerator_config["base_prompt"]
        self.generation_config = None
        self.name = "gemini"

        vertexai.init(project=self.project_id, location=self.location)
        self.model = GenerativeModel(self.vertex_model)
        self.base_prompt = self.base_prompt

    def generate(self, human_language_question):
        response = self.model.generate_content(
            self.base_prompt + human_language_question,
            generation_config=self.generation_config,
        )
        if isinstance(response, GenerationResponse):
            r = response.text
            r = r.replace("```sql", "")
        return r
