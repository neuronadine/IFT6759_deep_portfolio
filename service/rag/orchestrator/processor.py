from __future__ import annotations
from typing import TYPE_CHECKING, Any

# from web_scraper.

from rag.retrieval.factory import ChunkerFactory
from rag.config.python_schema import Config
from rag.feature_extractors.factory import FeatureExtractorFactory
from rag.models.factory import ResponseSchemaFactory
from rag.engine.exceptions import ProcessingError
from rag.engine.factory_handler import FactoryHandler
from rag.llm.factory import LLMExecuterFactory
from rag.message import create_message_from_prompt
from rag.output_parsers.factory import OutputParserFactory
from rag.part_data import PartData
from rag.prompt.factory import PromptBuilderFactory
from rag.schema_translators.factory import SchemaTranslatorFactory

if TYPE_CHECKING:
    from rag.message import Message

class FileProcessor:
    def __init__(self, config: Config):
        self.config = config

        # initialize handlers
        self.factory_handler = FactoryHandler({
            "feature_extractor": FeatureExtractorFactory(),
            "chunker": ChunkerFactory(),
            "llm_executer": LLMExecuterFactory(),
            "prompt_builder": PromptBuilderFactory(),
            "schema_translator": SchemaTranslatorFactory(),
            "response_schema": ResponseSchemaFactory(),
            "output_parser": OutputParserFactory()
        })
        self._configure()

    def _configure(self):
        self._document_model = self.factory_handler.get_document_model(self.config)
        self._llm = self.factory_handler.get_llm_processor(self.config)
        self._prompt_builder= self.factory_handler.get_prompt_builder(self.config)
        self._output_parser = self.factory_handler.get_output_parser(self.config)
        self.feature_extractor = self.factory_handler.get_feature_extrator(self.config)


    def _execute_feature_extractor(self, files: PartData | list[PartData]) -> dict[str, Any]:
        """Process files and extract embedded contents"""
        if not isinstance(files, list | PartData):
            raise TypeError(type(files))
        if isinstance(files, PartData):
            files = [files]

        try:
            processed_content = self._execute_feature_extractor(files=files)
            prompt = self._execute_prompt_builder()
            contents = [create_message_from_prompt(prompt, parts=processed_content)]
            result = self._execute_llm(contents=contents)
            return self._execute_output_parser(result)
        
        except Exception as e:
            raise ProcessingError(f"Error processing file: {e!s}") from e