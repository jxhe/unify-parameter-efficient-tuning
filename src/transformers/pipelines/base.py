from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, ClassVar, Dict, Generic, Mapping, Optional, Sized, TypeVar, Union

from transformers import BatchEncoding, PreTrainedTokenizer


# Syntactic sugar to indicate it can take multiple inputs of type T at once.
T = TypeVar("T")
MaybeBatch = Union[T, Sized[T]]


@dataclass
class PipelineConfig:
    tokenizer_kwargs: Optional[Dict[str, Any]] = None
    model_kwargs: Optional[Dict[str, Any]] = None


# Define all the generic types for a Pipeline
InputType = TypeVar("InputType")  # Pipeline input type
OutputType = TypeVar("OutputType")  # Pipeline output type
IntermediateType = TypeVar("IntermediateType")  # Model output type (i.e. after forward())
ConfigType = TypeVar("ConfigType", bound=PipelineConfig)  # Pipeline configuration type
ModelType = TypeVar("ModelType")  # Pipeline model type


class Pipeline(ABC, Generic[ConfigType, ModelType, InputType, IntermediateType, OutputType]):
    """"""

    __slots__ = ["task", "default_config", "_tokenizer", "_model", "_configs"]

    task: ClassVar[str]
    default_config: ClassVar[ConfigType]

    def __init__(self, tokenizer: PreTrainedTokenizer, model: ModelType):
        self._tokenizer = tokenizer
        self._model = model

        self._configs: Dict[str, ConfigType] = dict()
        self._configs["default"] = self.default_config

    @property
    def tokenizer(self) -> PreTrainedTokenizer:
        """
        Tokenizer associated with this pipeline
        Returns:
            PreTrainedTokenizer instance
        """
        return self._tokenizer

    @property
    def model(self) -> ModelType:
        """
        Model associated with this pipeline
        Returns:
            Pretrained model instance
        """
        raise self._model

    @property
    def configs(self) -> Mapping[str, ConfigType]:
        """
        Return all registered configuration's mapping on the pipeline
        Returns:
            Mapping[str, ConfigType]

        """
        return self._configs

    @property
    def default_config(self) -> ConfigType:
        return self.default_config

    def get_config(self, name: str) -> Optional[ConfigType]:
        """
        Attempt to retrieve a configuration from its registered name.
        If no configuration matches the requested name, None is returned.
        Args:
            name (:obj:`str`): Name of the configuration to look for

        Returns:
            Instance of ConfigType if the configuration associated with the requested name is present in the registry
            None if the requested named cannot be found in the registry
        """
        return self._configs.get(name, None)

    def register_config(self, name: str, config: ConfigType) -> ConfigType:
        """
        Register a configuration with the specified name uniquely identifying a configuration.
        Args:
            name (:obj:`str`): Name used to reference the configuration
            config (:obj:`ConfigType`): The configuration instance to associate to the specified name
        Returns:
            The provided ConfigType instance
        """
        self._configs[name] = config
        return config

    def delete_config(self, name: str) -> Optional[ConfigType]:
        """
        Attempt to delete a configuration if registered on the pipeline.
        The ConfigType instance associated with the provided name is returned if present on the pipeline,
        None is returned otherwise.
        Args:
            name (:obj:`str`): Name identifying the configuration to delete

        Returns:
            ConfigType instance if the name identifier was found on the pipeline.
            None if the name identifier is unknown on the pipeline.
        """
        return self._configs.pop(name, None)

    def __call__(
        self, inputs: MaybeBatch[InputType], config: Optional[Union[str, ConfigType]], **kwargs
    ) -> MaybeBatch[IntermediateType]:
        """

        Args:
            inputs (:obj:`InputType`):
            config (:obj:`ConfigType`)
            **kwargs:

        Returns:
        """
        # Retrieve the appropriate config if identifier is provided or None
        if isinstance(config, str):
            config = self.get_config(config)
        elif not config:
            config = self.default_config

        # Preprocess the input
        encodings = self.preprocess(inputs, config)

        # Forward the encoding through the model
        model_output = self.forward(encodings, config)

        # Apply any postprocessing steps required
        return self.postprocess(model_output, config)

    @abstractmethod
    def preprocess(self, inputs: MaybeBatch[InputType], config: ConfigType) -> BatchEncoding:
        """
        Process the raw inputs to generate a BatchEncoding representation through the use of a tokenizer.
        Args:
            inputs (:obj:`InputType`):
            config (:obj:`ConfigType`)

        Returns:
        """
        # TODO: preprocess while calling batch_encode_plus might support truncation and thus all the remaining
        #       steps [forward(), postprocess()] should supports inference and input reconstruction.
        raise NotImplementedError()

    @abstractmethod
    def forward(self, encodings: BatchEncoding, config: ConfigType) -> IntermediateType:
        """

        Args:
            encodings (:obj:`BatchEncoding`):
            config (:obj:`ConfigType`)
        Returns:
        """
        raise NotImplementedError()

    @abstractmethod
    def postprocess(self, model_output: IntermediateType, config: ConfigType) -> MaybeBatch[OutputType]:
        """

        Args:
            model_output (:obj:`IntermediateType`):
            config (:obj:`ConfigType`)

        Returns:
        """
        raise NotImplementedError()
