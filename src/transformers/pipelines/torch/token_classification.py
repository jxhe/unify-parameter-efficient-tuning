from transformers import BatchEncoding

from transformers.modeling_outputs import TokenClassifierOutput
from transformers.pipelines import MaybeBatch, TokenClassificationConfig, TokenClassificationOutput
from transformers.pipelines.outputs import Entity
from transformers.pipelines.torch import PreTrainedPipeline


# Token Classification

TokenClassificationInput = str


class TokenClassificationPipeline(
    PreTrainedPipeline[
        TokenClassificationConfig, TokenClassificationInput, TokenClassifierOutput, TokenClassificationOutput
    ]
):
    task = "token-classification"
    default_config = TokenClassificationConfig()

    def postprocess(self, encodings: BatchEncoding, model_output: TokenClassifierOutput, config: TokenClassificationConfig) -> MaybeBatch[TokenClassificationOutput]:
        # Retrieve labels
        labels = self._model.config.id2label

        # Convert to probabilities
        probs = model_output.logits.softmax(dim=-1)

        # Retrieve the argmax for each token over sequence axis
        batch_entities_labels = probs.argmax(dim=-1)

        batch_entities = []
        for batch_idx, entities_labels in enumerate(batch_entities_labels):
            entities = []
            for token_idx, label_idx in enumerate(entities_labels):
                label_idx = label_idx.item()
                if label_idx not in config.ignore_labels:
                    entities.append(Entity(
                        token=self._tokenizer.convert_ids_to_tokens([encodings["input_ids"][batch_idx][token_idx]])[0],
                        index=token_idx,
                        label=labels[label_idx],
                        score=probs[batch_idx, token_idx, label_idx].item(),
                    ))
            batch_entities.append(entities)

        return batch_entities


