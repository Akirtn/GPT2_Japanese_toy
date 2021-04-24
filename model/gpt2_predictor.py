from allennlp.predictors import Predictor

from pytorch_pretrained_bert.tokenization_gpt2 import GPT2Tokenizer
from pytorch_pretrained_bert.modeling_gpt2 import GPT2LMHeadModel

from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    LogitsProcessorList,
    MinLengthLogitsProcessor,
    TopKLogitsWarper,
    TemperatureLogitsWarper,
    BeamSearchScorer,
    )

from transformers import T5Tokenizer, AutoModelForCausalLM

import torch


MEDIUM_MODEL = "rinna/japanese-gpt2-medium"

class Gpt2Predictor(Predictor):
    """
    The HuggingFace implementation of GPT-2 is not an AllenNLP model;
    however, our demo only expects an AllenNLP ``Predictor``. Accordingly,
    we implement a ``Predictor`` that wraps the HuggingFace GPT-2 implementation.
    """
    def __init__(self,
                 model_name: str = MEDIUM_MODEL,
                 cache_size: int = 0) -> None:
        """
        Each cache element is about 8MB, so size accordingly.
        """
        # Cache stores tuples, so default value is a tuple
        # self.tokenizer = T5Tokenizer.from_pretrained(MEDIUM_MODEL)
        # self.model = AutoModelForCausalLM.from_pretrained(MEDIUM_MODEL)

        self.tokenizer = GPT2Tokenizer.from_pretrained(MEDIUM_MODEL)
        self.model = GPT2LMHeadModel.from_pretrained(MEDIUM_MODEL)

        # The end of text marker.
        #self.END_OF_TEXT = self.tokenizer.encoder["<|endoftext|>"]

    def simple_generate_json(self, inputs: dict) -> dict:
        # 推論
        input = self.tokenizer.encode(inputs["previous"], return_tensors="pt")
        output = self.model.generate(input, do_sample=True, max_length=30, num_return_sequences=3,output_scores=True)
        words = self.tokenizer.batch_decode(output)
        return {
            "logits":None,
            "probabilities":None ,
            "input":input,
            "words": words,
            "output": output
        }

    def generate_json(self, inputs: dict) -> dict:
        previous_str = inputs["previous"]
        next_str = inputs.get("next")
        topk = inputs.get("topk", 10)

        logits = self._predict(previous_str, next_str)
        probabilities = torch.nn.functional.softmax(logits)

        best_logits, best_indices = logits.topk(topk)
        best_words = [self.tokenizer.decode([idx.item()])
                      for idx in best_indices]
        best_probabilities = probabilities[best_indices].tolist()

        return {
            "logits": best_logits.tolist(),
            "probabilities": best_probabilities,
            "words": best_words,
            "output": previous_str + (next_str or "")
        }

    def _predict(self, previous: str, next: str = None) -> torch.Tensor:

        past_logits, past = (None, None)

        # CASE 1: Previously seen input, no next
        if next is None and past is not None:
            return past_logits

        # CASE 2: Previously seen input, yes next
        elif past is not None:
            token_ids = self.tokenizer.encode(next)
        # CASE 3: Brand new input, no next
        elif next is None:
            token_ids = self.tokenizer.encode(previous)
        # CASE 4: Brand new input, yes next
        else:
            token_ids = self.tokenizer.encode(previous) + self.tokenizer.encode(next)

        inputs = torch.LongTensor([token_ids])

        logits, present = self.model.generate(inputs)
        logits = logits[0, -1]

        return logits

    def __getitem__(self, index: int) -> str:
        return self.tokenizer.decode([index])