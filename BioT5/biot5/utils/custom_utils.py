from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from transformers import BatchEncoding
from dataclasses import dataclass
from transformers import AutoTokenizer, TensorType
import torch
import random
import string

@dataclass
class DataCollatorForUnimptT5:
    """
    [Copied from https://github.com/huggingface/transformers/blob/main/examples/flax/language-modeling/run_t5_mlm_flax.py]
    Data collator used for T5 span-masked language modeling.
    It is made sure that after masking the inputs are of length `data_args.max_seq_length` and targets are also of fixed length.
    For more information on how T5 span-masked language modeling works, one can take a look
    at the `official paper <https://arxiv.org/pdf/1910.10683.pdf>`__
    or the `official code for preprocessing <https://github.com/google-research/text-to-text-transfer-transformer/blob/master/t5/data/preprocessors.py>`__ .
    Args:
        tokenizer (:class:`~transformers.PreTrainedTokenizer` or :class:`~transformers.PreTrainedTokenizerFast`):
            The tokenizer used for encoding the data.
        noise_density (:obj:`float`):
            The probability with which to (randomly) mask tokens in the input.
        mean_noise_span_length (:obj:`float`):
            The average span length of the masked tokens.
        input_length (:obj:`int`):
            The expected input length after masking.
        target_length (:obj:`int`):
            The expected target length after masking.
        pad_token_id: (:obj:`int`):
            The pad token id of the model
        decoder_start_token_id: (:obj:`int):
            The decoder start token id of the model
    """

    tokenizer: AutoTokenizer
    noise_density: float
    mean_noise_span_length: float
    input_length: int
    target_length: int
    pad_token_id: int

    def __call__(self, examples: List[Dict[str, np.ndarray]]) -> BatchEncoding:
        # convert list to dict and tensorize input
        # flatten list of dicts in tuple to list of dicts
        if isinstance(examples[0], tuple):
            examples = [sample_dict for sample_tuple in examples for sample_dict in sample_tuple]

        examples_mlm = []
        examples_src_tgt = []
        for example_i in examples:
            if "labels" in example_i:
                examples_src_tgt.append(example_i)
            else:
                examples_mlm.append(example_i)
        
        batch = BatchEncoding(
            {
                k: np.array([examples_mlm[i][k] for i in range(len(examples_mlm))])
                for k, v in examples_mlm[0].items()
            }
        )

        input_ids = batch["input_ids"]
        batch_size, expandend_input_length = input_ids.shape

        mask_indices = np.asarray(
            [
                self.random_spans_noise_mask(expandend_input_length)
                for i in range(batch_size)
            ]
        )
        labels_mask = ~mask_indices

        input_ids_sentinel = self.create_sentinel_ids(mask_indices.astype(np.int8))
        labels_sentinel = self.create_sentinel_ids(labels_mask.astype(np.int8))

        batch["input_ids"] = self.filter_input_ids(input_ids, input_ids_sentinel)
        batch["labels"] = self.filter_input_ids(input_ids, labels_sentinel)

        if batch["input_ids"].shape[-1] != self.input_length:
            raise ValueError(
                f"`input_ids` are incorrectly preprocessed. `input_ids` length is {batch['input_ids'].shape[-1]}, but"
                f" should be {self.input_length}."
            )

        if batch["labels"].shape[-1] != self.target_length:
            raise ValueError(
                f"`labels` are incorrectly preprocessed. `labels` length is {batch['labels'].shape[-1]}, but should be"
                f" {self.target_length}."
            )

        batch_src_tgt = BatchEncoding(
            {
                k: np.array([examples_src_tgt[i][k] for i in range(len(examples_src_tgt))])
                for k, v in examples_src_tgt[0].items()
            }
        )
        batch_src_tgt['labels'][batch_src_tgt['labels'] == self.pad_token_id] = -100
        # pad batch['labels'] to the same the batch['input_ids']
        batch['labels'] = np.concatenate((batch['labels'], np.full((batch_size, self.input_length - self.target_length), -100)), axis=1)

        batch = {k: np.concatenate((batch[k], batch_src_tgt[k]), axis=0) for k in batch}

        batch = {k: torch.from_numpy(v) for k, v in batch.items()}
        
        return batch

    def create_sentinel_ids(self, mask_indices):
        """
        Sentinel ids creation given the indices that should be masked.
        The start indices of each mask are replaced by the sentinel ids in increasing
        order. Consecutive mask indices to be deleted are replaced with `-1`.
        """
        start_indices = mask_indices - np.roll(mask_indices, 1, axis=-1) * mask_indices
        start_indices[:, 0] = mask_indices[:, 0]

        sentinel_ids = np.where(
            start_indices != 0, np.cumsum(start_indices, axis=-1), start_indices
        )
        # sentinel_ids = np.where(
        #     sentinel_ids != 0, (len(self.tokenizer) - sentinel_ids), 0
        # )
        # For additional molecule and protein tokens
        sentinel_ids = np.where(
            sentinel_ids != 0, (self.tokenizer.vocab_size - sentinel_ids), 0
        )
        sentinel_ids -= mask_indices - start_indices

        return sentinel_ids

    def filter_input_ids(self, input_ids, sentinel_ids):
        """
        Puts sentinel mask on `input_ids` and fuse consecutive mask tokens into a single mask token by deleting.
        This will reduce the sequence length from `expanded_inputs_length` to `input_length`.
        """
        batch_size = input_ids.shape[0]

        input_ids_full = np.where(sentinel_ids != 0, sentinel_ids, input_ids)
        # input_ids tokens and sentinel tokens are >= 0, tokens < 0 are
        # masked tokens coming after sentinel tokens and should be removed
        input_ids = input_ids_full[input_ids_full >= 0].reshape((batch_size, -1))
        input_ids = np.concatenate(
            [
                input_ids,
                np.full((batch_size, 1), self.tokenizer.eos_token_id, dtype=np.int32),
            ],
            axis=-1,
        )
        return input_ids

    def random_spans_noise_mask(self, length):
        """This function is copy of `random_spans_helper <https://github.com/google-research/text-to-text-transfer-transformer/blob/84f8bcc14b5f2c03de51bd3587609ba8f6bbd1cd/t5/data/preprocessors.py#L2682>`__ .

        Noise mask consisting of random spans of noise tokens.
        The number of noise tokens and the number of noise spans and non-noise spans
        are determined deterministically as follows:
        num_noise_tokens = round(length * noise_density)
        num_nonnoise_spans = num_noise_spans = round(num_noise_tokens / mean_noise_span_length)
        Spans alternate between non-noise and noise, beginning with non-noise.
        Subject to the above restrictions, all masks are equally likely.

        Args:
            length: an int32 scalar (length of the incoming token sequence)
            noise_density: a float - approximate density of output mask
            mean_noise_span_length: a number

        Returns:
            a boolean tensor with shape [length]
        """

        orig_length = length

        num_noise_tokens = int(np.round(length * self.noise_density))
        # avoid degeneracy by ensuring positive numbers of noise and nonnoise tokens.
        num_noise_tokens = min(max(num_noise_tokens, 1), length - 1)
        num_noise_spans = int(np.round(num_noise_tokens / self.mean_noise_span_length))

        # avoid degeneracy by ensuring positive number of noise spans
        num_noise_spans = max(num_noise_spans, 1)
        num_nonnoise_tokens = length - num_noise_tokens

        # pick the lengths of the noise spans and the non-noise spans
        def _random_segmentation(num_items, num_segments):
            """Partition a sequence of items randomly into non-empty segments.
            Args:
                num_items: an integer scalar > 0
                num_segments: an integer scalar in [1, num_items]
            Returns:
                a Tensor with shape [num_segments] containing positive integers that add
                up to num_items
            """
            mask_indices = np.arange(num_items - 1) < (num_segments - 1)
            np.random.shuffle(mask_indices)
            first_in_segment = np.pad(mask_indices, [[1, 0]])
            segment_id = np.cumsum(first_in_segment)
            # count length of sub segments assuming that list is sorted
            _, segment_length = np.unique(segment_id, return_counts=True)
            return segment_length

        noise_span_lengths = _random_segmentation(num_noise_tokens, num_noise_spans)
        nonnoise_span_lengths = _random_segmentation(
            num_nonnoise_tokens, num_noise_spans
        )

        interleaved_span_lengths = np.reshape(
            np.stack([nonnoise_span_lengths, noise_span_lengths], axis=1),
            [num_noise_spans * 2],
        )
        span_starts = np.cumsum(interleaved_span_lengths)[:-1]
        span_start_indicator = np.zeros((length,), dtype=np.int8)
        span_start_indicator[span_starts] = True
        span_num = np.cumsum(span_start_indicator)
        is_noise = np.equal(span_num % 2, 1)

        return is_noise[:orig_length]

def tokenize_function_seq_desc(examples, tokenizer, max_length):
    tokenizer_seq_out = tokenizer(
        text=examples["seq"],
        return_attention_mask=False,
        return_tensors="np",
        padding="max_length",
        truncation=True,
        max_length=max_length,
    )

    tokenizer_desc_out = tokenizer(
        text=examples["desc"],
        return_attention_mask=False,
        return_tensors="np",
        padding="max_length",
        truncation=True,
        max_length=max_length,
    )

    if random.random() < 0.5:
        input_ids = tokenizer_seq_out["input_ids"]
        label_ids = tokenizer_desc_out["input_ids"]
    else:
        input_ids = tokenizer_desc_out["input_ids"]
        label_ids = tokenizer_seq_out["input_ids"]

    result = {"input_ids": input_ids, "labels": label_ids}

    return result


@dataclass
class DataCollatorForMLM:
    """
    Data collator used for language modeling. Inputs are dynamically padded to the maximum length of a batch if they
    are not all of the same length.

    Args:
        tokenizer (:class:`~transformers.PreTrainedTokenizer` or :class:`~transformers.PreTrainedTokenizerFast`):
            The tokenizer used for encoding the data.
        mlm_probability (:obj:`float`, `optional`, defaults to 0.15):
            The probability with which to (randomly) mask tokens in the input.

    .. note::

        For best performance, this data collator should be used with a dataset having items that are dictionaries or
        BatchEncoding, with the :obj:`"special_tokens_mask"` key, as returned by a
        :class:`~transformers.PreTrainedTokenizer` or a :class:`~transformers.PreTrainedTokenizerFast` with the
        argument :obj:`return_special_tokens_mask=True`.
    """

    tokenizer: AutoTokenizer
    mlm_probability: float = 0.15

    def __post_init__(self):
        if self.tokenizer.mask_token is None:
            raise ValueError(
                "This tokenizer does not have a mask token which is necessary for masked language modeling. "
                "You should pass `mlm=False` to train on causal language modeling instead."
            )

    def __call__(self, examples: List[Dict[str, np.ndarray]], pad_to_multiple_of: int = 16) -> Dict[str, np.ndarray]:
        # Handle dict or lists with proper padding and conversion to tensor.
        # convert list to dict and tensorize input
        # flatten list of dicts in tuple to list of dicts
        if isinstance(examples[0], tuple):
            examples = [sample_dict for sample_tuple in examples for sample_dict in sample_tuple]

        batch = self.tokenizer.pad(examples, pad_to_multiple_of=pad_to_multiple_of, return_tensors=TensorType.NUMPY, return_attention_mask=False)

        # If special token mask has been preprocessed, pop it from the dict.
        special_tokens_mask = batch.pop("special_tokens_mask", None)

        batch["input_ids"], batch["labels"] = self.mask_tokens(
            batch["input_ids"], special_tokens_mask=special_tokens_mask
        )
        # convert to tensor
        batch = {k: torch.from_numpy(v) for k, v in batch.items()}
        return batch

    def mask_tokens(
        self, inputs: np.ndarray, special_tokens_mask: Optional[np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        """
        labels = inputs.copy()
        # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
        probability_matrix = np.full(labels.shape, self.mlm_probability)
        special_tokens_mask = special_tokens_mask.astype("bool")

        probability_matrix[special_tokens_mask] = 0.0
        masked_indices = np.random.binomial(1, probability_matrix).astype("bool")
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = np.random.binomial(1, np.full(labels.shape, 0.8)).astype("bool") & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        # 10% of the time, we replace masked input tokens with random word
        indices_random = np.random.binomial(1, np.full(labels.shape, 0.5)).astype("bool")
        indices_random &= masked_indices & ~indices_replaced

        random_words = np.random.randint(self.tokenizer.vocab_size, size=labels.shape, dtype="i4")
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels


def tokenize_bert_function(examples, tokenizer, max_seq_length):
    tokenizer_out = tokenizer(
        text=examples["text"],
        return_attention_mask=False,
        return_special_tokens_mask=True, 
    )

    
    input_ids = tokenizer_out["input_ids"]
    special_tokens_mask = tokenizer_out["special_tokens_mask"]
    
    concatenated_ids = np.concatenate(input_ids)
    concatenated_mask = np.concatenate(special_tokens_mask)
    total_length = concatenated_ids.shape[0]
    if total_length >= max_seq_length:
        total_length = (total_length // max_seq_length) * max_seq_length

    concatenated_ids = concatenated_ids[:total_length].reshape(-1, max_seq_length)
    concatenated_mask = concatenated_mask[:total_length].reshape(-1, max_seq_length)

    result = {"input_ids": concatenated_ids, 
              "special_tokens_mask": concatenated_mask}

    return result

from transformers.data.data_collator import *
@dataclass
class DataCollatorForNIMLM:
    """
    data collator for natural instruction dataset modified for masked language modeling evalution (BERT), (not for generative models e.g. GPT, T5)
    """
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_source_length: Optional[int] = None
    max_target_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100
    return_tensors: str = "pt"
    add_task_name: bool = False
    add_task_definition: bool = True
    num_pos_examples: int = 0
    num_neg_examples: int = 0
    add_explanation: bool = False
    tk_instruct: bool = False
    text_only: bool = False

    def __call__(self, batch, return_tensors=None):

        if return_tensors is None:
            return_tensors = self.return_tensors

        sources = []
        for instance in batch:
            if self.tk_instruct:
                all_valid_encodings = [
                    # instruction only
                    {
                        "add_task_name": False,
                        "add_task_definition": True,
                        "num_pos_examples": 0,
                        "num_neg_examples": 0,
                        "add_explanation": False,
                    },
                    # example only
                    {
                        "add_task_name": False,
                        "add_task_definition": False,
                        "num_pos_examples": 2,
                        "num_neg_examples": 0,
                        "add_explanation": False,
                    },
                    # instruction + pos examples
                    {
                        "add_task_name": False,
                        "add_task_definition": True,
                        "num_pos_examples": 2,
                        "num_neg_examples": 0,
                        "add_explanation": False,
                    },
                    # instruction + pos examples + neg examples
                    {
                        "add_task_name": False,
                        "add_task_definition": True,
                        "num_pos_examples": 2,
                        "num_neg_examples": 2,
                        "add_explanation": False,
                    },
                    # instruction + pos (w. explanation)
                    {
                        "add_task_name": False,
                        "add_task_definition": True,
                        "num_pos_examples": 2,
                        "num_neg_examples": 0,
                        "add_explanation": True,
                    },
                ]
                encoding_schema = random.choice(all_valid_encodings)
                add_task_name = encoding_schema["add_task_name"]
                add_task_definition = encoding_schema["add_task_definition"]
                num_pos_examples = encoding_schema["num_pos_examples"]
                num_neg_examples = encoding_schema["num_neg_examples"]
                add_explanation = encoding_schema["add_explanation"]
            else:
                add_task_name = self.add_task_name
                add_task_definition = self.add_task_definition
                num_pos_examples = self.num_pos_examples
                num_neg_examples = self.num_neg_examples
                add_explanation = self.add_explanation

            task_input = ""
            # add the input first.
            task_input += "Now complete the following example -\n"
            task_input += f"Input: {instance['Instance']['input'].strip()}"
            if not task_input[-1] in string.punctuation:
                task_input += "."
            task_input += "\n"
            task_input += "Output: "

            task_name = ""
            if add_task_name:
                task_name += instance["Task"] + ". "

            definition = ""
            if add_task_definition:
                if isinstance(instance["Definition"], list):
                    definition = (
                        "Definition: " + instance["Definition"][0].strip()
                    )
                else:
                    definition = "Definition: " + instance["Definition"].strip()
                if not definition[-1] in string.punctuation:
                    definition += "."
                definition += "\n\n"

            # try to add positive examples.
            pos_examples = []
            for idx, pos_example in enumerate(
                instance["Positive Examples"][:num_pos_examples]
            ):
                pos_example_str = f" Positive Example {idx+1} -\n"
                pos_example_str += f"Input: {pos_example['input'].strip()}"
                if not pos_example_str[-1] in string.punctuation:
                    pos_example_str += "."
                pos_example_str += "\n"
                pos_example_str += f" Output: {pos_example['output'].strip()}"
                if not pos_example_str[-1] in string.punctuation:
                    pos_example_str += "."
                pos_example_str += "\n"
                if add_explanation and "explanation" in pos_example:
                    pos_example_str += (
                        f" Explanation: {pos_example['explanation'].strip()}"
                    )
                    if not pos_example_str[-1] in string.punctuation:
                        pos_example_str += "."
                    pos_example_str += "\n"
                pos_example_str += "\n"
                if (
                    len(
                        self.tokenizer(
                            definition
                            + " ".join(pos_examples)
                            + pos_example_str
                            + task_input
                        )["input_ids"]
                    )
                    <= self.max_source_length
                ):
                    pos_examples.append(pos_example_str)
                else:
                    break

            # try to add negative examples.
            neg_examples = []
            for idx, neg_example in enumerate(
                instance["Negative Examples"][:num_neg_examples]
            ):
                neg_example_str = f" Negative Example {idx+1} -\n"
                neg_example_str += f"Input: {neg_example['input'].strip()}"
                if not neg_example_str[-1] in string.punctuation:
                    neg_example_str += "."
                neg_example_str += "\n"
                neg_example_str += f" Output: {neg_example['output'].strip()}"
                if not neg_example_str[-1] in string.punctuation:
                    neg_example_str += "."
                neg_example_str += "\n"
                if add_explanation and "explanation" in neg_example:
                    neg_example_str += (
                        f" Explanation: {neg_example['explanation'].strip()}"
                    )
                    if not neg_example_str[-1] in string.punctuation:
                        neg_example_str += "."
                    neg_example_str += "\n"
                neg_example_str += "\n"
                if (
                    len(
                        self.tokenizer(
                            definition
                            + " ".join(pos_examples)
                            + " ".join(neg_examples)
                            + neg_example_str
                            + task_input
                        )["input_ids"]
                    )
                    <= self.max_source_length
                ):
                    neg_examples.append(neg_example_str)
                else:
                    break

            source = (
                task_name
                + definition
                + "".join(pos_examples)
                + "".join(neg_examples)
                + task_input
            )
            tokenized_source = self.tokenizer(source)["input_ids"]
            if len(tokenized_source) <= self.max_source_length:
                sources.append(source)
            else:
                sources.append(
                    self.tokenizer.decode(
                        tokenized_source[: self.max_source_length],
                        skip_special_tokens=True,
                    )
                )

        if self.text_only:
            model_inputs = {"inputs": sources}
        else:
            model_inputs = self.tokenizer(
                sources,
                max_length=self.max_source_length,
                padding=self.padding,
                return_tensors=self.return_tensors,
                truncation=True,
                pad_to_multiple_of=self.pad_to_multiple_of,
            )

        if "output" in batch[0]["Instance"] and batch[0]["Instance"]["output"]:
            # Randomly select one reference if multiple are provided.
            # labels = [random.choice(ex["Instance"]["output"]) for ex in batch]
            labels = [1 if "Yes" in ex["Instance"]["output"][0] else 0 for ex in batch]
            model_inputs["labels"] = torch.tensor(labels)
        else:
            model_inputs["labels"] = None

        model_inputs.pop('token_type_ids', None)
        model_inputs.pop('attention_mask', None)
        return model_inputs
