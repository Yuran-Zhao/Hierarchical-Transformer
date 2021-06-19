import pdb
from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    NamedTuple,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import numpy as np
import tokenizers
import torch
from transformers import BatchEncoding

from bottom_data_collator import MAX_LENGTH_OF_BLOCK

EncodedInput = List[int]
MAX_LENGTH_OF_FUNCTION = 50


@dataclass
class TopDataCollatorForFunctionCloneDetection:
    tokenizer: tokenizers.Tokenizer
    mlm: bool = True
    mlm_probability: float = 0.15
    pad_to_multiple_of: Optional[int] = None

    def __post_init__(self):
        # print(self.mlm, self.tokenzier.token_to_id("[MASK]"))
        # input()
        if self.mlm and self.tokenizer.token_to_id("[MASK]") is None:
            raise ValueError(
                "This tokenizer does not have a mask token which is necessary for masked language modeling. "
                "You should pass `mlm=False` to train on causal language modeling instead."
            )

    def __call__(
        self, examples: List[Union[List[int], torch.Tensor, Dict[str, torch.Tensor]]],
    ) -> Dict[str, torch.Tensor]:

        # print(examples)
        # Handle dict or lists with proper padding and conversion to tensor.
        if isinstance(examples[0], (dict, BatchEncoding)):
            # will go this way
            batch = self.pad(
                encoded_inputs=examples,
                return_tensors="pt",
                max_length=MAX_LENGTH_OF_FUNCTION,
                pad_to_multiple_of=self.pad_to_multiple_of,
            )

        # If special token mask has been preprocessed, pop it from the dict.
        # func1_special_tokens_mask = batch.pop("func1_special_tokens_mask", None)
        # func2_special_tokens_mask = batch.pop("func2_special_tokens_mask", None)

        # pdb.set_trace()
        # batch["input_ids"] = torch.squeeze(batch["input_ids"], dim=0)
        # batch["token_type_ids"] = torch.squeeze(batch["token_type_ids"], dim=0)
        # pdb.set_trace()
        return batch

    def pad(
        self,
        encoded_inputs: Union[
            BatchEncoding,
            List[BatchEncoding],
            Dict[str, EncodedInput],
            Dict[str, List[EncodedInput]],
            List[Dict[str, EncodedInput]],
        ],
        padding=True,
        max_length: Optional[int] = None,
        pad_to_multiple_of: Optional[int] = None,
        return_attention_mask: Optional[bool] = None,
        return_tensors=None,
        verbose: bool = True,
    ) -> BatchEncoding:
        """
        Pad a single encoded input or a batch of encoded inputs up to predefined length or to the max sequence length
        in the batch.
        Padding side (left/right) padding token ids are defined at the tokenizer level (with ``self.padding_side``,
        ``self.pad_token_id`` and ``self.pad_token_type_id``)
        .. note::
            If the ``encoded_inputs`` passed are dictionary of numpy arrays, PyTorch tensors or TensorFlow tensors, the
            result will use the same type unless you provide a different tensor type with ``return_tensors``. In the
            case of PyTorch tensors, you will lose the specific device of your tensors however.
        Args:
            encoded_inputs (:class:`~transformers.BatchEncoding`, list of :class:`~transformers.BatchEncoding`, :obj:`Dict[str, List[int]]`, :obj:`Dict[str, List[List[int]]` or :obj:`List[Dict[str, List[int]]]`):
                Tokenized inputs. Can represent one input (:class:`~transformers.BatchEncoding` or :obj:`Dict[str,
                List[int]]`) or a batch of tokenized inputs (list of :class:`~transformers.BatchEncoding`, `Dict[str,
                List[List[int]]]` or `List[Dict[str, List[int]]]`) so you can use this method during preprocessing as
                well as in a PyTorch Dataloader collate function.
                Instead of :obj:`List[int]` you can have tensors (numpy arrays, PyTorch tensors or TensorFlow tensors),
                see the note above for the return type.
            padding (:obj:`bool`, :obj:`str` or :class:`~transformers.file_utils.PaddingStrategy`, `optional`, defaults to :obj:`True`):
                    Select a strategy to pad the returned sequences (according to the model's padding side and padding
                    index) among:
                * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a
                    single sequence if provided).
                * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
                    maximum acceptable input length for the model if that argument is not provided.
                * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
                    different lengths).
            max_length (:obj:`int`, `optional`):
                Maximum length of the returned list and optionally padding length (see above).
            pad_to_multiple_of (:obj:`int`, `optional`):
                If set will pad the sequence to a multiple of the provided value.
                This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability
                >= 7.5 (Volta).
            return_attention_mask (:obj:`bool`, `optional`):
                Whether to return the attention mask. If left to the default, will return the attention mask according
                to the specific tokenizer's default, defined by the :obj:`return_outputs` attribute.
                `What are attention masks? <../glossary.html#attention-mask>`__
            return_tensors (:obj:`str` or :class:`~transformers.file_utils.TensorType`, `optional`):
                If set, will return tensors instead of list of python integers. Acceptable values are:
                * :obj:`'tf'`: Return TensorFlow :obj:`tf.constant` objects.
                * :obj:`'pt'`: Return PyTorch :obj:`torch.Tensor` objects.
                * :obj:`'np'`: Return Numpy :obj:`np.ndarray` objects.
            verbose (:obj:`bool`, `optional`, defaults to :obj:`True`):
                Whether or not to print more information and warnings.
        """
        # If we have a list of dicts, let's convert it in a dict of lists
        # We do this to allow using this method as a collate_fn function in PyTorch Dataloader
        if isinstance(encoded_inputs, (list, tuple)) and isinstance(
            encoded_inputs[0], (dict, BatchEncoding)
        ):
            encoded_inputs = {
                key: [example[key] for example in encoded_inputs]
                for key in encoded_inputs[0].keys()
            }

        required_input = encoded_inputs["func1_input_ids"]

        # If we have PyTorch/TF/NumPy tensors/arrays as inputs, we cast them as python objects
        # and rebuild them afterwards if no return_tensors is specified
        # Note that we lose the specific device the tensor may be on for PyTorch

        first_element = required_input[0]
        if isinstance(first_element, (list, tuple)):
            # first_element might be an empty list/tuple in some edge cases so we grab the first non empty element.
            index = 0
            while len(required_input[index]) == 0:
                index += 1
            if index < len(required_input):
                first_element = required_input[index][0]
        # At this state, if `first_element` is still a list/tuple, it's an empty one so there is nothing to do.
        if not isinstance(first_element, (int, list, tuple)):
            if isinstance(first_element, torch.Tensor):
                return_tensors = "pt" if return_tensors is None else return_tensors
            elif isinstance(first_element, np.ndarray):
                return_tensors = "np" if return_tensors is None else return_tensors
            else:
                raise ValueError(
                    f"type of {first_element} unknown: {type(first_element)}. "
                    f"Should be one of a python, numpy, pytorch or tensorflow object."
                )

            for key, value in encoded_inputs.items():
                encoded_inputs[key] = to_py_obj(value)

        required_input = encoded_inputs["func1_input_ids"]
        # if required_input and not isinstance(required_input[0], (list, tuple)):
        #     encoded_inputs = self._pad(
        #         encoded_inputs,
        #         max_length=max_length,
        #         # padding_strategy=padding_strategy,
        #         pad_to_multiple_of=pad_to_multiple_of,
        #         return_attention_mask=return_attention_mask,
        #     )
        #     return BatchEncoding(encoded_inputs, tensor_type=return_tensors)
        encoded_inputs = self._pad(
            encoded_inputs,
            max_length=max_length,
            # padding_strategy=padding_strategy,
            pad_to_multiple_of=pad_to_multiple_of,
            return_attention_mask=return_attention_mask,
        )
        batch_size = len(required_input)
        assert all(
            len(v) == batch_size for v in encoded_inputs.values()
        ), "Some items in the output dictionary have a different batch size than others."

        batch_outputs = {}
        for i in range(batch_size):
            inputs = dict((k, v[i]) for k, v in encoded_inputs.items())
            for key, value in inputs.items():
                if key not in batch_outputs:
                    batch_outputs[key] = []
                batch_outputs[key].append(value)

        return BatchEncoding(batch_outputs, tensor_type=return_tensors)

    def _pad(
        self,
        encoded_inputs: Union[Dict[str, EncodedInput], BatchEncoding],
        max_length: Optional[int] = None,
        padding_strategy: str = "LONGEST",  # PaddingStrategy = PaddingStrategy.MAX_LENGTH,
        pad_to_multiple_of: Optional[int] = None,
        return_attention_mask: Optional[bool] = None,
    ) -> dict:
        """
            Pad encoded inputs (on left/right and up to predefined length or max length in the batch)
            Args:
                encoded_inputs: Dictionary of tokenized inputs (`List[int]`) or batch of tokenized inputs (`List[List[int]]`).
                max_length: maximum length of the returned list and optionally padding length (see below).
                    Will truncate by taking into account the special tokens.
                padding_strategy: PaddingStrategy to use for padding.
                    - PaddingStrategy.LONGEST Pad to the longest sequence in the batch
                    - PaddingStrategy.MAX_LENGTH: Pad to the max length (default)
                    - PaddingStrategy.DO_NOT_PAD: Do not pad
                    The tokenizer padding sides are defined in self.padding_side:
                        - 'left': pads on the left of the sequences
                        - 'right': pads on the right of the sequences
                pad_to_multiple_of: (optional) Integer if set will pad the sequence to a multiple of the provided value.
                    This is especially useful to enable the use of Tensor Core on NVIDIA hardware with compute capability
                    >= 7.5 (Volta).
                return_attention_mask: (optional) Set to False to avoid returning attention mask (default: set to model specifics)
            """
        # pdb.set_trace()
        # Load from model defaults
        if return_attention_mask is None:
            return_attention_mask = False
            # return_attention_mask = "attention_mask" in self.model_input_names

        func1_required_input = encoded_inputs["func1_input_ids"]
        func1_special_tokens_mask = encoded_inputs["func1_special_tokens_mask"]

        func2_required_input = encoded_inputs["func2_input_ids"]
        func2_special_tokens_mask = encoded_inputs["func2_special_tokens_mask"]

        if (
            max_length is not None
            and pad_to_multiple_of is not None
            and (max_length % pad_to_multiple_of != 0)
        ):
            max_length = ((max_length // pad_to_multiple_of) + 1) * pad_to_multiple_of

        needs_to_be_padded = (
            padding_strategy != "DO_NOT_PAD"
            and any(len(func) < max_length for func in func1_required_input)
            # and any(len(func) < max_length for func in func2_required_input)
        )
        if needs_to_be_padded:
            (
                encoded_inputs["func1_input_ids"],
                encoded_inputs["func1_special_tokens_mask"],
            ) = self.pad_for_functions(
                func1_required_input, func1_special_tokens_mask, max_length
            )
        encoded_inputs["func1_masks"] = [
            [1 if 0 in block else 0 for block in func]
            for func in encoded_inputs["func1_special_tokens_mask"]
        ]

        needs_to_be_padded = (
            padding_strategy != "DO_NOT_PAD"
            # and any(len(func) < max_length for func in func1_required_input)
            and any(len(func) < max_length for func in func2_required_input)
        )
        if needs_to_be_padded:
            (
                encoded_inputs["func2_input_ids"],
                encoded_inputs["func2_special_tokens_mask"],
            ) = self.pad_for_functions(
                func2_required_input, func2_special_tokens_mask, max_length
            )
        encoded_inputs["func2_masks"] = [
            [1 if 0 in block else 0 for block in func]
            for func in encoded_inputs["func2_special_tokens_mask"]
        ]

        # elif return_attention_mask and "attention_mask" not in encoded_inputs:
        #     encoded_inputs["attention_mask"] = [1] * len(required_input)

        return encoded_inputs

    def pad_for_functions(
        self,
        functions_input_ids,
        functions_special_tokens_mask,
        max_length,
        padding_strategy: str = "LONGEST",
    ):
        for index in range(len(functions_input_ids)):

            difference = max(max_length - len(functions_input_ids[index]), 0)
            # if self.padding_side == "right":

            functions_special_tokens_mask[index] = functions_special_tokens_mask[
                index
            ] + [[] for _ in range(difference)]

            functions_input_ids[index] = functions_input_ids[index] + [
                [] for _ in range(difference)
            ]

            # for sub_index in range(max_length):
            #     input_ids[index][sub_index] = self.pad_blocks_helper(
            #         input_ids[index][sub_index], max_length,
            #     )
            (
                functions_input_ids[index],
                functions_special_tokens_mask[index],
            ) = self.pad_for_blocks(
                functions_input_ids[index],
                functions_special_tokens_mask[index],
                MAX_LENGTH_OF_BLOCK,
            )
        # pdb.set_trace()

        return functions_input_ids, functions_special_tokens_mask

    def pad_for_blocks(
        self,
        blocks_input_ids,
        blocks_special_tokens_mask,
        max_length,
        padding_strategy: str = "LONGEST",
    ):
        # pdb.set_trace()
        needs_to_be_padded = padding_strategy != "DO_NOT_PAD" and any(
            len(block) < max_length for block in blocks_input_ids
        )
        if needs_to_be_padded:
            for index in range(len(blocks_input_ids)):
                # NOTE: we need to add a '[CLS]' before the `required_input`
                # so the difference should subtract 1
                difference = max(max_length - len(blocks_input_ids[index]) - 1, 0)
                # if self.padding_side == "right":

                blocks_special_tokens_mask[index] = (
                    [1] + blocks_special_tokens_mask[index] + [1] * difference
                )

                # process each block
                blocks_input_ids[index] = (
                    [[self.tokenizer.token_to_id("[CLS]") for _ in range(3)]]
                    + blocks_input_ids[index]
                    + [[self.tokenizer.token_to_id("[PAD]") for _ in range(3)]]
                    * difference
                )
        return blocks_input_ids, blocks_special_tokens_mask

    # def pad_special_tokens_mask_for_blocks(self, special_tokens_mask, max_length):


def _collate_batch(examples, tokenizer, pad_to_multiple_of: Optional[int] = None):
    """Collate `examples` into a batch, using the information in `tokenizer` for padding if necessary."""
    # Tensorize if necessary.
    if isinstance(examples[0], (list, tuple)):
        examples = [torch.tensor(e, dtype=torch.long) for e in examples]

    # Check if padding is necessary.
    length_of_first = examples[0].size(0)
    are_tensors_same_length = all(x.size(0) == length_of_first for x in examples)
    if are_tensors_same_length and (
        pad_to_multiple_of is None or length_of_first % pad_to_multiple_of == 0
    ):
        return torch.stack(examples, dim=0)

    # If yes, check if we have a `pad_token`.
    if tokenizer._pad_token is None:
        raise ValueError(
            "You are attempting to pad samples but the tokenizer you are using"
            f" ({tokenizer.__class__.__name__}) does not have a pad token."
        )

    # Creating the full tensor and filling it with our data.
    max_length = max(x.size(0) for x in examples)
    if pad_to_multiple_of is not None and (max_length % pad_to_multiple_of != 0):
        max_length = ((max_length // pad_to_multiple_of) + 1) * pad_to_multiple_of
    result = examples[0].new_full([len(examples), max_length], tokenizer.pad_token_id)
    for i, example in enumerate(examples):
        if tokenizer.padding_side == "right":
            result[i, : example.shape[0]] = example
        else:
            result[i, -example.shape[0] :] = example
    return result


def to_py_obj(obj):
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj
