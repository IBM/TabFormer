from transformers.tokenization_utils import PreTrainedTokenizer

class TabFormerTokenizer(PreTrainedTokenizer):
    def __init__(
        self,
        unk_token="<|endoftext|>",
        bos_token="<|endoftext|>",
        eos_token="<|endoftext|>",
    ):

        super().__init__(bos_token=bos_token, eos_token=eos_token, unk_token=unk_token)