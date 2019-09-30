# slightly different from the other dataset reader

from typing import Dict, List
import json
import logging

from overrides import overrides

from allennlp.common import Params
from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import Field, TextField, LabelField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data.tokenizers import Tokenizer, WordTokenizer
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name



@DatasetReader.register("swag_reader")
class SwagReader(DatasetReader):
    """
    Reads a file from the Stanford Natural Language Inference (SNLI) dataset.  This data is
    formatted as jsonl, one json-formatted instance per line.  The keys in the data are
    "gold_label", "sentence1", and "sentence2".  We convert these keys into fields named "label",
    "premise" and "hypothesis".

    Parameters
    ----------
    tokenizer : ``Tokenizer``, optional (default=``WordTokenizer()``)
        We use this ``Tokenizer`` for both the premise and the hypothesis.  See :class:`Tokenizer`.
    token_indexers : ``Dict[str, TokenIndexer]``, optional (default=``{"tokens": SingleIdTokenIndexer()}``)
        We similarly use this for both the premise and the hypothesis.  See :class:`TokenIndexer`.
    """

    def __init__(self,
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 use_only_gold_examples: bool = False,
                 only_end: bool = False) -> None:
        super().__init__(lazy=False)
        self._tokenizer = tokenizer or WordTokenizer()
        self._token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}
        self.use_only_gold_examples = use_only_gold_examples
        self.only_end = only_end

    @overrides
    def _read(self, file_path: str):

        swag = pd.read_csv(file_path)

        if self.use_only_gold_examples and file_path.endswith('train.csv'):
            swag = swag[swag['gold-source'].str.startswith('gold')]

        for _, row in swag.iterrows():
            premise = row['sent1']
            endings = [row['ending{}'.format(i)] for i in range(4)]

            if self.only_end:
                # NOTE: we're JUST USING THE ENDING HERE, so hope that's what you intend
                hypos = endings
            else:
                hypos = ['{} {} {}'.format(row['sent1'], row['sent2'], end) for end in endings]

            yield self.text_to_instance(premise, hypos, label=row['label'] if hasattr(row, 'label') else None)

    @overrides
    def text_to_instance(self,  # type: ignore
                         premise: str,
                         hypotheses: List[str],
                         label: int = None) -> Instance:
        # pylint: disable=arguments-differ
        fields: Dict[str, Field] = {}
        # premise_tokens = self._tokenizer.tokenize(premise)
        # fields['premise'] = TextField(premise_tokens, self._token_indexers)

        # This could be another way to get randomness
        for i, hyp in enumerate(hypotheses):
            hypothesis_tokens = self._tokenizer.tokenize(hyp)
            fields['hypothesis{}'.format(i)] = TextField(hypothesis_tokens, self._token_indexers)

        if label is not None:
            fields['label'] = LabelField(label, skip_indexing=True)
        return Instance(fields)
