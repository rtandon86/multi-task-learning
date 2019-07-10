#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul  6 19:29:05 2019

@author: raghav
"""

from typing import Dict, List, Tuple
import logging
import os
from overrides import overrides
# NLTK is so performance orientated (ha ha) that they have lazy imports. Why? Who knows.
from nltk.corpus.reader.bracket_parse import BracketParseCorpusReader # pylint: disable=no-name-in-module
from nltk.tree import Tree
from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import TextField, SequenceLabelField, Field
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Token
from allennlp.common.checks import ConfigurationError


logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

@DatasetReader.register("pos_reader_test")
class PosReaderTest(DatasetReader):
    def __init__(self,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 use_pos_tags: bool = True,
                 lazy: bool = False,
                 label_namespace: str = "pos_labels") -> None:
        super().__init__(lazy=lazy)
        self._token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}
        self._use_pos_tags = use_pos_tags
        self._label_namespace = label_namespace

    @overrides
    def _read(self, file_path):
        # if `file_path` is a URL, redirect to the cache
        file_path = cached_path(file_path)
        files = []
        for r, d, f in os.walk(file_path):
            for file in f:
                if '.tree' in file:
                    files.append(os.path.join(r, file))
                    
        for f in files:
            directory, filename = os.path.split(f)
            logger.info("Reading instances from lines in file at: %s", file_path)
            for parse in BracketParseCorpusReader(root=directory, fileids=[filename]).parsed_sents():

                self._strip_functional_tags(parse)
            # This is un-needed and clutters the label space.
            # All the trees also contain a root S node.
                if parse.label() == "VROOT" or parse.label() == "TOP":
                    parse = parse[0]
                pos_tags = [x[1] for x in parse.pos()] if self._use_pos_tags else None
                yield self.text_to_instance(parse.leaves(), pos_tags)

    @overrides
    def text_to_instance(self, # type: ignore
                         tokens: List[str],
                         pos_tags: List[str] = None) -> Instance:

        # pylint: disable=arguments-differ
        text_field = TextField([Token(x) for x in tokens], token_indexers=self._token_indexers)
        fields: Dict[str, Field] = {"tokens": text_field}

        if self._use_pos_tags and pos_tags is not None:
            pos_tag_field = SequenceLabelField(labels=pos_tags, sequence_field=text_field,
                                               label_namespace=self._label_namespace)
            fields["tags"] = pos_tag_field
        elif self._use_pos_tags:
            raise ConfigurationError("use_pos_tags was set to True but no gold pos"
                                     " tags were passed to the dataset reader.")
        return Instance(fields)

    def _strip_functional_tags(self, tree: Tree) -> None:
        clean_label = tree.label().split("=")[0].split("-")[0].split("|")[0]
        tree.set_label(clean_label)
        for child in tree:
            if not isinstance(child[0], str):
                self._strip_functional_tags(child)

    def _get_gold_spans(self, # pylint: disable=arguments-differ
                        tree: Tree,
                        index: int,
                        typed_spans: Dict[Tuple[int, int], str]) -> int:
        # NLTK leaves are strings.
        if isinstance(tree[0], str):
            end = index + len(tree)
        else:
            # otherwise, the tree has children.
            child_start = index
            for child in tree:
                # typed_spans is being updated inplace.
                end = self._get_gold_spans(child, child_start, typed_spans)
                child_start = end
            span = (index, end - 1)
            current_span_label = typed_spans.get(span)
            if current_span_label is None:
                typed_spans[span] = tree.label()
            else:
                typed_spans[span] = tree.label() + "-" + current_span_label

        return end