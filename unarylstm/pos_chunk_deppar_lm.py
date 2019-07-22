#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  8 17:36:32 2019

@author: raghav
"""

import logging
from typing import Dict
from overrides import overrides
from typing import Optional

import torch

from allennlp.common import Params
from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.modules import Seq2SeqEncoder
from allennlp.modules.token_embedders import Embedding
from allennlp.nn import RegularizerApplicator, InitializerApplicator
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.models.biaffine_dependency_parser import BiaffineDependencyParser

from unarylstm.pos_tagger import PosSimpleTagger
from unarylstm.chunk_tagger import ChunkSimpleTagger
from unarylstm.lstm_swag import LstmSwag
from unarylstm.shortcut_connect import ShortcutConnectTextFieldEmbedder

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

@Model.register("pos_chunk_deppar_lm")
class LayerPOSChunkDepparLM(Model):
    """
    A class that implement two tasks: POS (Simple Tagger) and Chunk (Simple Tagger).
    
    Parameters
    ----------
    vocab: ``allennlp.data.Vocabulary``, required.
        The vocabulary fitted on the data.
    params: ``allennlp.common.Params``, required
        Configuration parameters for the multi-task model.
    regularizer: ``allennlp.nn.RegularizerApplicator``, optional (default = None)
        A reguralizer to apply to the model's layers.
    """

    def __init__(self, vocab: Vocabulary,
                 params: Params,
                 regularizer: Optional[RegularizerApplicator] = None):

        super(LayerPOSChunkDepparLM, self).__init__(vocab=vocab, regularizer=regularizer)

        # Base text Field Embedder
        text_field_embedder_params = params.pop("text_field_embedder")
        text_field_embedder = BasicTextFieldEmbedder.from_params(vocab=vocab, params=text_field_embedder_params)
        self._text_field_embedder = text_field_embedder

        ############
        # POS Stuffs
        ############
        pos_params = params.pop("pos")

        # Encoder
        encoder_pos_params = pos_params.pop("encoder")
        encoder_pos = Seq2SeqEncoder.from_params(encoder_pos_params)
        self._encoder_pos = encoder_pos

        # Tagger POS - Simple Tagger
        tagger_pos_params = pos_params.pop("tagger")
        # Can be updated to the pos model that is created
        tagger_pos = PosSimpleTagger(
            vocab=vocab,
            text_field_embedder=self._text_field_embedder,
            encoder=self._encoder_pos,
            label_namespace=tagger_pos_params.pop("label_namespace", "labels"),
            regularizer=regularizer,
        )
        self._tagger_pos = tagger_pos

        ############
        # Chunk Stuffs
        ############
        chunk_params = params.pop("chunk")

        # Encoder
        encoder_chunk_params = chunk_params.pop("encoder")
        encoder_chunk = Seq2SeqEncoder.from_params(encoder_chunk_params)
        self._encoder_chunk = encoder_chunk

        shortcut_text_field_embedder = ShortcutConnectTextFieldEmbedder(
            base_text_field_embedder=self._text_field_embedder, previous_encoders=[self._encoder_pos]
        )
        self._shortcut_text_field_embedder = shortcut_text_field_embedder

        # Tagger: Chunk - CRF Tagger
        tagger_chunk_params = chunk_params.pop("tagger")
        tagger_chunk = ChunkSimpleTagger(
            vocab=vocab,
            text_field_embedder=self._shortcut_text_field_embedder,
            encoder=self._encoder_chunk,
            label_namespace=tagger_chunk_params.pop("label_namespace", "labels"),
            label_encoding=tagger_chunk_params.pop("label_encoding", None),
            regularizer=regularizer,
        )
        self._tagger_chunk = tagger_chunk

        ###########
        # Dependency Parsing Stuffs
        ###########
        deppar_params = params.pop("deppar")

        # Encoder
        encoder_deppar_params = deppar_params.pop("encoder")
        encoder_deppar = Seq2SeqEncoder.from_params(encoder_deppar_params)
        self._encoder_deppar = encoder_deppar

        shortcut_text_field_embedder_deppar = ShortcutConnectTextFieldEmbedder(
            base_text_field_embedder=self._text_field_embedder, previous_encoders=[self._encoder_pos, self._encoder_chunk]
        )
        self._shortcut_text_field_embedder_deppar = shortcut_text_field_embedder_deppar

        # Parser: Dependency Parser - Biaffine Parser
        parser_deppar_params = deppar_params.pop("parser")
        embedding_deppar_params = deppar_params.pop("pos_tag_embedding")
        embedding = Embedding(num_embeddings = vocab.get_vocab_size('tokens'),
                              embedding_dim = embedding_deppar_params.pop_int("embedding_dim"),
                              vocab_namespace = embedding_deppar_params.pop("vocab_namespace")) 
        init_params = parser_deppar_params.pop("initializer", None)
        initializer = (
            InitializerApplicator.from_params(init_params) if init_params is not None else InitializerApplicator()
        )

        tagger_deppar = BiaffineDependencyParser(
                                        vocab=vocab,
                                        text_field_embedder=self._shortcut_text_field_embedder_deppar,
                                        encoder=self._encoder_deppar,
                                        tag_representation_dim= parser_deppar_params.pop_int("tag_representation_dim"),
                                        arc_representation_dim= parser_deppar_params.pop_int("arc_representation_dim"),
                                        pos_tag_embedding = embedding,
                                        use_mst_decoding_for_validation = parser_deppar_params.pop("use_mst_decoding_for_validation"),
                                        dropout = parser_deppar_params.pop_float("dropout"),
                                        input_dropout = parser_deppar_params.pop_float("input_dropout"),
                                        initializer = initializer,
                                        regularizer = regularizer)
        self._tagger_deppar = tagger_deppar

        ###########
        # LM Stuffs
        ###########
        lm_params = params.pop("lm")

        # Encoder
        encoder_lm_params = lm_params.pop("encoder")
        encoder_lm = Seq2SeqEncoder.from_params(encoder_lm_params)
        self._encoder_lm = encoder_lm

        shortcut_text_field_embedder_lm = ShortcutConnectTextFieldEmbedder(
            base_text_field_embedder=self._text_field_embedder, previous_encoders=[self._encoder_pos, self._encoder_chunk, self._encoder_deppar]
        )
        self._shortcut_text_field_embedder_lm = shortcut_text_field_embedder_lm

        # Classifier: LM
        tagger_lm = LstmSwag(
            vocab=vocab,
            text_field_embedder=self._shortcut_text_field_embedder_lm,
            encoder=self._encoder_lm)
        self._tagger_lm = tagger_lm

        logger.info("Multi-Task Learning Model has been instantiated.")

    @overrides
    def forward(self, tensor_batch, for_training: bool = False, task_name: str = "pos") -> Dict[str, torch.Tensor]:
        # pylint: disable=arguments-differ

        tagger = getattr(self, "_tagger_%s" % task_name)
        return tagger.forward(**tensor_batch)

    @overrides
    def get_metrics(self, task_name: str, reset: bool = False, full: bool = False) -> Dict[str, float]:

        task_tagger = getattr(self, "_tagger_" + task_name)
        return task_tagger.get_metrics(reset)

    @classmethod
    def from_params(cls, vocab: Vocabulary, params: Params, regularizer: RegularizerApplicator) -> "LayerPOSChunk":
        return cls(vocab=vocab, params=params, regularizer=regularizer)