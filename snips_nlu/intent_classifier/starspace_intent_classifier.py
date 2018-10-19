# coding=utf-8
from __future__ import unicode_literals

import io
from subprocess import check_call

from future.utils import iteritems
from pathlib import Path

from snips_nlu.constants import (DATA, DATA_PATH, ENTITY, ENTITY_KIND,
                                 INTENT_IDS, TEXT, UTTERANCES)
from snips_nlu.dataset import get_text_from_chunks
from snips_nlu.entity_parser.builtin_entity_parser import is_builtin_entity
from snips_nlu.intent_classifier import IntentClassifier
from snips_nlu.intent_classifier.featurizer import (
    _builtin_entity_to_feature, _entity_name_to_feature,
    _get_word_cluster_features, _normalize_stem)
from snips_nlu.languages import get_default_sep
from snips_nlu.preprocessing import tokenize_light
from snips_nlu.utils import temp_dir


class StartSpaceIntentClassifier(IntentClassifier):
    def __init__(self, config, **shared):
        super(StartSpaceIntentClassifier, self).__init__(config, **shared)
        self.language = None
        self.model = None
        self.starspace_binary_path = str(DATA_PATH / 'starspace')
        self.query_predict_binary_path = str(DATA_PATH / 'starspace')

    def fit(self, dataset):
        self.fit_builtin_entity_parser_if_needed(dataset)
        self.fit_custom_entity_parser_if_needed(dataset)

        starspace_dataset = self.dataset_to_starspace_file(dataset)

        with temp_dir() as tmp_dir:
            dataset_path = tmp_dir / "dataset.txt"
            with dataset_path.open("w", encoding="utf-8") as f:
                f.write(starspace_dataset)
            model_path = tmp_dir / "model"
            cmd = ["./%s" % self.starspace_binary_path, "train", "-trainFile",
                   str(dataset), "-model", str(model_path)]
            check_call(cmd)
            with io.open(model_path, "rb", encoding="utf-8") as f:
                self.model = f.read()

    def get_intent(self, text, intents_filter):
        with temp_dir() as tmp_dir:
            model_path = tmp_dir / "model"
            with model_path.open("w", encoding="utf-8") as f:
                f.write(self.model)
            prediction_path = tmp_dir / "prediction"
            cmd = ["./%s" % self.query_predict_binary_path, "test", "-model",
                   f.name, "-model", str(model_path), "-testFile",
                   "-predictionFile", str(prediction_path)]
            check_call(cmd)

    def dataset_to_starspace_file(self, dataset):
        self.language = dataset["language"]
        queries = []
        for intent in iteritems(dataset[MULTI_INTENTS]):
            intents = intent[INTENT_IDS]
            intents = sorted(intents)  # For testing
            joined_labels = " ".join(
                "__label__%s" % intent for intent in intents)
            for u in intent[UTTERANCES]:
                features = self.featurize_utterance(u)
                features = sorted(features)  # For testing
                queries.append(" ".join(features) + " " + joined_labels)
        return queries

    def featurize_utterance(self, utterance):
        features = []
        utterance_text = get_text_from_chunks(utterance[DATA])
        utterance_tokens = tokenize_light(utterance_text, self.language)
        if not utterance_tokens:
            return features

        word_clusters_features = _get_word_cluster_features(
            utterance_tokens, self.config.word_clusters_name, self.language)
        normalized_stemmed_tokens = [
            _normalize_stem(t, self.language, self.config.use_stemming)
            for t in utterance_tokens
        ]

        custom_entities = self.custom_entity_parser.parse(
            " ".join(normalized_stemmed_tokens))
        if self.config.unknownword_replacement_string
            custom_entities = [
                e for e in custom_entities
                if e["value"] != self.config.unknownword_replacement_string
            ]
        custom_entities_features = [
            _entity_name_to_feature(e[ENTITY_KIND], self.language)
            for e in custom_entities]

        builtin_entities = self.builtin_entity_parser.parse(
            utterance_text, use_cache=True)
        builtin_entities_features = [
            _builtin_entity_to_feature(ent[ENTITY_KIND], self.language)
            for ent in builtin_entities
        ]

        # We remove values of builtin slots from the utterance to avoid learning
        # specific samples such as '42' or 'tomorrow'
        filtered_normalized_stemmed_tokens = [
            _normalize_stem(chunk[TEXT], self.language,
                            self.config.use_stemming)
            for chunk in utterance[DATA]
            if ENTITY not in chunk or not is_builtin_entity(chunk[ENTITY])
        ]

        features = get_default_sep(self.language).join(
            filtered_normalized_stemmed_tokens)
        if builtin_entities_features:
            features += " " + " ".join(sorted(builtin_entities_features))
        if custom_entities_features:
            features += " " + " ".join(sorted(custom_entities_features))
        if word_clusters_features:
            features += " " + " ".join(sorted(word_clusters_features))
        return features


def read_prediction(path):
    path = Path(path)
    with path.open("w", encoding="utf-8") as f:
        predictions =

