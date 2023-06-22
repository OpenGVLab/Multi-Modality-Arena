"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import logging
import json
from typing import Dict

from omegaconf import OmegaConf
from .registry import registry


class Config:
    def __init__(self, cfg_path, DATA_DIR):
        self.config = {}
        registry.register("configuration", self)
        config = OmegaConf.load(cfg_path)
        model_config = self.build_model_config(config, DATA_DIR=DATA_DIR)
        preprocess_config = self.build_preprocess_config(config)
        self.config = OmegaConf.merge(model_config, preprocess_config)

    def _build_opt_list(self, opts):
        opts_dot_list = self._convert_to_dot_list(opts)
        return OmegaConf.from_dotlist(opts_dot_list)

    @staticmethod
    def build_model_config(config, **kwargs):
        model = config.get("model", None)
        assert model is not None, "Missing model configuration file."

        model_cls = registry.get_model_class(model.arch)
        assert model_cls is not None, f"Model '{model.arch}' has not been registered."

        model_type = kwargs.get("model.model_type", None)
        if not model_type:
            model_type = model.get("model_type", None)
        # else use the model type selected by user.

        assert model_type is not None, "Missing model_type."

        model_config_path = model_cls.default_config_path(model_type=model_type)

        edited_model = {"model": config["model"]}
        edited_model['model']['prompt_path'] = kwargs.get("DATA_DIR", '.') + '/' + edited_model['model']['prompt_path']
        edited_model['model']['llama_model'] = kwargs.get("DATA_DIR", '.') + '/' + edited_model['model']['llama_model']

        model_config = OmegaConf.create()
        # hiararchy override, customized config > default config
        model_config = OmegaConf.merge(
            model_config,
            OmegaConf.load(model_config_path),
            edited_model,
        )

        return model_config

    @staticmethod
    def build_preprocess_config(config):
        preprocess = config.get("preprocess", None)
        assert preprocess is not None, "Missing preprocess configuration file."
        preprocess_config = OmegaConf.create()
        preprocess_config = OmegaConf.merge(
            preprocess_config,
            {"preprocess": preprocess},
        )

        return preprocess_config

    def _convert_to_dot_list(self, opts):
        if opts is None:
            opts = []

        if len(opts) == 0:
            return opts

        has_equal = opts[0].find("=") != -1

        if has_equal:
            return opts

        return [(opt + "=" + value) for opt, value in zip(opts[0::2], opts[1::2])]

    def get_config(self):
        return self.config

    @property
    def preprocess_cfg(self):
        return self.config.preprocess

    @property
    def model_cfg(self):
        return self.config.model

    def pretty_print(self):
        logging.info("\n=====  Running Parameters    =====")
        logging.info(self._convert_node_to_json(self.config.run))

        logging.info("\n======  Dataset Attributes  ======")
        datasets = self.config.datasets

        for dataset in datasets:
            if dataset in self.config.datasets:
                logging.info(f"\n======== {dataset} =======")
                dataset_config = self.config.datasets[dataset]
                logging.info(self._convert_node_to_json(dataset_config))
            else:
                logging.warning(f"No dataset named '{dataset}' in config. Skipping")

        logging.info(f"\n======  Model Attributes  ======")
        logging.info(self._convert_node_to_json(self.config.model))

    def _convert_node_to_json(self, node):
        container = OmegaConf.to_container(node, resolve=True)
        return json.dumps(container, indent=4, sort_keys=True)

    def to_dict(self):
        return OmegaConf.to_container(self.config)


def node_to_dict(node):
    return OmegaConf.to_container(node)
