# coding=utf-8
# Copyright 2021 The HuggingFace Datasets Authors and the current dataset script contributor.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""ScienceBenchmark: real-world text-to-SQL over cordis/oncomx/sdss.

Mirrors seq2seq/datasets/spider/spider.py's feature schema and
train_syntax.json/dev_syntax.json convention, but with one deliberate
difference: spider.py gets each example's schema by live-introspecting the
SQLite file (dump_db_json_schema). That's fine for Spider, where the DB always
matches its tables.json exactly. It is NOT fine here -- cordis_temporary's
live SQLite schema has 3 columns (institutions.document_vectors,
project_members.region_code, project_members.region_name) that don't exist in
ScienceBenchmark's own published tables.json (verified: they're unreferenced
by any foreign key and document_vectors is an empty tsvector-style artifact
from the Postgres->SQLite conversion, not part of the benchmark's intended
queryable schema). data_all_in/preprocess/process_dataset.py builds tables.bin
(and its schema-relations matrix) from tables_merged.json, the same
concatenation of the 3 DBs' original tables.json files used below -- so
reading schema from there instead of the live DB keeps this dataset script
consistent with tables.bin by construction, instead of drifting from it.
"""

import json

import datasets


logger = datasets.logging.get_logger(__name__)


_CITATION = """\
@inproceedings{zhang2023sciencebenchmark,
  title={ScienceBenchmark: A Complex Real-World Benchmark for Evaluating Natural Language to SQL Systems},
  author={Zhang, Yi and Deriu, Jan and Katsogiannis-Meimarakis, George and Kosten, Catherine and Koutrika, Georgia and Stockinger, Kurt},
  year={2023}
}
"""

_DESCRIPTION = """\
ScienceBenchmark is a complex, real-world text-to-SQL benchmark over three scientific
databases (cordis, oncomx, sdss), used here as an out-of-domain evaluation set for a
model trained on Spider.
"""

_HOMEPAGE = "https://github.com/ckosten/sciencebenchmark_dataset"

_LICENSE = "CC BY-SA 4.0"


class ScienceBenchmark(datasets.GeneratorBasedBuilder):
    VERSION = datasets.Version("1.0.0")

    BUILDER_CONFIGS = [
        datasets.BuilderConfig(
            name="sciencebenchmark",
            version=VERSION,
            description="ScienceBenchmark: real-world text-to-SQL over cordis/oncomx/sdss",
        ),
    ]

    def __init__(self, *args, writer_batch_size=None, **kwargs):
        super().__init__(*args, writer_batch_size=writer_batch_size, **kwargs)
        self.schema_cache = dict()

    def _info(self):
        features = datasets.Features(
            {
                "query": datasets.Value("string"),
                "question": datasets.Value("string"),
                "db_id": datasets.Value("string"),
                "db_path": datasets.Value("string"),
                "db_table_names": datasets.features.Sequence(datasets.Value("string")),
                "db_column_names": datasets.features.Sequence(
                    {
                        "table_id": datasets.Value("int32"),
                        "column_name": datasets.Value("string"),
                    }
                ),
                "db_column_types": datasets.features.Sequence(datasets.Value("string")),
                "db_primary_keys": datasets.features.Sequence({"column_id": datasets.Value("int32")}),
                "db_foreign_keys": datasets.features.Sequence(
                    {
                        "column_id": datasets.Value("int32"),
                        "other_column_id": datasets.Value("int32"),
                    }
                ),
                "graph_idx": datasets.features.Value("int32"),
                "raw_question_toks": datasets.features.Sequence(datasets.Value("string")),
            }
        )
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            supervised_keys=None,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        downloaded_filepath = "./data_all_in/data/sciencebenchmark"

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "data_filepath": downloaded_filepath + "/output/train_syntax.json",
                    "tables_filepath": downloaded_filepath + "/output/tables_merged.json",
                    "db_path": downloaded_filepath + "/database",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "data_filepath": downloaded_filepath + "/output/dev_syntax.json",
                    "tables_filepath": downloaded_filepath + "/output/tables_merged.json",
                    "db_path": downloaded_filepath + "/database",
                },
            ),
        ]

    def _generate_examples(self, data_filepath, tables_filepath, db_path):
        """This function returns the examples in the raw (text) form."""
        logger.info("generating examples from = %s", data_filepath)
        if not self.schema_cache:
            for table_schema in json.load(open(tables_filepath, encoding="utf-8")):
                self.schema_cache[table_schema["db_id"]] = table_schema
        with open(data_filepath, encoding="utf-8") as f:
            sciencebenchmark = json.load(f)
            for idx, sample in enumerate(sciencebenchmark):
                db_id = sample["db_id"]
                schema = self.schema_cache[db_id]
                yield idx, {
                    "query": sample["query"],
                    "question": sample["question"],
                    "db_id": db_id,
                    "db_path": db_path,
                    "db_table_names": schema["table_names_original"],
                    "db_column_names": [
                        {"table_id": table_id, "column_name": column_name}
                        for table_id, column_name in schema["column_names_original"]
                    ],
                    "db_column_types": schema["column_types"],
                    "db_primary_keys": [{"column_id": column_id} for column_id in schema["primary_keys"]],
                    "db_foreign_keys": [
                        {"column_id": column_id, "other_column_id": other_column_id}
                        for column_id, other_column_id in schema["foreign_keys"]
                    ],
                    "graph_idx": sample["graph_idx"],
                    "raw_question_toks": sample["raw_question_toks"],
                }
