import os

from opencompass.datasets import CustomDataset
from opencompass.models import OpenAISDK
from opencompass.openicl.icl_evaluator import AccEvaluator
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from mmengine.config import read_base

with read_base():
    from opencompass.configs.datasets.longbench.longbenchhotpotqa.longbench_hotpotqa_gen_6b3efc import LongBench_hotpotqa_datasets
    from opencompass.configs.datasets.longbench.longbenchgov_report.longbench_gov_report_gen_54c5b0 import LongBench_gov_report_datasets

DATA_DIR = "./data"
SELECTED = os.environ.get("SELECTED_DATASET", "all")
MODEL_DIR = os.environ.get(
    "MODEL_DIR",
    os.path.abspath(os.path.join(os.getcwd(), "..", "..", "Qwen3.5-27B")),
)


def enabled(name):
    return SELECTED == "all" or SELECTED == name


def build_longbench_local(base_cfg, file_name, abbr, max_out_len):
    d = dict(base_cfg)
    d["abbr"] = abbr
    d["type"] = CustomDataset
    d["path"] = os.path.join(DATA_DIR, file_name)
    d["local_mode"] = True
    d.pop("name", None)
    d["infer_cfg"] = dict(d.get("infer_cfg", {}))
    d["infer_cfg"]["inferencer"] = dict(d["infer_cfg"].get("inferencer", {}))
    d["infer_cfg"]["inferencer"]["max_out_len"] = max_out_len
    return d


def build_ruler_dataset(file_name, abbr, max_out_len):
    return dict(
        abbr=abbr,
        type=CustomDataset,
        path=os.path.join(DATA_DIR, file_name),
        local_mode=True,
        reader_cfg=dict(
            input_columns=["prompt"],
            output_column="target_text",
        ),
        infer_cfg=dict(
            prompt_template=dict(
                type=PromptTemplate,
                template=dict(round=[dict(role="HUMAN", prompt="{prompt}")]),
            ),
            retriever=dict(type=ZeroRetriever),
            inferencer=dict(type=GenInferencer, max_out_len=max_out_len),
        ),
        eval_cfg=dict(
            evaluator=dict(type=AccEvaluator),
            pred_role="BOT",
        ),
    )


datasets = []

if enabled("hotpotqa"):
    datasets.append(build_longbench_local(
        LongBench_hotpotqa_datasets[0], "hotpotqa.jsonl", "hotpotqa", 1024))

if enabled("gov_report"):
    datasets.append(build_longbench_local(
        LongBench_gov_report_datasets[0], "gov_report.jsonl", "gov_report", 1024))

if enabled("retrieval_multi_point"):
    datasets.append(build_ruler_dataset(
        "retrieval_multi_point.jsonl", "retrieval_multi_point", 96))

if enabled("aggregation_keyword_aggregation"):
    datasets.append(build_ruler_dataset(
        "aggregation_keyword_aggregation.jsonl", "aggregation_keyword_aggregation", 128))

work_dir = "./output/local_accuracy_qwen35"

api_meta_template = dict(
    round=[
        dict(role="HUMAN", api_role="HUMAN"),
        dict(role="BOT", api_role="BOT", generate=True),
    ]
)

models = [
    dict(
        abbr="Qwen3.5-27B-vLLM",
        type=OpenAISDK,
        path="Qwen3.5-27B",
        openai_api_base="http://127.0.0.1:8001/v1",
        tokenizer_path=MODEL_DIR,
        key="EMPTY",
        meta_template=api_meta_template,
        temperature=0,
        query_per_second=1,
        max_out_len=1024,
        max_seq_len=32768,
        pred_postprocessor=dict(
            type="opencompass.utils.text_postprocessors.extract_non_reasoning_content"
        ),
        batch_size=2,
        verbose=True,
    )
]

del enabled
del build_longbench_local
del build_ruler_dataset
del LongBench_hotpotqa_datasets
del LongBench_gov_report_datasets
del MODEL_DIR
del DATA_DIR
del SELECTED
del os
del CustomDataset
del OpenAISDK
del AccEvaluator
del GenInferencer
del PromptTemplate
del ZeroRetriever
del read_base
