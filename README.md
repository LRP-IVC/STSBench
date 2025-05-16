# STSBench: A Spatio-temporal Scenario Benchmark for Multi-modal Large Language Models in Autonomous Driving

We introduce STSBench, a scenario-based framework to benchmark the holistic understanding of vision-language models (VLMs) for autonomous driving. The framework automatically mines pre-defined traffic scenarios from any dataset using ground-truth annotations, provides an intuitive user interface for efficient human verification, and generates multiple-choice questions for model evaluation. Applied to the nuScenes dataset, we present STSBench, the first benchmark that evaluates the spatio-temporal reasoning capabilities of VLMs based on comprehensive 3D perception.  Existing benchmarks typically target off-the-shelf or fine-tuned VLMs for images or videos from a single viewpoint and focus on semantic tasks such as object recognition, captioning, risk assessment, or scene understanding. In contrast, STSBench evaluates driving expert VLMs for end-to-end driving, operating on videos from multi-view cameras or LiDAR. It specifically assesses their ability to reason about both ego-vehicle actions and complex interactions among traffic participants, a crucial capability for autonomous vehicles. The benchmark features 43 diverse scenarios spanning multiple views and frames, resulting in 971 human-verified multiple-choice questions. A thorough evaluation uncovers critical shortcomings in existing models’ ability to reason about fundamental traffic dynamics in complex environments. These findings highlight the urgent need for architectural advances that explicitly model spatio-temporal reasoning. By addressing a core gap in spatio-temporal evaluation, STSBench enables the development of more robust and explainable VLMs for autonomous driving.


# Installation
Install the requirements:
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements
```

# STSnu Dataset Generation

STSBench extracts scenarios from the nuScenes dataset and subsequently derives multiple-choice questions. This process involves the following steps:

## Preparing nuScenes Data

This step involves extracting and formatting the ground truth data from the nuScenes dataset. The resulting format is specifically designed to facilitate consecutive scenario mining and the generation of multiple-choice questions.
```bash
python nuscenes_extractor.py --dataroot <path to>/nuscenes/v1.0-trainval/ --db_path nuScenes.db
```

## Scenario Mining
We automatically mine the pre-defined scenarios and save them in a scenario database. 
```bash
python mine_maneuvers.py --db_path nuScenes.db
```

## Scenario Sampling
To achieve a more balanced benchmark, we optionally sample over-represented scenarios. This step mitigated the impact of overly challenging examples by filtering out those with high occlusion and distant agents/objects.
```bash
python downsample_maneuvers.py --db_path nuScenes.db
```

## Verification

To ensure annotation quality, we employ a streamlined human verification process. Instead of exhaustive frame-by-frame review, annotators perform two key checks: Scenario Confirmation: Verify the presence of a mined scenario (rejecting false positives) and Negative Example Validation: Confirm that negative examples are indeed invalid (identifying false negatives).

```bash
python verify.py --dataroot <path to>/nuscenes/v1.0-trainval/ --db_path nuScenes.db
```

Our verification tool is built using [rerun](https://github.com/rerun-io/rerun) for visualization. An example of a verification scenario looks like:

### VQA Generation
Since different methods require different prompt styles and referals, we provide scripts to generate the prompts for every method we evaluated in the paper.

#### Hugging Face

```bash
python -m vqa_extractor.hf --db_path nuScenes.db --save_path STSnu.json
```

#### LLM (Llama 3.2, DeepSeek V3, GPT-4o)

```bash
python -m vqa_extractor.llm --db_path nuScenes.db --save_path STSnu.json
```

#### InternVL 2.5
```bash
python -m vqa_extractor.internvl --db_path nuScenes.db --save_path STSnu.json
```

#### Qwen2.5-VL 
```bash
python -m vqa_extractor.qwen --db_path nuScenes.db --save_path STSnu.json
```

#### Senna-VLM
```bash
python -m vqa_extractor.senna --db_path nuScenes.db --save_path STSnu.json
```

#### OmniDrive
```bash
python -m vqa_extractor.omnidrive --db_path nuScenes.db --save_path STSnu.json
```

#### DriveMM
```bash
python -m vqa_extractor.drivemm --db_path nuScenes.db --save_path STSnu.json
```
