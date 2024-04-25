# 3DBench - IJCAI 2024

This is an official repo for paper "3DBench: A Scalable 3D Benchmark and Instruction-Tuning Dataset". 

## About

We introduce 3DBench, a scalable benchmark designed for evaluating 3D-LLMs covering ten diverse multi-modal tasks with three types of evaluation metrics. Furthermore, we present a pipeline for automatically acquiring high-quality instruction-tuning datasets. Through extensive experiments, we validate the effectiveness of our dataset by cross-validate 3D-LLMs trained with various protocols. Our findings suggest that existing 3D-LLMs have considerable potential for further improvements in point cloud understanding and reasoning.

<!-- ![Teaser](assets/teaser.jpg) -->
<div style="text-align: center;">
    <img src="assets/pipeline.bmp" alt="Dialogue_Teaser" width=100% >
</div>

## 🔥 News

📆 Coming Soon

>📋Code to acquire scalable dateset will be available. 

📆 [2024/4/25]

>1.3DBench is on [arxiv](https://arxiv.org/abs/2404.14678).

>2.Instuction Tuning Dataset is available on [Baidu Netdisk](https://pan.baidu.com/share/init?surl=3sVl3JtbuJyfhEzChVPN8A&pwd=1234)

## Requirements


To install requirements:

```setup
pip install -r requirements.txt
```

>📋  Describe how to set up the environment, e.g. pip/conda/docker commands, download datasets, etc...

## 1.Acquiring Much House

To acquiring your own house, run this command:

```
python generate、.py --input-data <path_to_data> --alpha 10 --beta 20
```

>📋  Describe.

## 2.Acquiring Metadata

## 3.Acquiring Instruction Tuning Data

## Evaluation

To evaluate your results on 3DBench, run:

```eval
python eval.py --model-file mymodel.json --benchmark 3Dbench
```

>📋  Describe how to evaluate the trained models on benchmarks reported in the paper, give commands that produce the results (section below).

## Pre-trained Models

You can download pretrained models here:

- [My awesome model](https://drive.google.com/mymodel.pth) 

>📋  Give a link to where/how the pretrained models can be downloaded and how they were trained (if applicable).  Alternatively you can have an additional column in your results table with a link to the models.

## Results

Our model achieves the following performance on :

### [Image Classification on ImageNet](https://pan.baidu.com/share/init?surl=3sVl3JtbuJyfhEzChVPN8A&pwd=1234)

| Model name         | Top 1 Accuracy  | Top 5 Accuracy |
| ------------------ |---------------- | -------------- |
| My awesome model   |     85%         |      95%       |

>📋  Include a table of results from your paper, and link back to the leaderboard for clarity and context. If your main result is a figure, include that figure and link to the command or notebook to reproduce it. 

## 🔗 Citation

If you find our work and this codebase helpful, please consider starring this repo 🌟 and cite:

```bibtex
@article{2024arXiv240414678Z,
       author = {{Zhang}, Junjie and {Hu}, Tianci and {Huang}, Xiaoshui and {Gong}, Yongshun and {Zeng}, Dan},
        title = "{3DBench: A Scalable 3D Benchmark and Instruction-Tuning Dataset}",
      journal = {arXiv e-prints},
     keywords = {Computer Science - Computer Vision and Pattern Recognition},
         year = 2024,
}
```

## Contributing

>📋  Pick a licence and describe how to contribute to your code repository. 
