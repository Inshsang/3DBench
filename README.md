# 3DBench - IJCAI 2024

This is an official repo for paper "3DBench: A Scalable 3D Benchmark and Instruction-Tuning Dataset". 

## About

We introduce 3DBench, a scalable benchmark designed for evaluating 3D-LLMs covering ten diverse multi-modal tasks with three types of evaluation metrics. Furthermore, we present a pipeline for automatically acquiring high-quality instruction-tuning datasets. Through extensive experiments, we validate the effectiveness of our dataset by cross-validate 3D-LLMs trained with various protocols. Our findings suggest that existing 3D-LLMs have considerable potential for further improvements in point cloud understanding and reasoning.

<!-- ![Teaser](assets/teaser.jpg) -->
<div style="text-align: center;">
    <img src="assets/pipeline.bmp" alt="Dialogue_Teaser" width=100% >
</div>

## 🔥 News

📆 [2024/4/25]

>1.3DBench is on [arxiv](https://arxiv.org/abs/2404.14678).

>2.Instuction Tuning Dataset is available on [Baidu Netdisk](https://pan.baidu.com/share/init?surl=3sVl3JtbuJyfhEzChVPN8A&pwd=1234)

## Requirements

To install requirements:

```bash
# pip install -r requirements.txt
```

>1.📋  You should install necessary packages same with [Prothor](https://github.com/allenai/procthor) to acquire houses.

>2.📋  Then install necessary packages while processsing data and evaluating.

>3.📋  All Packages are ordinary and it will cost little time.

## 1.Acquiring Much House

To acquiring your own house, run this command:

```
python Gethouse.py
```

## 2.Processsing house for GT

To Processs your own gt, run this command:

```bash
python generateGT.py    # leading to easy gt
python CreateGT.py   # leading to text samples format as 'json'
```

>📋  All samples are formated in 'json' like [LAMM](https://github.com/OpenGVLab/LAMM).

## 3.Evaluating answers from your own model

## Evaluation

To evaluate your results on 3DBench, run:

```eval
python common_eval_3d.py    # for visul taskes
python gpt_eva.py    # for text taskes
```

>📋  You should use your own gpt api to replace the xxxx in gpt_eva.py.


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

>📋  The project is CC BY NC 4.0 (allowing only non-commercial use) and models trained using the dataset should not be used outside of research purposes.
