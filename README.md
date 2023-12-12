# CLEVR3D

**Xu Yan***, **Zhihao Yuan***, Yuhao Du, Yinghong Liao, Yao Guo, Shuguang Cui, and Zhen Li 
"Comprehensive Visual Question Answering on Point Clouds through Compositional Scene Manipulation
" [[arxiv]](https://arxiv.org/pdf/2112.11691.pdf).

> Our paper is accepted by TVCG (IEEE Transactions on Visualization and Computer Graphics)

 ![image](img/fig1.png)
 
 
If you find our work useful in your research, please consider citing:
```latex
@article{yan2023comprehensive,
  title={Comprehensive Visual Question Answering on Point Clouds through Compositional Scene Manipulation},
  author={Yan, Xu and Yuan, Zhihao and Du, Yuhao and Liao, Yinghong and Guo, Yao and Cui, Shuguang and Li, Zhen},
  journal={IEEE Transactions on Visualization \& Computer Graphics},
  number={01},
  pages={1--13},
  year={2023},
  publisher={IEEE Computer Society}
}
```


## Installation

### Requirements
- pytorch >= 1.8 
- transformers
- [PyTorch Lightning](https://lightning.ai/docs/pytorch/stable/)

## Data Preparation
The VQA3D data can be found in `data/CLEVR3D/CLEVR3D-REAL.json`. The data has the following structure:
```
{
"question":[
{
    "scan": "f62fd5fd-9a3f-2f44-883a-1e5cf819608e",
    "image_index": 0,
    "question": "Are there the same number of sofas and wide sinks?",
    "answer": "no",
    "template_filename": "compare_integer.json",
    "question_family_index": 0,
    "question_type": "equal_integer"
},
...
]}
```
The scan number is the same as [3RScan](https://github.com/WaldJohannaU/3RScan).
Please download the preprocessed 3RScan data from [Baidu Netdisk](https://pan.baidu.com/s/1q-K79cEeHzUaBJ1ZjkNxvw) (**ifei**). And modify the data path in `lib/config.py`.

## Training
```shell
cd <root dir of this repo>
python main.py --log_dir {LOGNAME} --use_scene_graph --preloading
```


## Evaluation
You cna download our weights from [OneDrive](https://cuhko365-my.sharepoint.com/:u:/g/personal/221019046_link_cuhk_edu_cn/EUZZSwJPTD9Btep3Z2lYa10BqxXJ4ecJydWa_pX5YQk9DQ?e=SkznPm)
```shell
python main.py --test --ckpt_path <dir for the pytorch checkpoint> --use_scene_graph --preloading
```
