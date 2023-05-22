# CLEVR3D

**Xu Yan***, **Zhihao Yuan***, Yuhao Du, Yinghong Liao, Yao Guo, Zhen Li, and Shuguang Cui, 
"*Comprehensive Visual Question Answering on Point Clouds through Compositional Scene Manipulation
*" [[arxiv]](https://arxiv.org/pdf/2112.11691.pdf).

 ![image](img/fig1.png)
 
 
If you find our work useful in your research, please consider citing:
```latex
@article{yan2021clevr3d,
  title={Comprehensive Visual Question Answering on Point Clouds through Compositional Scene Manipulation},
  author={Yan, Xu and Yuan, Zhihao and Du, Yuhao and Liao, Yinghong and Guo, Yao and Li, Zhen and Cui, Shuguang},
  journal={arXiv preprint arXiv:2112.11691},
  year={2021}
}
}
```
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


The paper is still under-review and the codes will be polished in the future...
