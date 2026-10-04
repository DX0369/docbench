<div align="center">

[English](./README.md)

# CHAOS-Bench

**C**haracter-level **H**allucination **A**ssessment for **O**CR **S**eeing-is-believing

</div>

<p align="center">
<a href="https://arxiv.org/abs/2607.04884"><b>📄 论文</b></a> •
<a href="https://github.com/Tencent-Hunyuan/HunyuanOCR"><b>💻 HunyuanOCR</b></a>
</p>

---

## 📖 简介

**CHAOS-Bench** 是一个用于检验视觉语言模型（VLM）**"所见即所得"**能力的诊断性评测基准，
即考察模型是**真正识别图像中的文字**，还是依赖语言先验**产生幻觉**、脑补出"看起来更合理"的词。

核心思路简单而有效：我们在真实的学术论文页面图像中，对选定词汇注入**字符级篡改**
（例如 `participant` → `qarticipant`、`about` → `abcut`）。这些被篡改的词在视觉上真实存在，
但在语言上并不合理。真正"看像素"的模型应当输出被篡改后的字符串；而依赖语言先验的模型则会
悄悄地把它"自动纠正"回合理的原词——从而暴露出感知与认知之间的鸿沟，也即一种 OCR 幻觉。

## 📊 数据统计

| 项目     | 数值                       |
| -------- | -------------------------- |
| 图片数量 | 500                        |
| 标注样本 | 500                        |
| 领域     | 学术论文页面               |
| 篡改类型 | 字符级词汇编辑（含边界框） |

## 🗂️ 数据格式

压缩包 `CHAOS-Bench.zip` 包含：

```
CHAOS-Bench/
├── CHAOS-Bench.jsonl     # 标注文件，每行一个 JSON 对象
└── images/               # 500 张页面图片（001.png ...）
```

`CHAOS-Bench.jsonl` 每行包含以下字段：

| 字段      | 类型   | 说明                                |
| --------- | ------ | ----------------------------------- |
| `data_id` | string | 样本编号，如 `"001"`                |
| `image`   | string | 图片相对路径，如 `"images/001.png"` |
| `change`  | list   | 该图片中一处或多处注入的篡改        |

`change` 列表中的每个元素描述一个被篡改的词：

| 字段         | 类型      | 说明                                                     |
| ------------ | --------- | -------------------------------------------------------- |
| `ocr_ans`    | string    | 图像中**实际渲染**的被篡改字符串（需要模型识别出的真值） |
| `origin_ans` | string    | 篡改前的**原始正确词**                                   |
| `bbox`       | list[int] | 该词的边界框，`[x1, y1, x2, y2]`                         |

### 示例

```json
{
  "data_id": "002",
  "image": "images/002.png",
  "change": [
    { "ocr_ans": "abcut", "origin_ans": "about", "bbox": [806, 809, 859, 826] },
    { "ocr_ans": "cf", "origin_ans": "of", "bbox": [765, 838, 787, 853] }
  ]
}
```

## 🧪 评测方式

1. 解压 `CHAOS-Bench.zip`。
2. 对每张图片，提示模型识别指定区域的文字（或进行整页识别），并定位模型输出中对应每个 `bbox` 的内容。
3. 将模型输出与 `ocr_ans` 比对：
   - 输出与 **`ocr_ans`** 一致 → 模型**真正读取了像素** ✅
   - 输出与 **`origin_ans`**（合理的原词）一致 → 模型依赖语言先验**产生了幻觉** ❌
4. 统计模型忠实还原被篡改 `ocr_ans` 的比例，作为"所见即所得 / 抗幻觉"得分。

## 📚 引用

如果 CHAOS-Bench 对你有帮助，请引用 HunyuanOCR-1.5：

```bibtex
@article{HunyuanOCR_1_5_2026,
  title   = {{HunyuanOCR-1.5}: Making Lightweight {OCR} {VLMs} Faster and Better},
  author  = {Li, Gengluo and Wan, Xingyu and Peng, Shangpin and Wang, Weinong and Feng, Hao and Du, Yongkun and Wu, Binghong and Ruan, Zheng and Lu, Zhiqiong and Wu, Liang and Lyu, Pengyuan and Shen, Huawen and Lin, Zibin and Hu, Shijing and Yang, Jieneng and Wen, Hongbing and Yu, Guanghua and Liu, Hong and Wang, Bochao and Ma, Can and Hu, Han and Zhang, Chengquan and Zhou, Yu},
  journal = {arXiv preprint arXiv:2607.04884},
  year    = {2026}
}
```
