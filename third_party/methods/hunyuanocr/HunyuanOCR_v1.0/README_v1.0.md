<div align="center">

[中文阅读](./README_zh_v1.0.md)

</div>

<div align="center">

# HunyuanOCR

</div>

<p align="center">
 <img src="./assets/hyocr-head-img.png" width="80%"/> <br>
</p>

<p align="center">
<a href="https://hunyuan.tencent.com/chat/HunyuanDefault?modelId=HY-OCR-1.0&mid=308&from=vision-zh"><b>🎯 Demo</b></a> |
<a href="https://huggingface.co/tencent/HunyuanOCR"><b>📥 Model Download</b></a> |
<a href="https://arxiv.org/abs/2511.19575"><b>📄 Technical Report</b></a>
</p>

## 🤝 Join Our Community

<div align="center">

|                      Wechat Discussion Group                      |                      Discord Group                       |
| :---------------------------------------------------------------: | :------------------------------------------------------: |
| <img src="./assets/qrcode_for_hunyuanocr_wechat.jpg" width="150"> | [Join HunyuanOCR Discord](https://discord.gg/XeD3p2MRDk) |

</div>

## 🔥 News

- **[2026/06/02]** 🎉 We have released two new benchmarks. [Chronicles-OCR](https://github.com/VirtualLUOUCAS/Chronicles-OCR) ([arXiv](https://arxiv.org/abs/2605.11960)), an open-source ancient-text perception benchmark covering the evolutionary trajectory of the "Seven Chinese Scripts", is jointly built by the **SSV Digital Culture Lab** and the **SSV Technical Architecture Department**, together with the **Palace Museum** and **Anyang Normal University**. We have also released [ChartArena](https://github.com/pspdada/ChartArena) ([arXiv](https://arxiv.org/abs/2606.01348)), a new chart-parsing benchmark supporting diverse chart types. Welcome to evaluate and provide your valuable feedback!
- **[2026/05/11]** 🎉 We have officially open-sourced two benchmarks on document parsing and text-image machine translation: [Wild-OmniDocBench](https://github.com/VirtualLUOUCAS/Wild_OmniDocBench) and [MMTIT-Bench](https://github.com/VirtualLUOUCAS/MMTIT_Bench). Welcome to evaluate and provide your valuable feedback!
- **[2026/04/08]** 🎉 Our works on document parsing and text-image machine translation have been accepted to the CVPR 2026 Main Conference! Check out the papers: [Towards Real-World Document Parsing via Realistic Scene Synthesis and Document-Aware Training](https://arxiv.org/abs/2603.23885) and [MMTIT-Bench: A Multilingual and Multi-Scenario Benchmark with Cognition-Perception-Reasoning Guided Text-Image Machine Translation](https://arxiv.org/abs/2603.23896).
- **[2026/01/13]** ⭐ We have released a stable official [online demo](https://hunyuan.tencent.com/chat/HunyuanDefault?modelId=HY-OCR-1.0&mid=308&from=vision-zh), feel free to try it out!
- **[2025/11/28]** 🛠️ We fixed vLLM inference bugs and hyperparameter configuration issues such as system prompt. It is recommended to use the latest vLLM installation steps and the [inference script](https://github.com/Tencent-Hunyuan/HunyuanOCR/blob/main/Hunyuan-OCR-master/Hunyuan-OCR-vllm/run_hy_ocr.py) for performance testing. Currently, there is still a certain accuracy difference between Transformers and the vLLM framework (we are working on fixing this).
- **[2025/11/25]** 📝 Inference code and model weights publicly available.

## 📖 Introduction

**HunyuanOCR** stands as a leading end-to-end OCR expert VLM powered by Hunyuan's native multimodal architecture. With a remarkably lightweight 1B parameter design, it has achieved multiple state-of-the-art benchmarks across the industry. The model demonstrates mastery in **complex multilingual document parsing** while excelling in practical applications including **text spotting, open-field information extraction, video subtitle extraction, and photo translation**.

## ✨ Key Features

- 💪 **Efficient Lightweight Architecture**: Built on Hunyuan's native multimodal architecture and training strategy, achieving SOTA performance with only 1B parameters, significantly reducing deployment costs.

- 📑 **Comprehensive OCR Capabilities**: A single model covering classic OCR tasks including text detection and recognition, complex document parsing, open-field information extraction and video subtitle extraction, while supporting end-to-end photo translation and document QA.

- 🚀 **Ultimate Usability**: Deeply embraces the "end-to-end" philosophy of large models - achieving SOTA results with single instruction and single inference, offering greater efficiency and convenience compared to industry cascade solutions.

- 🌏 **Extensive Language Support**: Robust support for over 100 languages, excelling in both single-language and mixed-language scenarios across various document types.

<div align="left">
  <img src="./assets/hyocr-pipeline-v1.png" alt="HunyuanOCR framework" width="80%">
</div>

## 🛠️ Dependencies and Installation

### System Requirements

- 🖥️ Operating System: Linux
- 🐍 Python: 3.12+ (recommended and tested)
- ⚡ CUDA: 12.9
- 🔥 PyTorch: 2.7.1
- 🎮 GPU: NVIDIA GPU with CUDA support
- 🧠 GPU Memory: 20GB (for vLLM)
- 💾 Disk Space: 6GB

## 🚀 Quick Start with vLLM (⭐ Recommended)

- **[HunyuanOCR Usage Guide](https://docs.vllm.ai/projects/recipes/en/latest/Tencent-Hunyuan/HunyuanOCR.html)**

### Installation

```bash
pip install vllm>=0.12.0
pip install -r requirements.txt
```

Note: We suggest to install [cuda-compat-12-9](https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/):

```bash
sudo dpkg -i cuda-compat-12-9_575.57.08-0ubuntu1_amd64.deb
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.9/compat:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
# verify cuda-compat-12-9
ls /usr/local/cuda-12.9/compat
```

### Model Deploy

```bash
vllm serve tencent/HunyuanOCR \
    --no-enable-prefix-caching \
    --mm-processor-cache-gb 0 \
    --gpu-memory-utilization 0.2
```

### Model Inference

```python
from vllm import LLM, SamplingParams
from PIL import Image
from transformers import AutoProcessor

def clean_repeated_substrings(text):
    """Clean repeated substrings in text"""
    n = len(text)
    if n<8000:
        return text
    for length in range(2, n // 10 + 1):
        candidate = text[-length:]
        count = 0
        i = n - length

        while i >= 0 and text[i:i + length] == candidate:
            count += 1
            i -= length

        if count >= 10:
            return text[:n - length * (count - 1)]

    return text

model_path = "tencent/HunyuanOCR"
llm = LLM(model=model_path, trust_remote_code=True)
processor = AutoProcessor.from_pretrained(model_path)
sampling_params = SamplingParams(temperature=0, max_tokens=16384)

img_path = "/path/to/image.jpg"
img = Image.open(img_path)
messages = [
    {"role": "system", "content": ""},
    {"role": "user", "content": [
        {"type": "image", "image": img_path},
        {"type": "text", "text": "检测并识别图片中的文字，将文本坐标格式化输出。"}
    ]}
]
prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = {"prompt": prompt, "multi_modal_data": {"image": [img]}}
output = llm.generate([inputs], sampling_params)[0]
print(clean_repeated_substrings(output.outputs[0].text))
```

### Alternatively, you can also use the provided demo script as follow:

```shell
cd Hunyuan-OCR-master/Hunyuan-OCR-vllm && python run_hy_ocr.py
```

## 🚀 Quick Start with Transformers

### Installation

```bash
pip install git+https://github.com/huggingface/transformers@82a06db03535c49aa987719ed0746a76093b1ec4
```

> **Note**: Currently, Transformers has a certain performance degradation compared to the vLLM framework (we are working hard to fix it), and we will merge the fixed version into the Transformers main branch later.

### Model Inference

```python
from transformers import AutoProcessor
from transformers import HunYuanVLForConditionalGeneration
from PIL import Image
import torch

model_name_or_path = "tencent/HunyuanOCR"
processor = AutoProcessor.from_pretrained(model_name_or_path, use_fast=False)
img_path = "path/to/your/image.jpg"
image_inputs = Image.open(img_path)
messages1 = [
    {"role": "system", "content": ""},
    {
        "role": "user",
        "content": [
            {"type": "image", "image": img_path},
            {"type": "text", "text": (
                "检测并识别图片中的文字，将文本坐标格式化输出。"
            )},
        ],
    }
]
messages = [messages1]
texts = [
    processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
    for msg in messages
]
inputs = processor(
    text=texts,
    images=image_inputs,
    padding=True,
    return_tensors="pt",
)
model = HunYuanVLForConditionalGeneration.from_pretrained(
    model_name_or_path,
    attn_implementation="eager",
    dtype=torch.bfloat16,
    device_map="auto"
)
with torch.no_grad():
    device = next(model.parameters()).device
    inputs = inputs.to(device)
    generated_ids = model.generate(**inputs, max_new_tokens=16384, do_sample=False)
if "input_ids" in inputs:
    input_ids = inputs.input_ids
else:
    print("inputs: # fallback", inputs)
    input_ids = inputs.inputs
generated_ids_trimmed = [
    out_ids[len(in_ids):] for in_ids, out_ids in zip(input_ids, generated_ids)
]
output_texts = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print(output_texts)
```

### Alternatively, you can also use the provided demo script as follow:

```shell
cd Hunyuan-OCR-master/Hunyuan-OCR-hf && python run_hy_ocr.py
```

## 💬 Application-oriented Prompts

| Task                       | Prompt                                                                                                                                                                                                                                                                                                                     |
| -------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Spotting**               | 检测并识别图片中的文字，将文本坐标格式化输出。                                                                                                                                                                                                                                                                             |
| **Document Parsing**       | • 识别图片中的公式，用LaTeX格式表示。<br><br>• 把图中的表格解析为 HTML。<br><br>• 解析图中的图表，对于流程图使用Mermaid格式表示，其他图表使用Markdown格式表示。<br><br>• 提取文档图片中正文的所有信息用markdown格式表示，其中页眉、页脚部分忽略，表格用html格式表达，文档中公式用latex格式表示，按照阅读顺序组织进行解析。 |
| **General Parsing**        | • 提取图中的文字。                                                                                                                                                                                                                                                                                                         |
| **Information Extraction** | • 输出Key的值。<br><br>• 提取图片中的: ['key1','key2', ...] 的字段内容，并按照JSON格式返回。<br><br>• 提取图中的字幕                                                                                                                                                                                                       |
| **Translation**            | 先提取文字，再将文字内容翻译为英文。若是文档，则其中页眉、页脚忽略。公式用latex格式表示，表格用html格式表示。                                                                                                                                                                                                              |

## 📊 Evaluation

> **Note**: Evaluation metrics for competing methods are taken from official reports when available; otherwise, they are reproduced using competitor models or interfaces with the recommended standard instructions.

> **Note**: The HunyuanOCR evaluation metrics are derived using the TensorRT framework, which may slightly differ from the inference methods using Transformers or vLLM.

### Text Spotting Performance on In-house Benchmark

| Model Type              | Methods               | Overall   | Art       | Doc       | Game      | Hand      | Ads       | Receipt   | Screen    | Scene     | Video     |
| ----------------------- | --------------------- | --------- | --------- | --------- | --------- | --------- | --------- | --------- | --------- | --------- | --------- |
| **Traditional methods** | PaddleOCR             | 53.38     | 32.83     | 70.23     | 51.59     | 56.39     | 57.38     | 50.59     | 63.38     | 44.68     | 53.35     |
| **Traditional methods** | BaiduOCR              | 61.9      | 38.5      | **78.95** | 59.24     | 59.06     | 66.7      | **63.66** | 68.18     | 55.53     | 67.38     |
| **General VLM**         | Qwen3VL-2B-Instruct   | 29.68     | 29.43     | 19.37     | 20.85     | 50.57     | 35.14     | 24.42     | 12.13     | 34.90     | 40.1      |
| **General VLM**         | Qwen3VL-235B-Instruct | 53.62     | 46.15     | 43.78     | 48.00     | 68.90     | 64.01     | 47.53     | 45.91     | 54.56     | 63.79     |
| **General VLM**         | Seed-1.6-Vision       | 59.23     | 45.36     | 55.04     | 59.68     | 67.46     | 65.99     | 55.68     | 59.85     | 53.66     | 70.33     |
| **OCR-Specific VLM**    | HunyuanOCR            | **70.92** | **56.76** | 73.63     | **73.54** | **77.10** | **75.34** | 63.51     | **76.58** | **64.56** | **77.31** |

> **Summary**: HunyuanOCR achieves the best overall performance (70.92%) across different scenarios, significantly outperforming both traditional OCR methods and general VLMs.

### Document Parsing Performance on OmniDocBench and Multilingual In-house Benchmark (Edit Distance)

| Model Type                     | Method           | Size | OmniDocBench |       |           |           | Wild-OmniDocBench |           |           |           | DocML     |
| :----------------------------- | :--------------- | :--- | :----------- | :---- | :-------- | :-------- | :---------------- | :-------- | :-------- | :-------- | :-------- |
|                                |                  |      | overall      | text  | formula   | table     | overall           | text      | formula   | table     |           |
| **General VLMs**               | Gemni-2.5-pro    | -    | 88.03        | 0.075 | 85.92     | 85.71     | 80.59             | 0.118     | 75.03     | 78.56     | 82.64     |
| **General VLMs**               | Qwen3-VL-235B    | 235B | 89.15        | 0.069 | 88.14     | 86.21     | 79.69             | 0.09      | 80.67     | 68.31     | 81.40     |
| **Specialized VLMs (Modular)** | MonkeyOCR-pro-3B | 3B   | 88.85        | 0.075 | 87.5      | 86.78     | 70.00             | 0.211     | 63.27     | 67.83     | 56.50     |
| **Specialized VLMs (Modular)** | MinerU2.5        | 1.2B | 90.67        | 0.047 | 88.46     | 88.22     | 70.91             | 0.218     | 64.37     | 70.15     | 52.05     |
| **Specialized VLMs (Modular)** | PaddleOCR-VL     | 0.9B | 92.86        | 0.035 | 91.22     | 90.89     | 72.19             | 0.232     | 65.54     | 74.24     | 57.42     |
| **Specialized VLMs (End2End)** | Mistral-OCR      | -    | 78.83        | 0.164 | 82.84     | 70.03     | -                 | -         | -         | -         | 64.71     |
| **Specialized VLMs (End2End)** | Deepseek-OCR     | 3B   | 87.01        | 0.073 | 83.37     | 84.97     | 74.23             | 0.178     | 70.07     | 70.41     | 57.22     |
| **Specialized VLMs (End2End)** | dots.ocr         | 3B   | 88.41        | 0.048 | 83.22     | 86.78     | 78.01             | 0.121     | 74.23     | 71.89     | 77.50     |
| **Specialized VLMs (End2End)** | **HunyuanOCR**   | 1B   | **94.10**    | 0.042 | **94.73** | **91.81** | **85.21**         | **0.081** | **82.09** | **81.64** | **91.03** |

> **Summary**: HunyuanOCR demonstrates superior performance in multilingual document parsing, achieving the lowest edit distances across most categories.

### Information Extraction (in-house Benchmark) and VQA Performance (OCRBench)

| Model                       | Cards     | Receipts  | Video Subtitles | OCRBench |
| :-------------------------- | :-------- | :-------- | :-------------- | :------- |
| DeepSeek-OCR                | 10.04     | 40.54     | 5.41            | 430      |
| PP-ChatOCR                  | 57.02     | 50.26     | 3.1             | -        |
| Qwen3-VL-2B-Instruct        | 67.62     | 64.62     | 3.75            | 858      |
| Seed-1.6-Vision             | 70.12     | 67.5      | 60.45           | 881      |
| Qwen3-VL-235B-A22B-Instruct | 75.59     | 78.4      | 50.74           | **920**  |
| Gemini-2.5-Pro              | 80.59     | 80.66     | 53.65           | 872      |
| **HunyuanOCR**              | **92.29** | **92.53** | **92.87**       | 860      |

> **Summary**: HunyuanOCR significantly outperforms larger models in cards/receipts processing and video subtitle extraction, while maintaining competitive performance on OCRBench.

### Text Image Translation (in-house Benchmark) Performance

| Method                 | Size   | Other2En | Other2Zh | DoTA (en2zh) |
| ---------------------- | ------ | -------- | -------- | ------------ |
| Gemini-2.5-Flash       | -      | 79.26    | 80.06    | 85.60        |
| Qwen3-VL-235B-Instruct | 235B   | 73.67    | 77.20    | 80.01        |
| Qwen3-VL-8B-Instruct   | 4B     | 75.09    | 75.63    | 79.86        |
| Qwen3-VL-4B-Instruct   | 4B     | 70.38    | 70.29    | 78.45        |
| Qwen3-VL-2B-Instruct   | 2B     | 66.30    | 66.77    | 73.49        |
| PP-DocTranslation      | -      | 52.63    | 52.43    | 82.09        |
| **HunyuanOCR**         | **1B** | 73.38    | 73.62    | 83.48        |

> **Summary**: HunyuanOCR using only 1B of parameters, achieved comparable results to Qwen3-VL-235B in photo translation tasks.

## 💡 Visualizations

<details>
<summary><u style="color: #2E64FE;">Click here to view detailed results.</u></summary>

### Text Spotting

Our model aims to output the text content and corresponding coordinate information of all text appearing in a text image at the line level. It performs exceptionally well in scenarios such as documents, artistic fonts, street views, handwriting, advertisements, invoices, screenshots, games, and videos.

<p align="left">
 <img src="./assets/spotting1_cropped.png" width="40%"/> <br>
 <img src="./assets/vis_document_23.jpg" width="40%"/> <br>
</p>

### Complex Document Processing

Digitizing scanned or photographed images of multilingual documents involves, specifically, organizing the text content within the images according to reading order, using LaTeX format for formulas, and expressing complex tables in HTML format.

<p align="left">
 <img src="./assets/vis_parsing_fig.png" width="40%"/> <br>
  <img src="./assets/show_res_parsing_fig.png" width="40%"/> <br>
  <img src="./assets/vis_parsing_table.png" width="40%"/> <br>
  <img src="./assets/vis_parsing_table_2.png" width="40%"/> <br>
  <img src="./assets/parsing_rgsj.png" width="40%"/> <br>
  <img src="./assets/parsing_rgsjz_2.png" width="40%"/> <br>
  <img src="./assets/qikai1.png" width="40%"/> <br>
  <img src="./assets/guwan1.png" width="40%"/> <br>
  <img src="./assets/parsing_chart1.png" width="40%"/> <br>
  <img src="./assets/vis_parsing_chart1.png" width="40%"/> <br>
  <img src="./assets/vis_parsing_chart2.png" width="40%"/> <br>
  <img src="./assets/vis_parsing_chart3.png" width="40%"/> <br>
</p>

### Open-field Information Extraction

For common cards and tickets, fields of interest (such as name/address/company) are parsed using standard JSON format.

<p align="left">
 <img src="./assets/vis_ie_1.png" width="40%"/> <br>
</p>

<p align="left">
 <img src="./assets/ie_parallel.jpg" width="25%"/> <br>
</p>

**Prompt:**
Extract the content of the fields: ['单价', '上车时间', '发票号码', '省前缀', '总金额', '发票代码', '下车时间', '里程数'] from the image and return it in JSON format.

**Response:**

```json
{
  "单价": "3.00",
  "上车时间": "09:01",
  "发票号码": "42609332",
  "省前缀": "陕",
  "总金额": "￥77.10元",
  "发票代码": "161002018100",
  "下车时间": "09:51",
  "里程数": "26.1km"
}
```

### Video Subtitle Extraction

Our model is capable of automatically extracting subtitles from videos, including bilingual ones.

<p align="left">
 <img src="./assets/vis_subtitle1.png" width="40%"/> <br>
 <img src="./assets/vis_subtitle2.png" width="40%"/> <br>
 <img src="./assets/vis_subtitle3.png" width="37.5%"/> <br>
</p>

### Image Text Translation

Our model is able to translate images of minor languages ​​taken into Chinese or English text format end-to-end. Currently, it mainly supports 14 frequently used minor languages ​​(specifically including: German, Spanish, Turkish, Italian, Russian, French, Portuguese, Arabic, Thai, Vietnamese, Indonesian, Malay, Japanese, and Korean) into Chinese/English, as well as Chinese-English translation function (it won the small model track championship in the ICDAR2025 document end-to-end translation competition).

<p align="left">
 <img src="./assets/translation2.png" width="40%"/> <br>
</p>

</details>

## 📚 Citation

```bibtex
@article{HunyuanOCR_2025,
  title   = {{HunyuanOCR Technical Report}},
  author  = {Team, Hunyuan Vision and Lyu, Pengyuan and Wan, Xingyu and Li, Gengluo and Peng, Shangpin and Wang, Weinong and Wu, Liang and Shen, Huawen and Zhou, Yu and Tang, Canhui and Yang, Qi and Peng, Qiming and Luo, Bin and Yang, Hower and Zhang, Xinsong and Zhang, Jinnian and Peng, Houwen and Yang, Hongming and Xie, Senhao and Zhou, Longsha and Pei, Ge and Wu, Binghong and Yan, Rui and Wu, Kan and Yang, Jieneng and Wang, Bochao and Liu, Kai and Zhu, Jianchen and Jiang, Jie and Linus and Hu, Han and Zhang, Chengquan},
  journal = {arXiv preprint arXiv:2511.19575},
  year    = {2025}
}

@inproceedings{MMTIT_Bench_2026,
  title     = {{MMTIT-Bench}: A Multilingual and Multi-Scenario Benchmark with Cognition-Perception-Reasoning Guided Text-Image Machine Translation},
  author    = {Li, Gengluo and Zhang, Chengquan and Liang, Yupu and Shen, Huawen and Zhang, Yaping and Lyu, Pengyuan and Wang, Weinong and Wan, Xingyu and Zeng, Gangyan and Hu, Han and others},
  booktitle = {Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  pages     = {16593--16602},
  year      = {2026}
}

@article{li2026towardsrealworlddocument,
  title   = {Towards Real-World Document Parsing via Realistic Scene Synthesis and Document-Aware Training},
  author  = {Li, Gengluo and Lyu, Pengyuan and Zhang, Chengquan and Shen, Huawen and Wu, Liang and Wan, Xingyu and Zeng, Gangyan and Hu, Han and Ma, Can and Zhou, Yu},
  journal = {arXiv preprint arXiv:2603.23885},
  year    = {2026}
}

@article{Chronicles_OCR_2026,
  title   = {{Chronicles-OCR}: A Cross-Temporal Perception Benchmark for the Evolutionary Trajectory of Chinese Characters},
  author  = {Li, Gengluo and Peng, Shangpin and Wan, Xingyu and Zhang, Chengquan and Feng, Hao and Xu, Xin and Wu, Pian and Li, Bang and Ding, Zengmao and Liu, Yongge and others},
  journal = {arXiv preprint arXiv:2605.11960},
  year    = {2026}
}

@article{ChartArena_2026,
  title   = {{ChartArena}: Benchmarking Chart Parsing across Languages, Scenarios, and Formats},
  author  = {Peng, Shangpin and Li, Gengluo and Wan, Xingyu and Zhang, Chengquan and Feng, Hao and Wu, Binghong and Shen, Huawen and Wang, Weinong and Cai, Ziyi and Tian, Zhuotao and Hu, Han and Ma, Can and Zhou, Yu},
  journal = {arXiv preprint arXiv:2606.01348},
  year    = {2026}
}

@article{StrucTab_2026,
  title   = {{StrucTab}: A Structured Optimization Framework for Table Parsing},
  author  = {Li, Gengluo and Peng, Shangpin and Zhang, Chengquan and Wu, Binghong and Feng, Hao and Wang, Weinong and Lyu, Pengyuan and Shen, Huawen and Wan, Xingyu and Tian, Zhuotao and Hu, Han and Ma, Can and Zhou, Yu},
  journal = {arXiv preprint arXiv:2606.29905},
  year    = {2026}
}
```

## 🙏 Acknowledgements

We would like to thank [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR), [MinerU](https://github.com/opendatalab/MinerU), [MonkeyOCR](https://github.com/Yuliang-Liu/MonkeyOCR), [DeepSeek-OCR](https://github.com/deepseek-ai/DeepSeek-OCR), [dots.ocr](https://github.com/rednote-hilab/dots.ocr) for their valuable models and ideas.
We also appreciate the benchmarks: [OminiDocBench](https://github.com/opendatalab/OmniDocBench), [OCRBench](https://github.com/Yuliang-Liu/MultimodalOCR/tree/main/OCRBench), [DoTA](https://github.com/liangyupu/DIMTDA).

Special thanks to vLLM and Hugging Face Communities for their Day-0 inference supports.
