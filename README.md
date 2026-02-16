# BackTrack4Drum

为鼓手练习生成无鼓伴奏轨。

使用 [Demucs](https://github.com/facebookresearch/demucs) (Meta AI) 从音乐中去除架子鼓。把你喜欢的歌扔进来，拿到去掉鼓的伴奏，跟着一起练。

**在线使用**：[https://huggingface.co/spaces/asherszhang/BackTrack4Drum](https://huggingface.co/spaces/asherszhang/BackTrack4Drum) — 无需安装，打开即用。

**Colab (GPU 加速)**：[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kzjeef/BackTrack4Drum/blob/main/BackTrack4Drum.ipynb) — 免费 T4 GPU，处理更快。

## 前置要求

- Docker
- NVIDIA GPU + [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)

## 快速开始

### 1. 构建镜像

```bash
docker compose build
```

### 2. 放入音频文件

将 MP3/WAV/FLAC 等音频文件放入 `input/` 目录：

```bash
cp your_song.mp3 input/
```

### 3. 运行

```bash
docker compose run --rm drum-remover
```

处理完成后，去掉鼓的文件会出现在 `output/` 目录中，文件名格式为 `原文件名_no_drums.mp3`。

## 高级用法

### 指定文件处理

```bash
docker compose run --rm drum-remover /data/input/song.mp3
```

### 调整输出比特率

```bash
docker compose run --rm drum-remover -b 192k
```

### 使用其他模型

```bash
docker compose run --rm drum-remover -m htdemucs_ft
```

可用模型：
| 模型 | 说明 |
|------|------|
| `htdemucs` | 默认，速度与质量平衡 |
| `htdemucs_ft` | 微调版，质量更高，速度较慢 |
| `htdemucs_6s` | 6 音轨分离（增加吉他、钢琴） |
| `mdx_extra` | MDX 架构 |

### 不使用 docker compose

```bash
docker build -t backtrack4drum .

docker run --rm --gpus 1 \
  -v $(pwd)/input:/data/input \
  -v $(pwd)/output:/data/output \
  backtrack4drum
```

## 在线部署

无需本地 GPU，直接在浏览器中使用。`app.py` + `requirements-spaces.txt` 适用于以下平台。

### Hugging Face Spaces

1. 创建新 [Space](https://huggingface.co/spaces)，选择 **Gradio** SDK。
2. 上传 `app.py` 和 `requirements-spaces.txt`（重命名为 `requirements.txt`）。
3. Space 自动构建并启动。

### ModelScope 创空间

1. 在 [ModelScope](https://modelscope.cn) 创建创空间，选择 **Gradio** SDK。
2. 克隆创空间仓库：
   ```bash
   git clone http://oauth2:<your_git_token>@www.modelscope.cn/studios/<用户名>/<空间名>.git
   ```
3. 将 `app.py` 和 `requirements-spaces.txt`（重命名为 `requirements.txt`）放入仓库，push 即可。

### 本地测试

```bash
pip install -r requirements-spaces.txt gradio
python app.py
```

浏览器打开 `http://localhost:7860`，上传音频文件测试。

## 支持的音频格式

MP3, WAV, FLAC, OGG, M4A, WMA, AAC

## 输出

- 格式：MP3（默认 320kbps）
- 文件名：`{原文件名}_no_drums.mp3`
