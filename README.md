# BackTrack

从音乐中去除或提取指定乐器的通用工具。

使用 [Demucs](https://github.com/facebookresearch/demucs) (Meta AI) 音源分离技术，支持鼓、吉他、贝斯、人声、钢琴。去掉某个乐器跟着练，或者单独提取某个乐器来采样、扒谱。

**在线使用**：[https://huggingface.co/spaces/asherszhang/BackTrack4Drum](https://huggingface.co/spaces/asherszhang/BackTrack4Drum) — 无需安装，打开即用。

**ModelScope 创空间**：[https://www.modelscope.cn/studios/jiejing/BackTrack4Drum](https://www.modelscope.cn/studios/jiejing/BackTrack4Drum/summary) — 国内访问更快。


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

**去除乐器**（得到不含该乐器的伴奏）：

```bash
# 去除鼓（默认）
docker compose run --rm drum-remover

# 去除吉他
docker compose run --rm drum-remover -r guitar

# 去除贝斯
docker compose run --rm drum-remover -r bass

# 去除人声
docker compose run --rm drum-remover -r vocals

# 去除钢琴
docker compose run --rm drum-remover -r piano
```

**提取乐器**（只保留该乐器）：

```bash
# 提取人声
docker compose run --rm drum-remover -e vocals

# 提取鼓
docker compose run --rm drum-remover -e drums

# 提取吉他
docker compose run --rm drum-remover -e guitar
```

输出文件在 `output/` 目录：
- 去除模式：`{原文件名}_no_{乐器}.mp3`
- 提取模式：`{原文件名}_only_{乐器}.mp3`

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
| `htdemucs_6s` | 默认，6 音轨分离（drums, bass, vocals, guitar, piano, other） |
| `htdemucs` | 4 音轨分离（drums, bass, vocals, other） |
| `htdemucs_ft` | 微调版，质量更高，速度较慢 |
| `mdx_extra` | MDX 架构 |

### 不使用 docker compose

```bash
docker build -t backtrack .

docker run --rm --gpus 1 \
  -v $(pwd)/input:/data/input \
  -v $(pwd)/output:/data/output \
  backtrack -r guitar

docker run --rm --gpus 1 \
  -v $(pwd)/input:/data/input \
  -v $(pwd)/output:/data/output \
  backtrack -e vocals
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
- 去除模式：`{原文件名}_no_{乐器}.mp3`
- 提取模式：`{原文件名}_only_{乐器}.mp3`
