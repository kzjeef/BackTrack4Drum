# Drum Remover

使用 [Demucs](https://github.com/facebookresearch/demucs) (Meta AI) 从音乐中去除架子鼓，基于 GPU 加速的 Docker 容器。

Demucs 将音乐分离为 4 个音轨（人声、鼓、贝斯、其他乐器），本工具将除鼓以外的音轨合并输出。

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
docker build -t drum-remover .

docker run --rm --gpus 1 \
  -v $(pwd)/input:/data/input \
  -v $(pwd)/output:/data/output \
  drum-remover
```

## 支持的音频格式

MP3, WAV, FLAC, OGG, M4A, WMA, AAC

## 输出

- 格式：MP3（默认 320kbps）
- 文件名：`{原文件名}_no_drums.mp3`
