# NInfer - Dual RTX 3060 (24GB VRAM) TP=2 고성능 추론 엔진

이 프로젝트는 **NVIDIA GeForce RTX 3060 12GB x 2장 (총 24GB VRAM, Ampere sm_86)** 환경에서 **Qwen3.8-27B (18GB 가중치)** 및 대형 모델을 **Tensor Parallelism (TP=2)** 과 **MTP 추측 디코딩(Speculative Decoding)** 을 통해 고속으로 구동할 수 있도록 최적화된 C++/CUDA 기반 추론 엔진입니다.

---

## 🚀 주요 기능 및 최적화

1. **Dual GPU Tensor Parallelism (TP=2)**
   - 두 장의 RTX 3060 12GB에 18GB 크기의 가중치를 각각 약 **9.44 GiB씩 1:1 완벽 분할** 탑재
   - CUDA Graph 및 P2P/NCCL 동기화를 통한 통신 오버헤드 최소화
2. **Ampere (sm_86) 아키텍처 완전 호환**
   - Blackwell 전용 명령어(TMA, W4A4, PDL)를 Ampere 환경에 맞게 대체 경로 및 동적 공유 메모리(Dynamic Shared Memory)로 최적화
3. **대용량 컨텍스트 지원 (최대 112K)**
   - INT8 Paged KV Cache 적용으로 12GB GPU 환경에서도 **최대 114,688 토큰 (약 112K Context)** 수용 가능
4. **MTP (Multi-Token Prediction) 추측 디코딩**
   - MTP 3-Draft 토큰 가속을 적용하여 단일 디코드 단계에서 다중 토큰을 병렬 생성
5. **표준 API 서빙 지원**
   - OpenAI 호환 API (/v1/chat/completions) 및 Anthropic Messages API 표준 제공

---

## 📊 실측 벤치마크 성능 (RTX 3060 12GB x 2)

- **모델**: Qwen3.8-27B (qwen3_8_27b.ninfer)
- **설정**: TP=2, INT8 KV Cache, MTP 3-Draft

| 동시 요청 수 (Concurrency) | 총 출력 토큰 | 총 소요 시간 | 총 처리량 (Throughput) | 요청당 생성 속도 |
| :---: | :---: | :---: | :---: | :---: |
| **1개 (단일 스트림)** | 128 tokens | 5.43 s | **23.55 tokens/s** | **23.58 tok/s** |
| **2개 (동시 요청)** | 256 tokens | 12.80 s | **20.00 tokens/s** | **16.66 tok/s** |
| **4개 (동시 요청)** | 512 tokens | 26.60 s | **19.25 tokens/s** | **9.54 tok/s** |

- **GPU VRAM 상태**: 
  - GPU 0: **11,627 MiB / 12,288 MiB** (94.6% 점유, 661 MiB 안정 여유)
  - GPU 1: **11,627 MiB / 12,288 MiB** (94.6% 점유, 661 MiB 안정 여유)

---

## 🛠️ 시작하기 (Quick Start)

### 1. Docker Compose로 실행

`ash
# 서비스 백그라운드 시작
docker compose up -d

# 실시간 로그 확인
docker compose logs -f

# 서비스 중지
docker compose down
`

### 2. docker-compose.yml 설정 예시

`yaml
services:
  ninfer-serve:
    image: ninfer:tp2-3060x2
    container_name: ninfer-tp2-srv
    restart: unless-stopped
    ports:
      - "8080:8080"
    volumes:
      - /data/ninfer-3060X2/models:/workspace/models:ro
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
    command: >
      ninfer-serve /workspace/models/qwen3_8_27b.ninfer
      --host 0.0.0.0
      --port 8080
      --tp 2
      --devices 0,1
      --max-context 114688
      --kv-capacity 114688
      --kv-dtype int8
      --spec mtp
      --draft-tokens 3
      --lm-head-draft
`

---

## 📡 API 호출 예시

### Python

`python
import urllib.request
import json

url = "http://localhost:8080/v1/chat/completions"
headers = {"Content-Type": "application/json"}
data = {
    "model": "qwen3.8-27b",
    "messages": [
        {"role": "user", "content": "안녕하세요! 간단한 자기소개를 부탁합니다."}
    ],
    "max_tokens": 200,
    "temperature": 0.7
}

req = urllib.request.Request(url, headers=headers, data=json.dumps(data).encode("utf-8"))
with urllib.request.urlopen(req) as resp:
    result = json.loads(resp.read().decode("utf-8"))
    print(result["choices"][0]["message"]["content"])
`

### cURL

`ash
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3.8-27b",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 100
  }'
`

---

## ⚙️ 주요 런타임 옵션 안내

- --tp 2: 2-GPU 텐서 병렬화 활성화
- --devices 0,1: 사용할 GPU 디바이스 번호 지정
- --max-context 114688: 최대 단일 컨텍스트 길이 (112K)
- --kv-capacity 114688: 전체 KV 캐시 풀 용량
- --kv-dtype int8: KV 캐시 8비트 정수 양자화 (VRAM 절약)
- --spec mtp --draft-tokens 3: MTP 추측 디코딩 3토큰 드래프트 가속
- --lm-head-draft: 드래프트 헤드에 메인 LM 헤드 가중치 재사용
- --max-concurrency 1~8: 동시 활성 디코드 배치 레인 수 지정 (기본값: 1)
