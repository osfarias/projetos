# Detecção de Emoções Faciais em Tempo Real

Protótipo desenvolvido para a disciplina de **Sistemas de Visão Computacional** (UFMA), sob orientação do Prof. Dr. Geraldo Braz Júnior.

O sistema captura vídeo da webcam, detecta rostos usando YOLOv8 finetunado e classifica a emoção facial de cada pessoa em tempo real.

## Demo

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.12-red)
![YOLOv8](https://img.shields.io/badge/YOLOv8-ultralytics-green)

## Emoções Reconhecidas

| Emoção | Emoção | Emoção | Emoção |
|--------|--------|--------|--------|
| <i class="fa-solid fa-face-angry"></i> Raiva | <i class="fa-solid fa-face-meh"></i> Desdenho | <i class="fa-solid fa-face-dizzy"></i> Nojo | <i class="fa-solid fa-face-flushed"></i> Medo |
| <i class="fa-solid fa-face-smile"></i> Feliz | <i class="fa-solid fa-face-meh-blank"></i> Neutro | <i class="fa-solid fa-face-sad-tear"></i> Triste | <i class="fa-solid fa-face-surprise"></i> Surpresa |

## Arquitetura

```
Webcam > YOLOv8 (detecção de rostos) > CNN (classificação de emoção) > Exibição
```

- **Detector de rostos:** YOLOv8 finetunado com Face-Detection-Dataset e Wider Face
- **Classificador de emoções:** CNN otimizada com Optuna (40 épocas)
- **Acurácia de validação:** ~66.78%

## Como Rodar

### 1. Clone o repositório

```bash
git clone https://github.com/SEU-USUARIO/SEU-REPO.git
cd fase-2
```

### 2. Crie um ambiente virtual

```bash
python -m venv .venv
source .venv/bin/activate        # Linux/Mac
# .venv\Scripts\activate         # Windows
```

### 3. Instale as dependências

```bash
pip install -r requirements.txt
```

> Se tiver GPU NVIDIA com CUDA:
> ```bash
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
> ```

### 4. Execute

```bash
python reconhecimento_emocoes_webcam.py
```

Pressione **Q** para encerrar.

## Estrutura do Projeto

```
fase-2/
├── reconhecimento_emocoes_webcam.py   # Script principal (protótipo)
├── modelo-optuna/
│   └── modelo-eq-2-svc-fase-2-40-ep-reg-acc-6678.pth
├── modelo-yolo/
│   └── best.pt
├── requirements.txt
├── read-me.txt
└── README.md
```

## Requisitos

- Python 3.10+
- Webcam funcional
- GPU com CUDA (opcional, melhora performance)

## Tecnologias

- [PyTorch](https://pytorch.org/) - framework de deep learning
- [Ultralytics YOLOv8](https://docs.ultralytics.com/) - detecção de objetos
- [OpenCV](https://opencv.org/) - captura de vídeo e exibição
- [Optuna](https://optuna.org/) - otimização de hiperparâmetros (treinamento)

## Equipe 2

- Arnaud
- Guilherme
- Osevaldo
- Wesley
- Yure

## Disciplina

Sistemas de Visão Computacional - UFMA  
Prof. Dr. Geraldo Braz Júnior
