import cv2
import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms
from ultralytics import YOLO
# Universidade Federal do Maranhão
# Programa de Pós-Graduação em Ciência da Computação – UFMA
# Curso de Ciência da Computação - UFMA
# Disciplina de Sistemas de Visão Computacional
# Professor Dr. Geraldo Braz Júnior
# Disciplina de Sistemas de Visão Computacional
# Projeto: Detecção de Emoções - Fase 2
# Equipe 2: Arnaud, Guilherme, Osevaldo, Wesley e Yure
# *****************************************************
# Critérios:
# Reconhecimento da pessoa ou da expressão facial da pessoa numa imagem
# Continuar a Fase 1: Determinar a expressão facial da pessoa (precisa que os rostos tenham sido detectados)
# Etapas:
# Otimizar a arquitetura usando Optuna (Ciclo Network Archtecture Search - NAS)
# XXX:
# Registrar os logs de execução no weight and bias - https://wandb.ai/site/
# Criar aplicação com os modelos, que seja capaz de usar uma webcam e capturar todas as faces num ambiente. 
# Após isso, indicar no boundingbox a expressão facial
# Identificar o rosto: Para tanto, avalie a utilização da Yolo 
# [Opcional] Adicionar explicabilidade (GradCam)

# Caminhos dos modelos.
CAMINHO_MODELO_EMOCOES = 'modelo-optuna/modelo-eq-2-svc-fase-2-40-ep-reg-acc-6678.pth'
# Modelo Yolo finetunado com datasets Face-Detection-Dataset para rostos mais 
# centralizados e Wider Face para rostos com mais 
# variância
CAMINHO_MODELO_YOLO    = 'modelo-yolo/best.pt'

# sobrescrevendo os nomes originais das emoções para português
EMOCOES = ['raiva', 'desdenho', 'nojo', 'medo', 'feliz', 'neutro', 'triste', 'surpresa']

# Setup da de verificação da GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Dispositivo: {}'.format(device))

checkpoint = torch.load(CAMINHO_MODELO_EMOCOES, map_location=device, weights_only=False)
# Extrai a acurácia salva no checkpoint para verificação da métrica que o modelo alcançou
acc_validacao = checkpoint.get('acuracia_validacao', None)
acuracia = 'Acurácia não encontrada no checkpoint'
if acc_validacao is not None:
    acuracia = '{:.2f}%'.format(acc_validacao * 100)
#.

# =========================================================================================
# Tela de apresentação do protótipo
# Para reconhecimento de emoções faciais em tempo real via webcam.
# Utiliza YOLO para detecção de rostos e uma CNN treinada para classificação de emoções.
# =========================================================================================
print('=' * 60)
print('  UNIVERSIDADE FEDERAL DO MARANHÃO - UFMA')
print('  Sistemas de Visão Computacional')
print('  Prof. Dr. Geraldo Braz Júnior')
print('=' * 60)
print('  Projeto: Detecção de Emoções Faciais - Fase 2')
print('  Equipe 2: Arnaud, Guilherme, Osevaldo, Wesley e Yure')
print('-' * 60)
print('  Modelo de emoções: CNN otimizada com Optuna (40 épocas)')
print('  Detector de rostos: YOLOv8 finetunado')
print('  Acurácia de validação: {}'.format(acuracia))
print('-' * 60)
print('  Instruções:')
print('    - Posicione seu rosto na frente da webcam')
print('    - A emoção detectada aparece sobre o rosto')
print('    - Pressione Q para encerrar')
print('=' * 60)
print()
#.


# Carregamento do modelo de emoções.
modelo_emocoes = checkpoint['arquitetura']
modelo_emocoes.load_state_dict(checkpoint['pesos'])
modelo_emocoes.to(device)
modelo_emocoes.eval()

print('O modelo de emoções {} foi carregado com sucesso.'.format(CAMINHO_MODELO_EMOCOES))

# Carregamento do YOLO.
modelo_yolo = YOLO(CAMINHO_MODELO_YOLO)
print('Modelo YOLO {} carregado com sucesso.'.format(CAMINHO_MODELO_YOLO))

# Transformações aplicadas a cada rosto detectado antes de passar pelo modelo de emoções.
# Descobri que precisa converter pra escala de cinza e redimensionar pra 48x48
# porque o modelo foi treinado com imagens nesse formato.
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# TODO: tratar exceção se não tiver webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print('ERRO: Não foi possível abrir a webcam.')
    print('Verifique se a câmera está conectada e não está sendo usada por outro programa.')
    exit(1)

print('Webcam iniciada. Pressione Q para encerrar.')

# Cria a janela uma única vez antes do loop pra evitar problemas de múltiplas janelas
cv2.namedWindow('Projeto Final SCV - Equipe 2 - Fase 2 - Detecção de Emoções', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Projeto Final SCV - Equipe 2 - Fase 2 - Detecção de Emoções', 960, 720)

# HACK: Suavização temporal (temporal smoothing)
# Problema: o YOLO às vezes falha em detectar o rosto em alguns frames isolados,
# fazendo o retângulo "piscar" (flickering). Para resolver isso, guardo a última
# detecção e mantenho ela na tela por alguns frames mesmo se o YOLO não detectar nada.
# Explica o conceito de "window" e "threshold" pra manter detecções consistentes entre frames
# https://stackoverflow.com/questions/61810241/object-detection-consistency-when-working-with-videos-frame-by-frame
# Basicamente guarda os últimos rostos detectados e uma contagem de quantos frames
# cada detecção pode sobreviver sem ser re-detectada.
deteccoes_anteriores = []
# quantos frames a detecção fica antes de sumir
FRAMES_PERSISTENCIA = 8  

while True:
    ret, frame = cap.read()
    if not ret:
        break

    resultados = modelo_yolo(frame, verbose=False, conf=0.25)

    deteccoes_atuais = []

    for resultado in resultados:
        for box in resultado.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            rosto = frame[y1:y2, x1:x2]
            if rosto.size == 0:
                continue

            # unsqueeze(0) adiciona a dimensão de batch que o modelo espera
            # (descobri isso no erro: expected 4D input got 3D)
            entrada = transform(rosto).unsqueeze(0).to(device)

            # torch.no_grad() desativa o cálculo de gradientes porque aqui é só inferência
            with torch.no_grad():
                saida = modelo_emocoes(entrada)
                emocao_idx = saida.argmax(dim=1).item()
                emocao = EMOCOES[emocao_idx]
                # softmax converte a saída em probabilidades de 0 a 1
                confianca = torch.softmax(saida, dim=1)[0][emocao_idx].item()

            # DEGBUG: descomentar pra ver os valores no terminal
            # print('emocao_idx: {}, emocao: {}, confianca: {:.2f}'.format(emocao_idx, emocao, confianca))

            deteccoes_atuais.append({
                'caixa': (x1, y1, x2, y2),
                'emocao': emocao,
                'confianca': confianca,
                'frames_restantes': FRAMES_PERSISTENCIA
            })

    # Suavização temporal: se não detectou nada agora, mantém as detecções
    # anteriores por mais alguns frames pra evitar o flickering.
    deteccoes_para_exibir = deteccoes_atuais.copy()

    if len(deteccoes_atuais) == 0 and len(deteccoes_anteriores) > 0:
        for deteccao in deteccoes_anteriores:
            deteccao['frames_restantes'] -= 1
            if deteccao['frames_restantes'] > 0:
                deteccoes_para_exibir.append(deteccao)

    deteccoes_anteriores = deteccoes_para_exibir

    # Desenha os retângulos e o texto na imagem
    for deteccao in deteccoes_para_exibir:
        x1, y1, x2, y2 = deteccao['caixa']
        emocao = deteccao['emocao']
        confianca = deteccao['confianca']

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        texto = '{} {:.0f}%'.format(emocao, confianca * 100)
        cv2.putText(frame, texto, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow('Projeto Final SCV - Equipe 2 - Fase 2 - Detecção de Emoções', frame)

    if cv2.waitKey(1) & 0xFF in (ord('q'), ord('Q')):
        break

cap.release()
cv2.destroyAllWindows()
print('Encerrado.')
