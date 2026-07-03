# DEMGD como contribuição científica — estudo e veredito honesto

Este documento reúne um estudo a fundo do método **Deterministic Extended
Markovian Geometric Diffusion (DEMGD)** com o objetivo de responder: *dá para
transformá-lo numa boa contribuição científica?* Combina (1) uma análise
matemática do escore, (2) experimentos controlados contra baselines fortes e
(3) posicionamento na literatura. A conclusão é honesta, inclusive onde é
desfavorável.

> TL;DR — O escore de importância da DEMGD é **rank-idêntico ao KDE inverso**
> (densidade local invertida): a roupagem de "difusão markoviana" não acrescenta
> nada ao escore de 1 passo. O método é, na prática, um **subamostrador
> adaptativo à densidade, determinístico e quase-linear**. Ele não é o melhor em
> nenhuma métrica de qualidade isolada (FPS/k-means empatam ou superam), mas é
> **muito mais barato que k-means** e **mais robusto que FPS** para classificação
> em redução agressiva. Isso sustenta uma contribuição **empírica/de engenharia
> honesta**, não um método estado-da-arte. A alegação de novidade por "difusão"
> não sobrevive à revisão sem uma reformulação (ver §5).

---

## 1. O que o método realmente calcula (análise)

Importância no código: `imp_i = P_ii · grau_i`, onde `P = D⁻¹W` é a matriz de
Markov de um grafo k-NN com kernel `w = exp(-d²)` e auto-laço (`w_ii = 1`).

- Com kernel gaussiano, `w_ii = 1`, logo `P_ii = 1/rowsum_i` (rowsum = grau
  **ponderado**). Se `grau_i` fosse o grau ponderado, então
  **`P_ii · grau_i = 1` para todo ponto** — escore constante, degenerado.
- No código, `grau_i` é a **contagem não-ponderada** de vizinhos
  (`np.diff(indptr)`), o que salva da degeneração. Mas então:

  `imp_i = (nº de vizinhos) / (Σ_j exp(-d_ij²)) ≈ 1 / (afinidade média) = KDE⁻¹`.

**Verificação empírica** (`analysis_core.py`): correlação de Spearman entre a
importância da DEMGD e o KDE inverso (`1/afinidade média`) = **+1.000** em três
datasets (gaussiano, blobs, swiss roll). Ou seja, como *ranking*, o escore é
**exatamente densidade local inversa**.

Implicação: o escore de 1 passo **não é uma quantidade de difusão informativa** —
é um estimador de densidade. A probabilidade de retorno em *t* passos
`(Pᵗ)_ii` (t ≥ 3) **descorrelaciona** do KDE inverso (Spearman cai para
−0.18 … +0.33), e essa sim é a quantidade de difusão real (= *Auto-Diffusion
Function*/HKS). Mas ver §4: usada de forma ingênua num grafo k-NN, ela é um
detector de features fraco.

## 2. Evidência experimental (vs baselines fortes)

Baselines: aleatório, FPS (farthest-point), voxel-grid, k-means (coreset).
Scripts em `research/`. Resumo:

| Experimento | Resultado da DEMGD | Leitura honesta |
|---|---|---|
| **Classificação a jusante** (digits/breast-cancer/wine, 1-NN) | Compete com k-means; **supera FPS e aleatório em redução agressiva** (FPS colapsa: 0.69 em digits@5%) | Melhor exibição da DEMGD |
| **Equalização de densidade** (CV do kNN em nuvem gaussiana) | DEMGD 0.575 ≈ FPS 0.559; melhor que voxel/k-means/aleatório | Empata com FPS |
| **Preservação de fronteira** (disco denso) | Ganho marginal (rim boost 1.07) | Sem vantagem clara |
| **Detecção de features em densidade uniforme** (cubo) | AUC 0.54 ≈ acaso; `(Pᵗ)_ii` no máx 0.60 | **Falha** — é método de densidade |
| **Manifold learning** (swiss roll enviesado) | DEMGD 0.937 < FPS 1.000 = voxel 1.000 | Perde para FPS/voxel |
| **Custo/escala** (keep 20%) | **82× mais rápido que k-means** a N=40k; mais rápido que FPS a N alto | **Vitória mais clara** |

Padrão geral: a DEMGD **não vence em qualidade** de forma consistente, mas tem o
melhor **custo-benefício** (quase-linear, determinística, sem treino) e é
**robusta onde o FPS falha** (classificação em redução agressiva, porque o FPS
escolhe outliers).

## 3. Posicionamento na literatura (o terreno está lotado)

- **Seleção de instâncias** é um campo de 50 anos com protocolo maduro (García et
  al., *IEEE TPAMI* 2012). Seleção **baseada em densidade** já existe
  (Carbonera & Abel, LDIS 2015; Global-Density IS 2021; DBI, *PLOS ONE* 2024).
- **Redução de grafos para seleção de instâncias** (*Artificial Intelligence
  Review*, 2024) já compara 35 técnicas incluindo escores de passeio
  aleatório/difusão (**PageRank**, **resistência efetiva**). É o competidor mais
  próximo.
- **Coresets** com garantia (leverage/ridge-leverage, sensitivity sampling;
  Spielman-Srivastava; Loukas coarsening) dominam o lado teórico da amostragem.
- Em **nuvens de pontos**: FPS (Eldar 1997; PointNet++ 2017), voxel (PCL/Open3D),
  Poisson-disk (Corsini 2012), curvatura (Pauly 2002), e aprendidos
  (SampleNet 2020; APES 2023; GP-PCS 2024). O concorrente direto de mesmo
  "encaixe" é **Chen et al., *Fast Resampling of 3D Point Clouds via Graphs*,
  IEEE TSP 2018** (grafo → escore por ponto → preserva features).
- **A quantidade de difusão de retorno** já tem nome desde 2009: *Auto-Diffusion
  Function* (Gębal 2009) / *Heat Kernel Signature* (Sun-Ovsjanikov-Guibas 2009),
  usada como **descritor/detector de features** — não para decimação.

**Veredito de novidade:** usar centralidade de difusão/passeio para seleção
**não é novo**; a *Auto-Diffusion Function* **não é nova**. O único ponto
inédito estreito seria "usar a probabilidade de retorno de difusão num grafo
k-NN como critério de *decimação*, sem autovalores" — mas o escore de 1 passo da
DEMGD nem é isso (é KDE inverso). Sem reformulação, **não há novidade defensável**.

## 4. Por que o "diffusion/feature-preserving" não se sustenta como está

- O escore de 1 passo = densidade inversa (§1) ⇒ não vê features quando features
  ≠ baixa densidade.
- No teste do cubo (densidade uniforme, features = arestas/quinas), nenhum escore
  de difusão passa de AUC ~0.60. A "preservação de features" herdada da origem
  (simplificação de malhas) só ocorre porque **fronteiras têm baixa densidade** —
  é um efeito de densidade, não de curvatura.

## 5. Caminhos viáveis para uma contribuição (com honestidade sobre cada um)

Ordenados por (probabilidade de sucesso × honestidade), não por ambição.

1. **Contribuição empírica/de engenharia (recomendada, baixo risco).**
   Enquadrar como *"um subamostrador adaptativo à densidade, determinístico e
   quase-linear"*: a reformulação O(N²)→O(N log N) (100×+), a decimação
   iterativa por supressão de mínimos locais (peeling que gera padrão
   blue-noise) e um **benchmark honesto** vs FPS/voxel/aleatório/LDIS/k-means
   mostrando qualidade competitiva a **fração do custo do k-means** e robustez
   onde o FPS falha. Protocolo estatístico do campo (KEEL ≥40 datasets,
   Friedman + Wilcoxon + Nemenyi). Venue realista: workshop/periódico aplicado.
   **Não** alegar novidade de difusão.

2. **Paper de análise/reabilitação (baixo risco, honesto).**
   "Analisamos a DEMGD, provamos que o escore = KDE inverso, corrigimos a
   complexidade e caracterizamos *quando* decimação por densidade ajuda/atrapalha
   vs FPS/k-means/coresets." Papers de análise honesta têm valor.

3. **Enquadramento teórico: equalização de densidade → Laplace-Beltrami
   (médio risco, alto retorno).** Coifman-Lafon mostram que a densidade enviesa o
   operador de difusão e que a normalização α=1 a desacopla. Provar que
   **remover iterativamente os máximos de densidade leva a densidade empírica ao
   uniforme**, i.e., a DEMGD alcança *por poda* o que α=1 alcança *por
   reponderação*, com o Laplaciano resultante convergindo ao Laplace-Beltrami.
   Contribuição: "subamostragem como normalização de densidade para manifold
   learning". (Nos meus testes a DEMGD perde para FPS nessa tarefa — então o
   teorema precisaria vir com reponderação dos pontos mantidos.)

4. **Novo escore genuinamente não-densidade (alto risco, pesquisa aberta).**
   Combinar densidade com geometria local (PCA local/curvatura) ou usar
   `(Pᵗ)_ii` com normalização adequada. Meus pivôs sugerem que upgrades ingênuos
   de difusão **não** entregam ganho claro; se acabar precisando de curvatura,
   recai em Pauly 2002. Sem payoff garantido.

## 6. Recomendação

O enredo "método esquecido, agora melhorado" é melhor servido por **(1)+(2)**:
honesto, alcançável e real. O conteúdo científico verdadeiro que você já tem é
(a) a **reformulação eficiente** (grande, mensurável) e (b) uma **caracterização
empírica honesta** de um subamostrador por densidade. Alegar novidade de
"difusão markoviana" para o escore atual **não passaria** por um revisor que
calcula `imp = KDE⁻¹` em uma linha. Se quiser mirar mais alto, o caminho (3)
(equalização → Laplace-Beltrami, com reponderação) é o único com teoria real —
mas exige trabalho matemático e ainda assim compete com FPS.

---

### Citações-âncora
- Coifman & Lafon, *Diffusion maps*, ACHA 21(1):5–30, 2006.
- García, Derrac, Cano, Herrera, *Prototype Selection...*, IEEE TPAMI 34(3), 2012.
- *Graph reduction techniques for instance selection*, Artif. Intell. Rev., 2024.
- Gębal et al., *Shape Analysis Using the Auto Diffusion Function*, SGP/CGF 2009.
- Sun, Ovsjanikov, Guibas, *HKS*, CGF (SGP) 2009.
- Chen, Tian, Feng, Vetro, Kovačević, *Fast Resampling of 3D Point Clouds via
  Graphs*, IEEE TSP 66(3):666–681, 2018.
- Carbonera & Abel, *LDIS* 2015 / Global-Density IS 2021.
- Eldar et al., *Farthest Point Strategy*, IEEE TIP 1997; Pauly et al., *Efficient
  Simplification of Point-Sampled Surfaces*, IEEE Vis 2002.

*Reprodução: `research/analysis_core.py` (escore = KDE⁻¹), `exp1_downstream.py`,
`exp2_geometry.py`, `pivot_features.py`, `pivot_manifold.py`.*
