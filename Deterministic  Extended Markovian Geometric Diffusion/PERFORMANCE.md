# Otimização de desempenho do DEMGD

Estudo do custo computacional do método **Deterministic Extended Markovian
Geometric Diffusion** e uma reimplementação (`FastDEMGD.py`) focada em
velocidade e na qualidade da redução de dados. O código original
(`DeterministicEMGD.py`) foi mantido intacto como referência.

## 1. Gargalos identificados no código original

O `profiling` (via `cProfile`) e testes de escala revelaram dois regimes:

### 1.1 Construção do grafo KNN — `__createGraphKNN__`
Domina o tempo com os parâmetros padrão. Escala aproximadamente linear, mas com
uma constante enorme de Python puro:

- `__euclideanDistance__`: laço Python elemento a elemento usando `pow`.
- `__insertEdge__`: busca binária seguida de `list.insert`, que é **O(grau)**
  por causa do deslocamento da lista — e é chamada simetricamente.
- `__binarySearch__`: chamada centenas de milhares de vezes.

Além disso, o laço de KNN tem um **comportamento incorreto**: as listas
`knn`/`distances` são inicializadas uma vez por par de *buckets* (não por nó) e
compartilhadas entre todos os nós; as arestas finais acabam ligadas apenas ao
**último** nó do laço (`self.buckets[i][j]` com `j` no valor final). Ou seja, o
grafo de vizinhança construído não corresponde ao KNN pretendido.

### 1.2 Caminho de redução/deleção — `__EMGD__` → `__deleteInstance__`
É o gargalo **catastrófico O(N²)** quando a redução de fato acontece. Cada
instância removida dispara:

- `__indexCorrector__`: renumera **todas** as arestas de **todos** os vértices
  (O(E) por remoção);
- `__findDiagonal__`: reconstrói a diagonal inteira **após cada** deleção (O(E));
- `del self.set[index]`: deslocamento O(N) da lista a cada remoção.

Medição (remoção forçada de ~metade dos pontos numa passada):

| N | tempo (original) |
|---:|---:|
| 500 | 0,08 s |
| 1.000 | 0,29 s (×4,2) |
| 2.000 | 1,08 s (×4,1) |
| 4.000 | 4,70 s (×4,7) |

Dobrar N quadruplica o tempo — assinatura clara de complexidade quadrática.

### 1.3 Observação semântica
Com o `multiplier=1` padrão, a importância mínima (`P_ii · grau`) é sempre 1.0
e o limiar de remoção fica abaixo disso, de modo que **nada é removido** e o
laço encerra na primeira iteração. A redução só é exercitada com `multiplier`
menor.

## 2. Reimplementação — `FastDEMGD.py`

Mesma modelagem matemática (grafo KNN → *kernel* gaussiano `exp(-d²)` →
normalização em matriz de Markov `P = D⁻¹W` → importância `P_ii · grau` →
remoção dos pontos abaixo de `média − multiplier·desvio`, iterando), porém:

- **KNN exato** com KD-tree (`scipy.spatial.cKDTree`), O(N log N), corrigindo o
  bug do grafo de vizinhança;
- operador de difusão como **matriz esparsa** (`scipy.sparse`);
- cálculo de importância **totalmente vetorizado** com NumPy;
- redução por **recomputação barata** da difusão sobre o conjunto que encolhe a
  cada passada, eliminando a maquinaria O(N²) de deleção incremental
  (`__indexCorrector__`/`__findDiagonal__`);
- a supressão não-máxima (não remover clusters densos de candidatos de uma vez)
  é feita de forma **totalmente vetorizada**: remove-se, por passada, apenas os
  candidatos que são **mínimos locais de importância** no subgrafo de
  candidatos (desempate estrito por índice), sem laço Python.

É um **substituto direto**: mesma assinatura pública e mesmo retorno
`(reducedSet, removedSet)`. `maxPerBucket` e `propagation` são aceitos por
compatibilidade (não são mais necessários).

## 3. Resultados

Velocidade (mesma máquina; `benchmark.py` reproduz):

**Caminho padrão (construção do grafo):**

| N | original | fast | ganho |
|---:|---:|---:|---:|
| 1.000 | 0,066 s | 0,004 s | ~18× |
| 4.000 | 0,302 s | 0,008 s | ~39× |
| 16.000 | 1,639 s | 0,031 s | ~53× |

**Caminho de redução ativa (`multiplier=-2`):**

| N | original | fast | ganho |
|---:|---:|---:|---:|
| 500 | 0,085 s | 0,012 s | ~7× |
| 2.000 | 1,077 s | 0,018 s | ~60× |
| 4.000 | 4,918 s | 0,030 s | **~162×** |

O ganho **cresce com N** porque o original é O(N²) e a nova versão é ~O(N log N).

**Escala grande (apenas fast; o original é inviável nesse regime):**

| N | fast | mantidos | removidos |
|---:|---:|---:|---:|
| 50.000 | 0,23 s | 31.905 | 18.095 |
| 100.000 | 0,40 s | 69.864 | 30.136 |
| 200.000 | 0,85 s | 138.958 | 61.042 |

### 3.1 Onde o tempo é gasto agora (o piso prático)

Após vetorizar a supressão, em 200.000 pontos o tempo (~0,85 s) fica bem
distribuído: consulta KNN exata no KD-tree (~0,26 s, o núcleo irredutível),
montagem/simetrização das matrizes esparsas (~0,14 s), supressão vetorizada
(~0,12 s) e conversão final para lista (~0,06 s). Não há mais um único gargalo
dominante — estamos próximos do piso para uma implementação NumPy/SciPy.

Ganhos adicionais só viriam mudando a **natureza** do cálculo (KNN aproximado
por grade/`pykeops`, execução em GPU, ou compilação com Numba/Cython), com
retorno decrescente e mais complexidade/dependências.

**Qualidade da redução** (distância média/máxima de cada ponto original ao ponto
mantido mais próximo — menor = melhor cobertura): a média fica em ~0,07–0,16 em
dados na escala de dezenas de unidades, e a inspeção visual mostra que os pontos
removidos se concentram nos núcleos densos dos clusters, preservando estrutura,
contornos e *outliers* — o comportamento esperado de um método de redução.

## 4. Como usar

```python
# antes:
from DeterministicEMGD import DeterministicEMGD
# depois (substituto direto):
from FastDEMGD import DeterministicEMGD

reduced, removed = DeterministicEMGD(pontos, percentage=0.70, k=5, multiplier=0)
```

Requisitos: `numpy`, `scipy`. Reproduza os números com `python3 benchmark.py`.
