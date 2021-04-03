from DeterministicEMGD import DeterministicEMGD
from SetGenerator import generateSet2D
from Plot import plotList2D

Set = generateSet2D(1000)
ReducedSet, RemovedSet = DeterministicEMGD(Set, 0.80)
plotList2D(Set, ReducedSet, RemovedSet)
