Table 1: MNIST Results (MLPs)
================================================================================
Architecture         LR         Batch    Opt      Dropout    Test Acc     Runtime
----------------------------------------------------------------------------------------
MLP (shallow)        0.001      64       Adam     0.2        97.85%         78.69s
MLP (medium)         0.001      64       Adam     0.2        98.15%         86.48s
MLP (deep)           0.001      64       Adam     0.2        97.81%         92.54s

Table 2: CIFAR-10 Results (MLPs)
Architecture        LR        Batch  Opt    Dropout  Val Acc (best)  Top 3 Configs
MLP (shallow)       0.001     64     Adam   0.2      49.84%         49.84%, 49.42%, 48.90%
MLP (medium)        0.0001    32     Adam   0.2      52.22%         52.22%, 49.16%, 48.26%
MLP (deep)          0.0001    32     Adam   0.2      48.80%         48.80%, 47.06%, 44.70%

Table: MNIST Results (CNNs)
Architecture         LR         Batch    Weight Decay    Test Acc     Runtime
-------------------------------------------------------------------------------------
CNN (simple)         0.01       64       0               98.38%         22.04s
CNN (enhanced)       0.01       128      0               98.80%         15.08s

================================================================================

================================================================================
Table: CIFAR-10 Results (CNNs)
================================================================================
Architecture         LR         Batch    Weight Decay    Test Acc     Runtime
-------------------------------------------------------------------------------------
CNN (simple)         0.001      64       0.005           61.85%         18.60s
CNN (enhanced)       0.001      32       0               73.96%         41.49s


