### Forward
输入
+ `hidden`: $\text H \in \mathbb{R}^{B\times S\times D}$
+ `weight`: $\text W \in \mathbb{R}^{V\times D}$
+ `label`: $y \in \{0, \dots, V-1\}^{B\times S}$

将所有`token`展平: $N=B\times S$, 记`i=0,...,N-1`,对应`Hidden`向量$h_i \in \mathbb{R}^D$,`label`为$y$

1. `logits`
$$
z_{ij} = \sum_{d=0}^{D-1} h_{id}W_{jd} \quad j = 0, \dots, V-1
$$
2. `logsumexp`
$$
m_i = \max_{j}z_{ij}, \quad l_i =  -z_{i,y_i} + m_i + \log \sum_{j=0}^{V-1}e^{z_{ij}-m_i}
$$
3. `loss`
$$
L = \frac{1}{N}\sum_{0}^{N-1}l_i 
$$

### Backward
已知上游梯度为$\frac{\partial L}{\partial L}=1$, 我们需要计算$\frac{\partial L}{\partial H}$和$\frac{\partial L}{\partial W}$
1. 对`logits`的梯度
$$
\frac{\partial L}{\partial z_{ij}} = \frac{1}{N}(p_{ij}-\delta_{j, y_i})
$$
2. 对`hidden`的梯度
$$
\frac{\partial L}{\partial h_{id}} = \sum_{j=0}^{V-1}\frac{\partial L}{\partial z_{ij}}W_{jd}=\frac{1}{N}(p_i-e_{y_i})^\text TW
$$
3. 对`weight`的梯度
$$
\frac{\partial L}{\partial W_{jd}} = \sum_{i=0}^{N-1}\frac{\partial L}{\partial z_{ij}}h_{id}=\frac{1}{N}\sum^{N-1}_{i=0}(p_i-e_{y_i})h_i^\text T
$$

### Benchmark
```
===== B=1, S=1024, D=768, V=32000 =====
  Standard peak memory : 363.78 MB
  Manual   peak memory : 491.80 MB
  Fused    peak memory : 113.79 MB
  Reduction vs standard : 3.2x
  Reduction vs manual   : 4.3x

===== B=2, S=1024, D=768, V=32000 =====
  Standard peak memory : 616.04 MB
  Manual   peak memory : 866.07 MB
  Fused    peak memory : 116.06 MB
  Reduction vs standard : 5.3x
  Reduction vs manual   : 7.5x

===== B=4, S=1024, D=768, V=32000 =====
  Standard peak memory : 1122.05 MB
  Manual   peak memory : 1622.13 MB
  Fused    peak memory : 122.10 MB
  Reduction vs standard : 9.2x
  Reduction vs manual   : 13.3x

===== B=1, S=1024, D=4096, V=32000 =====
  Standard peak memory : 784.28 MB
  Manual   peak memory : 910.30 MB
  Fused    peak memory : 532.29 MB
  Reduction vs standard : 1.5x
  Reduction vs manual   : 1.7x

===== B=2, S=1024, D=4096, V=32000 =====
  Standard peak memory : 1048.29 MB
  Manual   peak memory : 1298.32 MB
  Fused    peak memory : 548.31 MB
  Reduction vs standard : 1.9x
  Reduction vs manual   : 2.4x

===== B=4, S=1024, D=4096, V=32000 =====
  Standard peak memory : 1580.30 MB
  Manual   peak memory : 2080.38 MB
  Fused    peak memory : 580.35 MB
  Reduction vs standard : 2.7x
  Reduction vs manual   : 3.6x

===== B=1, S=1024, D=8192, V=32000 =====
  Standard peak memory : 1298.28 MB
  Manual   peak memory : 1426.30 MB
  Fused    peak memory : 1048.29 MB
  Reduction vs standard : 1.2x
  Reduction vs manual   : 1.4x

===== B=2, S=1024, D=8192, V=32000 =====
  Standard peak memory : 1580.29 MB
  Manual   peak memory : 1830.32 MB
  Fused    peak memory : 1080.31 MB
  Reduction vs standard : 1.5x
  Reduction vs manual   : 1.7x
```