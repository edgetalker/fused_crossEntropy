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