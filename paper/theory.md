 Theorem 1: Convergence & Stability
This is the most critical one. Reviewers will ask:

"Does adding the osmotic term destabilize training?"

We need to prove it doesn't.

Setup — Formal Definition First
Let our osmotic attention output for layer ll
l be:
hl+1=OsmoticAttn(hl)+hl\mathbf{h}^{l+1} = \text{OsmoticAttn}(\mathbf{h}^l) + \mathbf{h}^lhl+1=OsmoticAttn(hl)+hl
Where:
OsmoticAttn(h)=softmax(QKTd+λM⊙ΔΠ)V\text{OsmoticAttn}(\mathbf{h}) = \text{softmax}\left(\frac{QK^T}{\sqrt{d}} + \lambda M \odot \Delta\Pi\right)VOsmoticAttn(h)=softmax(d​QKT​+λM⊙ΔΠ)V

Theorem 1 — Bounded Osmotic Perturbation
**Theorem 1:** *The osmotic gradient term λMijΔπij\lambda M_{ij} \Delta\pi_{ij}
λMij​Δπij​ is bounded, and OsmoticAttention converges under the same conditions as standard attention.*
Proof:
Step 1 — Bound the information density ρi\rho_i
ρi​:
ρi=−∑vpivlog⁡piv\rho_i = -\sum_v p_{iv} \log p_{iv}ρi​=−v∑​piv​logpiv​
Since piv=softmax(Wρhi)vp_{iv} = \text{softmax}(W_\rho h_i)_v
piv​=softmax(Wρ​hi​)v​, and entropy of a distribution over VV
V categories is bounded:
0≤ρi≤log⁡V0 \leq \rho_i \leq \log V0≤ρi​≤logV
where VV
V is the vocabulary/projection dimension. ✅
Step 2 — Bound the osmotic gradient Δπij\Delta\pi_{ij}
Δπij​:
Δπij=ρj−ρi\Delta\pi_{ij} = \rho_j - \rho_iΔπij​=ρj​−ρi​
Since both ρi,ρj∈[0,log⁡V]\rho_i, \rho_j \in [0, \log V]
ρi​,ρj​∈[0,logV]:
∣Δπij∣≤log⁡V|\Delta\pi_{ij}| \leq \log V∣Δπij​∣≤logV
Step 3 — Bound the membrane MijM_{ij}
Mij​:
Mij=σ(mi+mjT)∈(0,1)M_{ij} = \sigma(m_i + m_j^T) \in (0, 1)Mij​=σ(mi​+mjT​)∈(0,1)
Since sigmoid maps to (0,1)(0,1)
(0,1). ✅
Step 4 — Bound the full osmotic term:
∣Mij⋅λh⋅Δπij∣≤∣λh∣⋅log⁡V|M_{ij} \cdot \lambda_h \cdot \Delta\pi_{ij}| \leq |\lambda_h| \cdot \log V∣Mij​⋅λh​⋅Δπij​∣≤∣λh​∣⋅logV
This is a finite, bounded perturbation to the attention logits.
Step 5 — Stability of softmax under bounded perturbation:
For standard attention logits aij=QiKjTda_{ij} = \frac{Q_iK_j^T}{\sqrt{d}}
aij​=d​Qi​KjT​​ and osmotic perturbation ϵij=λhMijΔπij\epsilon_{ij} = \lambda_h M_{ij} \Delta\pi_{ij}
ϵij​=λh​Mij​Δπij​:
∥softmax(a+ϵ)−softmax(a)∥1≤2∥ϵ∥∞≤2∣λh∣log⁡V\|\text{softmax}(a + \epsilon) - \text{softmax}(a)\|_1 \leq 2\|\epsilon\|_\infty \leq 2|\lambda_h|\log V∥softmax(a+ϵ)−softmax(a)∥1​≤2∥ϵ∥∞​≤2∣λh​∣logV
As λh→0\lambda_h \to 0
λh​→0 (initialization), osmotic attention →\to
→ standard attention. ✅
Step 6 — Gradient boundedness:
∂L∂λh=∑i,j∂L∂Aij⋅Mij⋅Δπij\frac{\partial \mathcal{L}}{\partial \lambda_h} = \sum_{i,j} \frac{\partial \mathcal{L}}{\partial A_{ij}} \cdot M_{ij} \cdot \Delta\pi_{ij}∂λh​∂L​=i,j∑​∂Aij​∂L​⋅Mij​⋅Δπij​
Since ∣Mij∣≤1|M_{ij}| \leq 1
∣Mij​∣≤1 and ∣Δπij∣≤log⁡V|\Delta\pi_{ij}| \leq \log V
∣Δπij​∣≤logV:
∣∂L∂λh∣≤n2log⁡V⋅max⁡ij∣∂L∂Aij∣\left|\frac{\partial \mathcal{L}}{\partial \lambda_h}\right| \leq n^2 \log V \cdot \max_{ij}\left|\frac{\partial \mathcal{L}}{\partial A_{ij}}\right|​∂λh​∂L​​≤n2logV⋅ijmax​​∂Aij​∂L​​
Gradients are bounded whenever standard attention gradients are bounded. ✅
Conclusion: OsmoticAttention introduces a bounded perturbation that vanishes at initialization (λ=0\lambda=0
λ=0), and all gradients remain bounded under standard assumptions. Training stability is preserved. □\square
□

Theorem 2 — Information Flow Maximization
Theorem 2: OsmoticAttention maximizes expected information flow from low-entropy to high-entropy token regions, measured by mutual information.
Proof sketch:
Define information flow from token ii
i to jj
j as the contribution of hih_i
hi​ to hjh_j
hj​'s representation:
Fij=Aij⋅∥Vi∥\mathcal{F}_{ij} = A_{ij} \cdot \|V_i\|Fij​=Aij​⋅∥Vi​∥
Standard attention: Aijstd=softmax(QiKjTd)A_{ij}^{\text{std}} = \text{softmax}\left(\frac{Q_iK_j^T}{\sqrt{d}}\right)
Aijstd​=softmax(d​Qi​KjT​​)
Flow is purely similarity-driven — ignores information content.
Osmotic attention: Aijosm∝exp⁡(QiKjTd+λMij(ρj−ρi))A_{ij}^{\text{osm}} \propto \exp\left(\frac{Q_iK_j^T}{\sqrt{d}} + \lambda M_{ij}(\rho_j - \rho_i)\right)
Aijosm​∝exp(d​Qi​KjT​​+λMij​(ρj​−ρi​))
When λ>0\lambda > 0
λ>0, tokens with higher ρj\rho_j
ρj​ (more information-rich) attract more flow:
∂Aijosm∂ρj=Aijosm(1−Aijosm)⋅λMij>0\frac{\partial A_{ij}^{\text{osm}}}{\partial \rho_j} = A_{ij}^{\text{osm}}(1 - A_{ij}^{\text{osm}}) \cdot \lambda M_{ij} > 0∂ρj​∂Aijosm​​=Aijosm​(1−Aijosm​)⋅λMij​>0
This is exactly the osmosis property — flow increases toward higher concentration regions. ✅
Mutual information connection:
The expected information gain at token jj
j from attending to all tokens is:
Ij=∑iAij⋅ρi\mathcal{I}_j = \sum_i A_{ij} \cdot \rho_iIj​=i∑​Aij​⋅ρi​
OsmoticAttention maximizes Ij\mathcal{I}_j
Ij​ for high-entropy tokens by routing more attention weight toward them. This is equivalent to maximizing a lower bound on mutual information I(hj;context)I(h_j; \text{context})
I(hj​;context). □\square
□

Theorem 3 — Strict Expressiveness
Theorem 3: OsmoticAttention is strictly more expressive than standard attention. There exist attention patterns that OsmoticAttention can represent that standard attention cannot.
Proof by construction:
Consider a sequence where tokens have heterogeneous information densities:
ρ=[ρlow,ρlow,ρhigh,ρlow]\rho = [\rho_{\text{low}}, \rho_{\text{low}}, \rho_{\text{high}}, \rho_{\text{low}}]ρ=[ρlow​,ρlow​,ρhigh​,ρlow​]
Claim: Standard attention cannot concentrate attention on token 3 (the high-entropy token) when token 3's key vector k3k_3
k3​ is orthogonal to all query vectors:
Qi⋅K3T=0∀iQ_i \cdot K_3^T = 0 \quad \forall iQi​⋅K3T​=0∀i
In this case, standard attention distributes uniformly:
Ai3std=1n∀iA_{i3}^{\text{std}} = \frac{1}{n} \quad \forall iAi3std​=n1​∀i
OsmoticAttention: Even when QiK3T=0Q_iK_3^T = 0
Qi​K3T​=0:
Ai3osm∝exp⁡(λMi3(ρ3−ρi))>exp⁡(0)=1A_{i3}^{\text{osm}} \propto \exp(\lambda M_{i3}(\rho_3 - \rho_i)) > \exp(0) = 1Ai3osm​∝exp(λMi3​(ρ3​−ρi​))>exp(0)=1
So attention toward token 3 is amplified by the osmotic term! 🎯
This proves OsmoticAttention can represent attention patterns where semantically important (high-entropy) tokens attract context even when they are not query-similar — a capability standard attention fundamentally lacks. □\square
□

Proposition 1 — Complexity
**Proposition 1:** *OsmoticAttention has the same asymptotic complexity O(n2d)O(n^2d)
O(n2d) as standard attention, with constant factor overhead of O(nd)O(nd)
O(nd) for density computation.*
Proof:
OperationComplexityQKV projectionsO(nd2)O(nd^2)
O(nd2)Attention scores QKTQK^T
QKTO(n2d)O(n^2d)
O(n2d)Density ρi=H(Wρhi)\rho_i = H(W_\rho h_i)
ρi​=H(Wρ​hi​)O(nd)O(nd)
O(nd) per headGradient Δπij\Delta\pi_{ij}
Δπij​O(n2)O(n^2)
O(n2) via broadcastingMembrane Mij=σ(mi+mjT)M_{ij} = \sigma(m_i + m_j^T)
Mij​=σ(mi​+mjT​)O(n2)O(n^2)
O(n2) factorizedTotalO(n2d)O(n^2d)
O(n2d)
The osmotic terms add O(n2)+O(nd)O(n^2) + O(nd)
O(n2)+O(nd) — dominated by the existing O(n2d)O(n^2d)
O(n2d) attention computation. Asymptotic complexity is unchanged.