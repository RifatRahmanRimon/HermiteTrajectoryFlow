# Hermite Conditional Paths for Flow Matching

## Notation

| Symbol | Meaning |
|---|---|
| $\mathcal{D} = \{\mathbf{X}^{(k)}\}_{k=1}^{K}$ | dataset of $K$ observed trajectories |
| $\mathbf{X}^{(k)} = \{(t_i, x_i)\}_{i=1}^{N_k}$ | trajectory $k$, $x_i \in \mathbb{R}^d$, $t_1 < \cdots < t_{N_k}$ |
| $M \in \{0,1\}^{N_k \times d}$ | observation mask; $M_{i,j}=1$ iff coordinate $j$ observed at $t_i$ |
| $\mathcal{P} = \{(p,v)\}$ | pairing set; coordinate $v$ is the time-derivative of coordinate $p$ |
| $\mathcal{U}$ | unpaired coordinates, $\mathcal{U} = [d] \setminus \bigcup_{(p,v)\in\mathcal{P}}\{p,v\}$ |
| $m$ | B-spline degree |
| $\sigma$ | conditional path noise scale |
| $\mu^{(k)}:[t_1,t_{N_k}] \to \mathbb{R}^d$ | conditional path of trajectory $k$ |
| $u_\theta(t,x)$ | velocity network |

$\mathcal{H}[\,\cdot\,]$ denotes the cubic Hermite interpolant, $\mathcal{B}_m[\,\cdot\,]$ the
B-spline interpolant of degree $m$:

$$
\mathcal{H}\big[\{(\tau_i, y_i, y'_i)\}\big]: \quad h(\tau_i) = y_i, \quad h'(\tau_i) = y'_i
$$

$$
\mathcal{B}_m\big[\{(\tau_i, y_i)\}\big]: \quad b(\tau_i) = y_i, \quad b \in C^{m-1}
$$

State is assumed **second-order**: for each $(p,v) \in \mathcal{P}$ the true dynamics
satisfy $\dot{x}^{(p)}(t) = x^{(v)}(t)$ exactly.

---

## Algorithm 1 — Conditional path construction, hedge variant

**Input:** $\mathbf{X} = \{(t_i, x_i)\}_{i=1}^{N}$, mask $M$, pairing $\mathcal{P}$, unpaired $\mathcal{U}$, degree $m$
**Output:** $\mu(\cdot), \ \mu'(\cdot)$

---

1. **for** each $(p,v) \in \mathcal{P}$ **do**
2. $\quad \mathcal{I}_{pv} \leftarrow \{\, i : M_{i,p} = 1 \ \wedge \ M_{i,v} = 1 \,\}$
3. $\quad$ **if** $|\mathcal{I}_{pv}| \ge 2$ **then**
4. $\quad\quad h_p \leftarrow \mathcal{H}\big[\{(t_i,\ x_i^{(p)},\ x_i^{(v)})\}_{i \in \mathcal{I}_{pv}}\big]$
5. $\quad$ **else**
6. $\quad\quad h_p \leftarrow \mathcal{B}_m\big[\{(t_i, x_i^{(p)})\}_{i : M_{i,p}=1}\big]$    ▹ fallback
7. $\quad$ **end if**
8. $\quad b_v \leftarrow \mathcal{B}_m\big[\{(t_i, x_i^{(v)})\}_{i : M_{i,v}=1}\big]$
9. **end for**
10. **for** each $u \in \mathcal{U}$ **do**
11. $\quad b_u \leftarrow \mathcal{B}_m\big[\{(t_i, x_i^{(u)})\}_{i : M_{i,u}=1}\big]$
12. **end for**
13. **return**

$$
\mu(t)^{(j)} =
\begin{cases}
h_p(t), & j = p,\ (p,v)\in\mathcal{P}\\[2pt]
b_v(t), & j = v,\ (p,v)\in\mathcal{P}\\[2pt]
b_u(t), & j = u \in \mathcal{U}
\end{cases}
\qquad
\mu'(t)^{(j)} =
\begin{cases}
h_p'(t), & j = p\\[2pt]
b_v'(t), & j = v\\[2pt]
b_u'(t), & j = u
\end{cases}
$$

**Number of fits:** $d$. **Guarantee:** $\mu'(t_i)^{(p)} = x_i^{(v)}$ for all $i \in \mathcal{I}_{pv}$.
**Relaxation:** $\mu'(t)^{(p)} \ne \mu(t)^{(v)}$ for $t \notin \{t_i\}$.

---

## Algorithm 2 — Conditional path construction, pure variant

**Input:** as Algorithm 1
**Output:** $\mu(\cdot), \ \mu'(\cdot)$

---

1. **for** each $(p,v) \in \mathcal{P}$ **do**
2. $\quad \mathcal{I}_{pv} \leftarrow \{\, i : M_{i,p} = 1 \ \wedge \ M_{i,v} = 1 \,\}$
3. $\quad h_p \leftarrow \mathcal{H}\big[\{(t_i,\ x_i^{(p)},\ x_i^{(v)})\}_{i \in \mathcal{I}_{pv}}\big]$
4. **end for**
5. **for** each $u \in \mathcal{U}$ **do**
6. $\quad b_u \leftarrow \mathcal{B}_m\big[\{(t_i, x_i^{(u)})\}_{i : M_{i,u}=1}\big]$
7. **end for**
8. **return**

$$
\mu(t)^{(j)} =
\begin{cases}
h_p(t), & j = p\\[2pt]
h_p'(t), & j = v\\[2pt]
b_u(t), & j = u
\end{cases}
\qquad
\mu'(t)^{(j)} =
\begin{cases}
h_p'(t), & j = p\\[2pt]
h_p''(t), & j = v\\[2pt]
b_u'(t), & j = u
\end{cases}
$$

**Number of fits:** $|\mathcal{P}| + |\mathcal{U}|$. **Guarantee:** $\mu'(t)^{(p)} = \mu(t)^{(v)}$ for all $t$.
**Relaxation:** $h_p \in C^1$ only, so $\mu'(t)^{(v)} = h_p''(t)$ is piecewise linear and
discontinuous at the knots $\{t_i\}_{i \in \mathcal{I}_{pv}}$.

---

## Algorithm 3 — Training

**Input:** $\mathcal{D}$, pairing $\mathcal{P}$, degree $m$, noise $\sigma$, network $u_\theta$, steps $S$
**Output:** trained $\theta$

---

1. **for** $k = 1, \dots, K$ **do**
2. $\quad \mu^{(k)}, \mu^{(k)\prime} \leftarrow \textbf{Algorithm 1 or 2}\big(\mathbf{X}^{(k)}, M^{(k)}, \mathcal{P}, \mathcal{U}, m\big)$    ▹ precompute
3. **end for**
4. **for** $s = 1, \dots, S$ **do**
5. $\quad k \sim \mathrm{Unif}\{1,\dots,K\}$
6. $\quad t \sim \mathrm{Unif}\big[t_1^{(k)},\, t_{N_k}^{(k)}\big]$
7. $\quad x \sim \mathcal{N}\big(\mu^{(k)}(t),\ \sigma^2 I_d\big)$
8. $\quad \mathcal{L}(\theta) \leftarrow \big\| u_\theta(t, x) - \mu^{(k)\prime}(t) \big\|_2^2$
9. $\quad \theta \leftarrow \theta - \eta \nabla_\theta \mathcal{L}(\theta)$
10. **end for**
11. **return** $\theta$

---

## Algorithm 4 — Inference

**Input:** initial state $x_{t_0}$, query time $t^\star$, step count $L$, trained $u_\theta$
**Output:** $\hat{x}_{t^\star}$

---

1. $\Delta t \leftarrow (t^\star - t_0)/L$
2. $\hat{x} \leftarrow x_{t_0}$
3. **for** $\ell = 0, \dots, L-1$ **do**
4. $\quad \hat{x} \leftarrow \hat{x} + \Delta t \cdot u_\theta\big(t_0 + \ell \Delta t,\ \hat{x}\big)$    ▹ or RK4
5. **end for**
6. **return** $\hat{x}$

For the stochastic variant, replace line 4 with an Euler–Maruyama step and train a
score network $s_\phi$ jointly with $u_\theta$ in Algorithm 3.

---

## Remarks

**R1 (identity constraint).** Existing methods fit each coordinate $j \in [d]$
independently, giving $\mu'(t)^{(p)} \ne \mu(t)^{(v)}$ in general, even though
$\dot{x}^{(p)} = x^{(v)}$ holds exactly for the true trajectory. Both algorithms
impose this constraint at the interpolant level: Algorithm 1 at the knots,
Algorithm 2 everywhere.

**R2 (approximation order).** Cubic Hermite interpolation attains
$\|h - f\|_\infty = O(\Delta^4)$ and $\|h' - f'\|_\infty = O(\Delta^3)$ for
$\Delta = \max_i (t_{i+1}-t_i)$, matching cubic B-splines. The distinction in R1 is
therefore not asymptotic in $\Delta$.

**R3 (degree).** Two-point Hermite interpolation matching derivatives to order $\kappa$
at both endpoints imposes $2(\kappa+1)$ conditions and determines a polynomial of
degree $2\kappa+1$. With only $(x^{(p)}, x^{(v)})$ observed, $\kappa = 1$ and the
construction is fixed at cubic.

**R4 (mask intersection).** Line 2 of both algorithms requires $p$ and $v$ to be
observed simultaneously. Under independent per-coordinate missingness with rate
$\rho$, $\mathbb{E}|\mathcal{I}_{pv}| = (1-\rho)^2 N$. Report the fallback rate of
line 6 (Algorithm 1) alongside all results.

**R5 (reduction).** Setting $\mathcal{P} = \emptyset$ recovers the B-spline baseline
exactly in both algorithms.
