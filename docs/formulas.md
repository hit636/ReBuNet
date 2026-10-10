# Core formulas and implementation map

These formulas document the implemented reconstruction. Symbols use NB mean
`mu`, concentration `k > 0`, occurrence probability `p`, mixture weights `w`,
and observed-target indicator `v`. See the README for reproduction limitations.

## Positive count distribution

For a negative binomial parameterized by mean and concentration:

$$P_{NB}(y;\mu,k)=\frac{\Gamma(y+k)}{\Gamma(k)\Gamma(y+1)}\left(\frac{k}{k+\mu}\right)^k\left(\frac{\mu}{k+\mu}\right)^y.$$

$$Z(\mu,k)=1-P_{NB}(0;\mu,k),\qquad f_+(y)=P_{NB}(y)/Z,\quad y\ge1.$$

$$M(\mu,k)=E[Y\mid Y>0]=\mu/Z(\mu,k).$$

`models/distributions.py`: `nb_log_prob`, `zero_normalizer`, `positive_mean`.
Log-space arithmetic and `expm1` avoid cancellation near zero. A small-mean
series expansion is used where direct division becomes unstable.

## Ordered positive means and inverse

$$m_0=1+\operatorname{softplus}(a),\qquad m_1=m_0+\operatorname{softplus}(b)+\delta.$$

`models/rebunet.py`: `MarkHead` orders the **conditional positive means**.
Each underlying NB mean solves $M(\mu_j,k_j)=m_j$ using safeguarded Newton
iterations with bisection. Ordering raw NB means alone does not ensure this
conditional ordering when component concentrations differ.

$$\frac{\partial\mu}{\partial m}=\frac1{M_\mu},\qquad
\frac{\partial\mu}{\partial k}=-\frac{M_k}{M_\mu}.$$

`models/distributions.py`: `_InverseMean.backward` implements this implicit
gradient. It does not differentiate through the root-finding iterations.

## Hurdle mixture, forecast and loss

$$P(Y=0)=1-p,\qquad P(Y=y)=p\sum_jw_jf_{+,j}(y),\quad y\ge1.$$

$$\widehat y=E[Y]=p\sum_jw_jm_j,\qquad
\mathcal L=-\frac{\sum_{i,t}v_{i,t}\log P(Y_{i,t}=y_{i,t})}{\sum_{i,t}v_{i,t}}.$$

`Forecast.log_prob`, `Forecast.mean`, and `masked_nll` implement these equations.
The loss uses available observations only. The all-masked batch returns zero
loss; unavailable dates remain on the calendar and are not labeled zero demand.
The ordinary NB baseline bypasses the hurdle and retains its own zero mass.

## Renewal propagation

For a scalar age state $r_h$ and occurrence probability $p_h$:

$$r_{h+1}=(1-p_h)(r_h+1).$$

This is a mean-state approximation. The exact reachable-age decoder instead
propagates age probabilities $\pi_h(a)$ using age-specific hazards $q_h(a)$:

$$p_h=\sum_a\pi_h(a)q_h(a),\quad
\pi_{h+1}(0)=p_h,\quad
\pi_{h+1}(a+1)=\pi_h(a)(1-q_h(a)).$$

The capped decoder merges surviving overflow into its final bucket. Both
preserve probability mass. Positive mark heads in this implementation use the
mean age $\sum_a a\pi_h(a)$ rather than a mixture of age-specific mark heads.
See `models/rebunet.py`: `ReBuNet.forward` for scalar, EAD and CAD updates.

## Evaluation

$$\mathrm{MAE}=\operatorname{mean}|y-\widehat y|,\qquad
\mathrm{RMSSE}_i=\sqrt{\mathrm{MSE}_i/s_i},\qquad
\mathrm{DRPS}(F,y)=\sum_{z=0}^{\infty}(F(z)-\mathbf1[y\le z])^2.$$

Here $s_i$ is the training first-difference squared-error scale, using valid
adjacent observations. The M5 variant excludes leading zero sales for scaling.
Unavailable targets are excluded. Per-series eligibility, zero scales, revenue
weights and adaptive DRPS tail bounds are handled in `evaluation/metrics.py`
and `evaluation/probability.py`. Bottom-level bWRMSSE and official 12-level M5
WRMSSE are separate outputs; see the README for their respective protocols.
