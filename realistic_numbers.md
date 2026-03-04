Short answer: for a 70B run, don’t assume 4.7% unless your setup matches theirs closely.

Paper’s 4.7% is from their measured setup (`/Users/rohanshravan/Documents/New project/Opus.md:3107`), where sequence/token economics were different (very long `L_train` in GPT-2 Large case, plus their exact infra optimizations).

A practical estimate for your setup is:

\[
\text{overhead} \approx \frac{(N + K_{proxy}) \cdot L_{score}}{(\rho N)\cdot L_{train}} + \text{(small sketch+comm)}
\]

Using your defaults in code (`/Users/rohanshravan/Documents/New project/production/config.py:11-15,48`):
- `N=16`, `rho=0.5`, `L_score=512`, `K_proxy=8`, `L_train=4096`
- token-ratio term = `0.375` => **37.5%** before extra overhead terms.

If you use paper-like `N=32`, `rho=0.5`, `K_proxy=8`, `L_score=512`:
- at `L_train=4096` => **31.25%**
- at `L_train=8192` => **15.6%**
- at `L_train=24576` => **5.2%** (this is why paper can be near 4.7%).

So for 70B with `L_train` around 4k–8k, expect roughly **~15% to ~40%** extra compute if you score every step with these settings, not 4.7%.

Yes, Ghost + CountSketch + frozen preconditioner are exactly the mechanisms that keep this from being far worse (paper says naive online selection would be much slower), but they do not guarantee 4.7% in all regimes.