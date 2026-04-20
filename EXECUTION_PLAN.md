# 执行计划 · Machine Learning for Intelligent Agents with Limited Resources

> **项目周期**：2026-04-06 → 2026-08-22（20 周）
> **当前定位**：Week 3（2026-04-20 ~ 04-26）
> **导师**：Richard Mayr　**Tutor**：Aurora Constantin　**UID**：s2798452

本计划由 Proposal 精读而来，每一条都对应代码仓库里的具体模块 / 脚本 / 测试。勾选框用于每周对照进度。

---

## 0. 当前状态总览

| 阶段 | Proposal 任务 | 代码模块 | 状态 |
|------|---------------|----------|------|
| T1 | Literature + Setup（Wks 1–2） | 整个仓库骨架已搭完 | 本周内跑通 `pytest` 即达成 M1 |
| T2 | Unconstrained Static M1/M2（Wks 3–5） | `src/monitors/m1_unconstrained.py`, `m2_soft_accept.py` | 🔲 |
| T3 | Counting Monitors M3/M4（Wks 2–3） | `src/monitors/m3_counting.py`, `m4_minimal.py` | 🔲 **本周主攻** |
| T4 | GA + Exhaustive（Wks 4–5） | `src/optim/genetic.py`, `exhaustive.py` | 🔲 |
| T5 | Dynamic Monitor（Wks 6–8） | `src/monitors/dynamic.py` | 🔲 |
| T6 | Experiments + Analysis（Wks 7–9） | `experiments/phase*.py` + `src/analysis/` | 🔲 |
| T7 | Bonus: Analytic Approximation（Wks 10–11） | `src/analysis/fitting.py` | 🔲 |
| T8 | Dissertation Writing（Wks 10–20） | — | 🔲 |

---

## 1. Milestones（对应 Proposal Table 3）

| ID | Wk | 描述 | 验收标准 |
|----|----|------|----------|
| M1 | 2  | Literature 读完，模拟器过边界检查 | `pytest tests/test_boundaries.py` 全绿 |
| M2 | 5  | M1–M4 Phase 1–2 完成，exhaustive ground truth 验证 | `results/csv/phase1.csv` 和 `phase2.csv` 有完整条目 |
| M3 | 9  | Dynamic monitor 训练完成，∆(n,k) 计算完毕，中期报告 | `phase2.csv` 的 `Dynamic` 行齐全 |
| M4 | 11 | 所有实验完成，图表定稿 | `results/figures/` 齐全 |
| M5 | 20 | Dissertation 提交 | PDF → MyEd |

---

## 2. Week-by-Week 时间线（含今日可做的具体动作）

### Week 3（4/20–4/26，**本周**）— 收尾 T1 + 启动 T3

- [ ] **Mon 4/20** `python -m venv .venv && pip install -r requirements.txt`
- [ ] **Mon 4/20** `pytest tests/` → 确认 boundary + recursion 全绿 ⇒ **M1 达成**
- [ ] **Tue 4/21** 跑通 `python -m src.core.boundary_checks`（应打印 0.5 / 0.0）
- [ ] **Tue 4/21** 初始化 git 仓库 + 推到 University GitLab
- [ ] **Wed 4/22** 在 Python REPL 里手动训练一次 M4（k=2, n=11）验证 Adam 收敛
- [ ] **Thu 4/23** 跑 `python -m experiments.phase1_exploratory`（即使只跑 5 restarts 也行）
- [ ] **Fri 4/24** 导师 30 分钟 meeting：带上 Phase 1 初步 CSV
- [ ] **Sat 4/25** 扫描 nesting violations，记录哪些 (n,k) 需要加 restart
- [ ] **Sun 4/26** Week 4 预热：读 Hellman–Cover 1970 + Cover–Freedman–Hellman 1976

### Week 4（4/27–5/3）— T2 启动 + T4 exhaustive

- [ ] Phase 1 所有 (n,k) ∈ {11,21}×{2,3} 全部跑满 20 restarts
- [ ] 实现 exhaustive search 并对 (n=5,k=2) 的 gradient 结果做 ground-truth 对比
- [ ] 若 gradient 与 exhaustive 差距 > 1% → 加 restart / 换 lr / 跑 GA

### Week 5（5/4–5/10）— T2/T4 收尾 + **M2 达成** + 交付 D1

- [ ] M1/M2 warm-start from M3 optimum（代码已写好 `to_m1_warm_start`）
- [ ] GA 在 (n=21, k=3, M1) 上独立验证 gradient 结果
- [ ] 打包 **D1**：M1–M4 静态 codebase + `README.md` + `tests/` 全绿 + tag `v0.1`

### Week 6–8（5/11–5/31）— T5 Dynamic

- [ ] Phase 2 静态结果全部落地（`phase2_core.py` 的静态部分）
- [ ] `DynamicMonitor`（full parameterisation）在 n≤51 跑通
- [ ] `SharedDynamicMonitor` 在 n=101 跑通
- [ ] 计算 Δ(n,k) 并检查 H4 方向（n↑ → Δ↑, k↑ → Δ↓）
- [ ] 实现 H2 annealing trace（`src/optim/anneal.py` 已就绪）

### Week 7–9（5/25–6/14，overlapping）— T6 分析 + **M3 中期**

- [ ] 用 `src/analysis/heatmap.py` 对每个最优 monitor 出 H*/T* 热图
- [ ] 用 `src/analysis/posterior_scatter.py` 出 50k sample 的 (p_i, s_i) 散点
- [ ] 用 `src/analysis/tridiagonality.py` 计算 τ(H*)，写入 CSV
- [ ] 用 `src/analysis/class_gaps.py` 汇总 Δ^acc, Δ^count, Δ^minimal 表
- [ ] 写 5–6 页中期报告，交导师 review
- [ ] 交付 **D2**：dynamic code + empirical `f_Mi(n,k)` CSV + class-gap tables

### Week 10–11（6/15–6/28）— T7 奖励 + **M4 达成**

- [ ] 用 `src/analysis/fitting.py` 拟合三种候选形式，按 AIC/BIC 选优
- [ ] R² < 0.9 则**直接放弃**本项，立刻转 T8 写作
- [ ] 所有图表定稿存入 `results/figures/`
- [ ] 交付 **D3**：analysis report + baseline comparison + A/B 分类

### Week 10–20（6/15–8/22）— T8 Writing

- [ ] **Wk 10–13** Chapter 1 Introduction + Chapter 2 Background 初稿
- [ ] **Wk 14** 全稿发导师（≥1 周提前量）
- [ ] **Wk 15–17** Chapter 3 Method + Chapter 4 Results + Chapter 5 Discussion
- [ ] **Wk 18** Final proofread + 去掉 placeholder 图
- [ ] **Wk 19** 参考文献终稿，检查 DOI 链接
- [ ] **Wk 20** `M5` **提交 2026-08-22**

---

## 3. 硬性技术决策（已在代码里固化）

| 决策 | 代码位置 | 说明 |
|------|----------|------|
| `DEFAULT_DTYPE = float64` | `src/core/recursion.py` | f(n,k) 量级到 1e-3，float32 会污染 nesting |
| `torch.softmax` on logits | monitors/*.py | 自然保证 row-stochastic |
| Seed = 12345 + restart_idx | `src/optim/gradient.py` | 每次 restart 可复现 |
| 早停 patience=500 | `train_one` | 避免在 plateau 上浪费 compute |
| Warm-start hooks | `M3.to_m1_warm_start`, `M4.to_m3_warm_start` | 从简单类爬到复杂类 |
| Shared MLP dynamic | `SharedDynamicMonitor` | n=201 时参数数量脱钩 |

---

## 4. 风险与预案（Proposal Table 2 的实操版）

| 风险 | 触发条件 | 立即行动 |
|------|----------|----------|
| 局部最优 | 某 (n,k) 下 restart std/mean > 10% | 加到 30 restarts；warm-start from 相邻 (n±10, k); 回跑 GA |
| Dynamic 太慢 | n=101 单次 > 10 min | 切到 SharedDynamicMonitor；hidden=32 |
| Nesting 被破坏 | `test_nesting.py` failing | 对违反的那一类重训 3× restart 数 |
| 奖励项无解析解 | Wk 10 末 R² < 0.9 | 直接丢掉 T7，省时间给 T8 |
| Dynamic gap 接近 0 | Δ(n,k) < 1e-4 | Null result 也写进 dissertation；改讨论 sufficient-k threshold |
| Compute 不够 | laptop 跑不动 n=101 | 转到 DICE cluster（ssh + slurm），代码没有 cuda 硬依赖 |

---

## 5. 每周例会（与导师 Richard Mayr）

- **Wk 1–10**：每周 30 分钟
- **Wk 11–20**：每两周一次
- 每次例会前 24 小时把本周进度（用本文件勾选框）邮件发送
- 每个 deliverable 至少提前 1 周把草稿发给导师

---

## 6. 如何开始（今天 4/20 可以做的 3 件事）

```bash
# 1. 进入项目目录
cd D:\Users\12057\Desktop\FYP\Code

# 2. 创建虚拟环境 + 安装依赖
python -m venv .venv
.\.venv\Scripts\activate    # Windows
pip install -r requirements.txt

# 3. 跑完整测试套件（M1 里程碑）
pytest tests/ -v

# 4. 初始化 git 并推到 GitLab
git init
git add .
git commit -m "Initial framework: M1-M4 monitors, core recursion, boundary tests"
git remote add origin git@git.ecdf.ed.ac.uk:s2798452/fyp-majority-monitor.git
git push -u origin main
```

跑通之后，本周剩余时间花在 `python -m experiments.phase1_exploratory` 上。**任何一处 nesting violation 或边界测试失败都要在当周解决**，否则会把延迟推到 Phase 2。

---

## 7. Hypotheses Tracker（实验出结果时填）

| ID | 描述 | 判据 | 现状 |
|----|------|------|------|
| H1 | Optimal static ≈ stochastic counter | `Δ^count / f_M1 ≤ 10%` across grid | ⏳ |
| H2 | Randomisation is necessary | Anneal τ→0 推高 f(n,k) | ⏳ |
| H3 | Dynamic monitor structure（open） | 报告 per-row entropy + state→posterior 单调性 | ⏳ |
| H4 | Δ(n,k) 随 n↑ 增, 随 k↑ 减 | 双变量趋势图 | ⏳ |
| H5 | Soft acceptance 有帮助 | `Δ^acc > 0` 在小 k 处 | ⏳ |

---

## 8. 代码清单（此次搭建内容）

```
majority-monitor/
├── README.md
├── LICENSE                         MIT
├── pyproject.toml
├── requirements.txt
├── EXECUTION_PLAN.md               <- this file
├── .gitignore
├── src/
│   ├── __init__.py
│   ├── core/
│   │   ├── recursion.py            π^(h)_i 的 O(n²k²) 可微递推  ★
│   │   ├── loss.py                 majority_loss(monitor, n)
│   │   └── boundary_checks.py      f(n,1)=0.5, f_saturating=0
│   ├── monitors/
│   │   ├── base.py                 抽象基类
│   │   ├── m1_unconstrained.py     2 k(k-1)+1 参数
│   │   ├── m2_soft_accept.py       2 k(k-1)+k 参数
│   │   ├── m3_counting.py          3k 参数，自带 warm-start
│   │   ├── m4_minimal.py           k+2 参数
│   │   └── dynamic.py              DynamicMonitor + SharedDynamicMonitor
│   ├── optim/
│   │   ├── gradient.py             Adam + 多重启动 + early stop
│   │   ├── anneal.py               Gumbel-Softmax / 温度退火
│   │   ├── genetic.py              GA: tournament / Gaussian / crossover
│   │   └── exhaustive.py           小 (n,k) 的 grid-search ground truth
│   ├── analysis/
│   │   ├── heatmap.py              H*, T* 热图
│   │   ├── posterior_scatter.py    (p_i, s_i) 散点
│   │   ├── tridiagonality.py       τ(H) score
│   │   ├── class_gaps.py           Δ^acc, Δ^count, Δ^minimal
│   │   └── fitting.py              C k^-α / C exp(-βk) / C (n/k)^-γ
│   └── baselines/
│       └── mk_random_walk.py       Kontorovich M(k) baseline
├── tests/
│   ├── test_boundaries.py          ★ M1 milestone 的判据
│   ├── test_recursion.py
│   └── test_nesting.py
├── experiments/
│   ├── phase1_exploratory.py       n∈{11,21}, k∈{2,3}
│   ├── phase2_core.py              n∈{51,101}, k∈{2,3,5} + dynamic
│   └── phase3_stretch.py           n=201, k∈{10,20}
├── scripts/
│   └── reproduce.sh                一键重生成所有 CSV
└── results/
    ├── csv/
    ├── figures/
    └── logs/
```
