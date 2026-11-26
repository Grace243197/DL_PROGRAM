# W13 中期结果表（草稿）

> 更新：2026-11-26 ｜ 状态：**草稿**，等待蔡的 within 四格 × 3 seeds 回填
> 数据：UTA-RLDD 60 人，22,553 个合格 5s 窗口；subject-disjoint 42/9/9；
> within subject×class 随机 70/15/15（seed 20261105）
> 关联：PR #2（within 划分 + RF）、PR #3（W11 模型线）、`W12_交接说明.md`

## 1. 主表：窗口级 Macro-F1

3 seeds 均值 ± SD（RF、W11 四格）。within 只有 RF（3 seeds）、CNN-Frame 和
CNN-Frame-Prior（seed 42 本地预览）已完成；其余标“待蔡”。

| 模型 | subject-disjoint val | subject-disjoint test | subject-disjoint 视频级 | subject-disjoint 95% CI | within val | within test | within 视频级 |
|---|---:|---:|---:|---:|---:|---:|---:|
| **RF-Prior** | 0.433 ± 0.002 | **0.453 ± 0.003** | — | — | 0.811 ± 0.003 | **0.816 ± 0.004** | — |
| CNN-Frame | 0.371 ± 0.006 | 0.232 ± 0.015 | 0.228 | 0.153–0.299 | 0.999* | **0.9985*** | 1.000* |
| CNN-Frame-Prior | 0.434 ± 0.008 | 0.253 ± 0.023 | 0.235 | 0.167–0.331 | 0.996* | **0.9973*** | 1.000* |
| CNN-GRU | 0.388 ± 0.021 | 0.253 ± 0.013 | 0.277 | 0.181–0.327 | 待蔡 | 待蔡 | 待蔡 |
| CNN-GRU-Fusion | 0.439 ± 0.023 | **0.268 ± 0.021** | 0.224 | 0.183–0.348 | 待蔡 | 待蔡 | 待蔡 |
| 多数类基线 | — | 0.167 | — | — | — | — | — |

`*` CNN-Frame / CNN-Frame-Prior within 是 **seed 42 单 seed 本地预览**：
CNN-Frame epochs 19、best 14、QWK 0.9982、CI 0.9967–0.9997；
CNN-Frame-Prior epochs 10、best 5、QWK 0.9967、CI 0.9936–0.9994。
不是 3-seed 正式结果。

**读法**

- subject-disjoint 下所有 CNN 格都远低于 RF-Prior；严格负结果。
- within 下 RF 升到 0.816，CNN-Frame 直接接近满分；这是同受试者/同视频窗口
  跨集合造成的虚高，只能作为 H3 证据，不能当真实检测性能。
- subject-disjoint test 已被多次评估，**当前 test 数字只能标 exploratory**；
  within test 按“最终配置只评一次”使用。

## 2. 每类召回（test，3 seeds 均值）

| 模型 | Alert | Low-vigilant | Drowsy |
|---|---:|---:|---:|
| RF-Prior（subject-disjoint） | **0.452** | 0.352 | **0.567** |
| RF-Prior（within） | 0.842 | 0.789 | 0.817 |
| CNN-Frame（subject-disjoint） | 0.135 | 0.269 | 0.306 |
| CNN-Frame-Prior（subject-disjoint） | 0.210 | **0.355** | 0.213 |
| CNN-GRU（subject-disjoint） | 0.079 | **0.368** | 0.371 |
| CNN-GRU-Fusion（subject-disjoint） | 0.169 | 0.327 | 0.356 |
| CNN-Frame（within，seed 42） | 0.998 | 0.998 | 0.999 |
| CNN-Frame-Prior（within，seed 42） | 0.996 | 0.997 | 1.000 |

**失败模式**：subject-disjoint 下 CNN 的主要问题是 **Alert 召回崩溃**
（0.079–0.210），而 RF 能到 0.452。先验注入确实把 Alert 召回从 0.135/0.079
提高到 0.210/0.169，但仍然远低于 RF；这才是 CNN 宏平均被拉低的主因，而不是
单纯的 Low-vigilant 难分。

## 3. 2×2 主效应（subject-disjoint test，探索性）

按 seed 配对，4 格 × 3 seeds：

| 效应 | 均值 ± seed SD | 三个 seed |
|---|---:|---|
| 时序（GRU − Frame） | **+0.0180 ± 0.0111** | +0.0298 / +0.0164 / +0.0078 |
| 先验（Prior − No prior） | +0.0180 ± 0.0250 | −0.0108 / +0.0312 / +0.0337 |
| 交互（GRU 效应 − Frame 效应） | −0.0049 ± 0.0225 | −0.0192 / −0.0165 / +0.0211 |

边际均值：Frame 0.2426 / GRU 0.2605；No prior 0.2425 / Prior 0.2606。

- 时序效应三个 seed 方向一致，但只有 +0.018；H1 的“GRU 有帮助但幅度小”
  得到初步支持。
- 先验效应被 seed 噪声淹没（seed 42 反向）；交互不清晰；H2 目前证据很弱。
- 所有 CNN 格仍远低于 RF-Prior 0.453，中期报告不要过度解读这组小效应。

## 4. H3 落差

| 模型 | subject-disjoint test | within test | 落差 |
|---|---:|---:|---:|
| RF-Prior | 0.453 | 0.816 | **+0.363** |
| CNN-Frame（seed 42 预览） | 0.232 | 0.9985 | **+0.766** |
| CNN-Frame-Prior（seed 42 预览） | 0.253 | 0.9973 | **+0.744** |

within 接近饱和，说明在这套数据上“同受试者/同视频窗口随机切分”几乎把任务
变成了身份识别；这正是 Proposal 批评的 within-subject 虚高。H1/H2 的
2×2 效应应主要在 subject-disjoint 上读。

## 5. 必须披露的 caveats

1. **test-peeking**：subject-disjoint test 已用于 seed-42 四格、调参配置和
   12 个主对比 run；当前 test 数字标 exploratory。within test 目前是干净的，
   最终配置只评一次。
2. **9 人 val 太小**：不要再用 9 人 val 选模型；选模型用 42 个训练受试者的
   5 折 subject-level CV（W11 已跑）。
3. **within 饱和**：train/val/test 共享受试者和视频；不能把 0.997–0.999
   当成真实驾驶场景性能。
4. **环境差异**：蔡用 torch 2.14.1+cu126；毛本机用 torch 2.11.0+cu128。
   互跑 CNN-Frame subject-disjoint seed 42：0.2196 vs 0.2285（Δ≈0.009），
   趋势一致；正式数字以蔡的 `dl_env` 为准。
5. **RF 没有视频级聚合**：当前 RF 脚本只出窗口级指标，与 CNN 的视频级列
   不可直接比较；表里用“—”。

## 6. 机制探针初版（clean within test）

对 CNN-Frame-Prior 的 seed-42 checkpoint 在 within test 上统计 FiLM gate
（576 维，3,378 个窗口）：

| 统计量 | 值 |
|---|---:|
| gate 均值 / 标准差 | 0.689 / 0.278 |
| gate 最小 / 最大 | ~0 / 1.0 |
| 低饱和比例（<0.01） | 0.20% |
| 高饱和比例（>0.99） | 4.04% |
| 每类 gate 均值（alert / low / drowsy） | 0.691 / 0.687 / 0.688 |
| 每受试者 gate 均值范围 | 0.658–0.717 |
| 每维 gate 均值范围 | 0.086–0.980 |

- 在 **clean within test** 上，gate 的类间差异几乎为零（0.687–0.691），
  受试者间范围也很窄；它更像一个全局的、按先验整体缩放的调制，而不是
  class-specific 的判别器。
- 高饱和维度约 4%，低饱和约 0.2%；gate 没有整体塌缩。
- 这只是单 checkpoint、within test、clean 条件的初版；真正的 H2 机制证据
  要看 corruption 条件下 gate 均值/方差是否变化（W14）。
- 探针脚本 `scripts/probe_gate.py` 目前在毛的本地运行分支，尚未入库；等
  W11 合并后可以作为 model-side 小 PR 加入。

## 7. 待回填 / 下一步

- [ ] 蔡：within 四格 × 3 seeds（`--split-column within_split`），回填主表
      与每类召回；其中 CNN-Frame / CNN-Frame-Prior 的 seed-42 本地预览已补。
- [ ] 蔡：within 的 CNN-GRU / CNN-GRU-Fusion × 3 seeds（主表剩余两行）。
- [x] 毛：2×2 主效应探索性分析（见 §3）。
- [ ] 毛/蔡：corruption 条件下 gate 探针（W14），比较 gate 均值/方差变化。
- [ ] 两人：确认中期报告的主表口径（窗口级为主，视频级为辅；test 标
      exploratory）。
- [ ] 毛：数据/结果章节文字；蔡：方法/模型章节文字；W13 合稿。
- [ ] 可选：给 RF 补 subject-level bootstrap CI（当前脚本未输出）。
- [ ] 可选：把 2×2 效应用更正式的模型（seed 作为随机效应）复核；目前只有
      描述性均值 ± seed SD，n=3。

## 8. 中期报告数据/结果章节骨架（毛负责）

建议按以下顺序写（英文报告可直接用这些段落标题）：

1. **Dataset and splits**
   - UTA-RLDD，60 人，22,553 个合格 5s 窗口（16 帧，160×160）。
   - 主口径：subject-disjoint 42/9/9（seed 20261105），同一受试者只在一个集合。
   - H3 对照：within-subject subject×class 70/15/15（seed 20261105），60 人
     都出现在三个集合；原 `split` 列未改。
2. **Metrics**
   - 窗口级 Macro-F1（主指标）、每类召回、QWK、Balanced Accuracy。
   - 视频级 Macro-F1（mean probability + majority），用于 CNN 格。
   - subject-level bootstrap 95% CI（1,000 次重采样）。
3. **Subject-disjoint results**
   - RF-Prior 0.453 ± 0.003 是最强基线。
   - 四格 CNN 0.232–0.268；严格负结果；瓶颈在冻结特征。
   - 视频级聚合救不回来；QWK 为负。
4. **Per-class failure mode**
   - CNN 的 Alert 召回 0.079–0.210，RF 为 0.452；先验注入把它提高到
     0.169–0.210，但仍不够。
   - Low-vigilant 对 RF 很难（0.352），对 CNN 反而略好（0.269–0.368）。
5. **Within-subject contrast (H3)**
   - RF 0.453 → 0.816（+0.363）；CNN-Frame 0.232 → 0.9985（+0.766）；
     CNN-Frame-Prior 0.253 → 0.9973（+0.744）。
   - within 饱和是因为共享受试者/视频；只作为虚高证据，不能当部署性能。
   - 正式四格 × 3 seeds within 待蔡回填。
6. **2×2 exploratory**
   - 时序 +0.018 ± 0.011；先验 +0.018 ± 0.025；交互 −0.005 ± 0.023。
   - 效应很小；H1 弱支持，H2 证据更弱；不要过度解读。
7. **Mechanism probe (initial)**
   - FiLM gate 均值 0.689、标准差 0.278；每类均值 0.687–0.691；
     每受试者 0.658–0.717；低饱和 0.2% / 高饱和 4.0%。
   - clean 条件下 gate 不是 class-specific；corruption 探针留到 W14。
8. **Limitations**
   - subject-disjoint test 已被多次评估，只能标 exploratory。
   - 9 人 val 太小；选模型用 42 人 subject-level CV。
   - 冻结 backbone；是否微调留到中期之后决定。
   - 环境差异（torch 2.11 vs 2.14）带来约 0.009 的 test 波动。
9. **Next steps**
   - 蔡：within 四格 × 3 seeds，出正式表。
   - corruption 鲁棒性 + gate 探针（W14）。
   - 中期后决定是否微调 backbone。
   - NTHU-DDD 外部验证，拿一个干净的 test 数字。
