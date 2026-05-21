---
title: "随机森林算法：新手也能看懂的入门指南"
description: "用直观图示理解随机森林的核心思想、训练流程、优缺点和常见调参方法。"
pubDate: 2026-05-21
tags: ["机器学习", "随机森林", "算法入门"]
draft: false
---

如果你刚开始学习机器学习，随机森林通常是一个很值得先掌握的算法：它不需要太复杂的数据预处理，既能做分类，也能做回归，还常常能给出不错的基准效果。

一句话概括：

> 随机森林就是让很多棵“有点不一样”的决策树一起判断，最后用投票或平均的方式得到更稳定的结果。

<figure>
  <svg viewBox="0 0 920 360" role="img" aria-label="随机森林由多棵决策树投票得到最终预测" style="width:100%;height:auto;border-radius:8px;background:#f7fafc;">
    <rect x="0" y="0" width="920" height="360" fill="#f7fafc"/>
    <text x="460" y="42" text-anchor="middle" font-size="24" font-weight="700" fill="#1f2937">随机森林的直观理解：多棵树一起投票</text>
    <g fill="none" stroke="#94a3b8" stroke-width="3">
      <path d="M170 105 C180 155 195 190 230 230"/>
      <path d="M170 105 C145 155 120 190 88 230"/>
      <path d="M375 105 C385 155 400 190 435 230"/>
      <path d="M375 105 C350 155 325 190 293 230"/>
      <path d="M580 105 C590 155 605 190 640 230"/>
      <path d="M580 105 C555 155 530 190 498 230"/>
    </g>
    <g>
      <circle cx="170" cy="100" r="26" fill="#2563eb"/>
      <circle cx="88" cy="235" r="22" fill="#22c55e"/>
      <circle cx="230" cy="235" r="22" fill="#f97316"/>
      <text x="170" y="308" text-anchor="middle" font-size="18" fill="#111827">树 1：类别 A</text>
    </g>
    <g>
      <circle cx="375" cy="100" r="26" fill="#2563eb"/>
      <circle cx="293" cy="235" r="22" fill="#22c55e"/>
      <circle cx="435" cy="235" r="22" fill="#22c55e"/>
      <text x="375" y="308" text-anchor="middle" font-size="18" fill="#111827">树 2：类别 A</text>
    </g>
    <g>
      <circle cx="580" cy="100" r="26" fill="#2563eb"/>
      <circle cx="498" cy="235" r="22" fill="#f97316"/>
      <circle cx="640" cy="235" r="22" fill="#22c55e"/>
      <text x="580" y="308" text-anchor="middle" font-size="18" fill="#111827">树 3：类别 B</text>
    </g>
    <g>
      <path d="M690 180 H770" stroke="#64748b" stroke-width="4" stroke-linecap="round"/>
      <path d="M755 164 L780 180 L755 196" fill="none" stroke="#64748b" stroke-width="4" stroke-linecap="round" stroke-linejoin="round"/>
      <rect x="790" y="120" width="92" height="120" rx="8" fill="#111827"/>
      <text x="836" y="166" text-anchor="middle" font-size="18" fill="#fff">最终</text>
      <text x="836" y="198" text-anchor="middle" font-size="26" font-weight="700" fill="#86efac">A</text>
    </g>
  </svg>
  <figcaption>单棵树容易“看走眼”，多棵树一起判断通常更稳。</figcaption>
</figure>

## 1. 先理解决策树

随机森林的基本零件是**决策树**。决策树像一套连续提问的规则：

- 年龄是否大于 30？
- 收入是否高于某个阈值？
- 是否有历史购买记录？

每回答一个问题，数据就被分到不同分支，直到到达叶子节点，叶子节点给出预测结果。

决策树的优点是直观、可解释；缺点是容易过拟合。也就是说，一棵树如果长得太深，可能会把训练数据里的偶然噪声也记住，到了新数据上反而表现变差。

## 2. 随机森林为什么更稳？

随机森林通过两个“随机”来制造很多棵不完全相同的树。

第一个随机是**样本随机**：每棵树训练时，不使用完全相同的数据，而是从原始训练集中有放回地抽样。这个过程叫 Bootstrap 抽样。

第二个随机是**特征随机**：每次树在选择分裂条件时，不看全部特征，而是随机抽取一部分特征来挑选最优分裂。

<figure>
  <svg viewBox="0 0 960 420" role="img" aria-label="随机森林训练流程" style="width:100%;height:auto;border-radius:8px;background:#fff7ed;">
    <rect width="960" height="420" fill="#fff7ed"/>
    <text x="480" y="42" text-anchor="middle" font-size="24" font-weight="700" fill="#1f2937">随机森林训练流程</text>
    <g font-size="16" fill="#111827">
      <rect x="48" y="118" width="150" height="88" rx="8" fill="#ffffff" stroke="#fb923c" stroke-width="3"/>
      <text x="123" y="152" text-anchor="middle" font-weight="700">原始数据集</text>
      <text x="123" y="180" text-anchor="middle" fill="#475569">样本 + 特征</text>
      <path d="M210 162 H300" stroke="#ea580c" stroke-width="4" stroke-linecap="round"/>
      <path d="M284 145 L310 162 L284 179" fill="none" stroke="#ea580c" stroke-width="4" stroke-linecap="round" stroke-linejoin="round"/>
      <rect x="318" y="84" width="170" height="70" rx="8" fill="#ffffff" stroke="#fb923c" stroke-width="3"/>
      <text x="403" y="126" text-anchor="middle">Bootstrap 抽样 1</text>
      <rect x="318" y="176" width="170" height="70" rx="8" fill="#ffffff" stroke="#fb923c" stroke-width="3"/>
      <text x="403" y="218" text-anchor="middle">Bootstrap 抽样 2</text>
      <rect x="318" y="268" width="170" height="70" rx="8" fill="#ffffff" stroke="#fb923c" stroke-width="3"/>
      <text x="403" y="310" text-anchor="middle">Bootstrap 抽样 n</text>
      <path d="M502 119 H592" stroke="#ea580c" stroke-width="4" stroke-linecap="round"/>
      <path d="M576 102 L602 119 L576 136" fill="none" stroke="#ea580c" stroke-width="4" stroke-linecap="round" stroke-linejoin="round"/>
      <path d="M502 211 H592" stroke="#ea580c" stroke-width="4" stroke-linecap="round"/>
      <path d="M576 194 L602 211 L576 228" fill="none" stroke="#ea580c" stroke-width="4" stroke-linecap="round" stroke-linejoin="round"/>
      <path d="M502 303 H592" stroke="#ea580c" stroke-width="4" stroke-linecap="round"/>
      <path d="M576 286 L602 303 L576 320" fill="none" stroke="#ea580c" stroke-width="4" stroke-linecap="round" stroke-linejoin="round"/>
      <rect x="610" y="84" width="138" height="70" rx="8" fill="#ffffff" stroke="#2563eb" stroke-width="3"/>
      <text x="679" y="126" text-anchor="middle">决策树 1</text>
      <rect x="610" y="176" width="138" height="70" rx="8" fill="#ffffff" stroke="#2563eb" stroke-width="3"/>
      <text x="679" y="218" text-anchor="middle">决策树 2</text>
      <rect x="610" y="268" width="138" height="70" rx="8" fill="#ffffff" stroke="#2563eb" stroke-width="3"/>
      <text x="679" y="310" text-anchor="middle">决策树 n</text>
      <path d="M760 119 C805 135 820 158 830 198" stroke="#334155" stroke-width="4" fill="none" stroke-linecap="round"/>
      <path d="M760 211 H827" stroke="#334155" stroke-width="4" stroke-linecap="round"/>
      <path d="M760 303 C805 287 820 264 830 224" stroke="#334155" stroke-width="4" fill="none" stroke-linecap="round"/>
      <rect x="834" y="166" width="86" height="96" rx="8" fill="#111827"/>
      <text x="877" y="206" text-anchor="middle" fill="#fff" font-weight="700">投票</text>
      <text x="877" y="234" text-anchor="middle" fill="#fff">或平均</text>
    </g>
  </svg>
  <figcaption>随机森林的核心不是“某一棵树很强”，而是“很多棵差异化的树一起降低误差”。</figcaption>
</figure>

这两个随机性有一个重要作用：降低不同树之间的相似度。树与树越不一样，它们犯同一个错误的概率就越低，整体预测就越稳。

## 3. 分类和回归怎么预测？

随机森林既能做分类，也能做回归。

| 任务类型 | 每棵树输出 | 森林最终输出 |
| --- | --- | --- |
| 分类 | 一个类别 | 多数投票结果 |
| 回归 | 一个数值 | 所有树预测值的平均值 |

例如，一个垃圾邮件识别模型中，100 棵树里有 76 棵认为某封邮件是垃圾邮件，24 棵认为不是，那么随机森林通常会输出“垃圾邮件”。

如果是房价预测，100 棵树分别预测出 100 个价格，最终结果就是这些价格的平均值。

## 4. 它为什么叫“森林”？

因为模型里有很多棵树。单棵树就像一个经验丰富但可能有偏见的人；随机森林像是组织了一个评审团，让不同的人从不同角度判断，最后综合意见。

更正式一点说，随机森林属于**集成学习**。集成学习的思想是：把多个弱一些或中等强度的模型组合起来，得到一个更稳、更强的整体模型。

## 5. 一个小例子：判断客户是否会购买

假设我们有一个电商数据集，特征包括：

- 用户年龄
- 浏览次数
- 是否加入购物车
- 历史购买次数
- 最近一次访问距今多少天

随机森林会训练很多棵树。不同树可能关注不同信息：

- 树 1 更看重“是否加入购物车”
- 树 2 更看重“历史购买次数”
- 树 3 更看重“最近一次访问时间”

最终每棵树给出“会买”或“不会买”的判断，再由森林投票决定。

<figure>
  <svg viewBox="0 0 900 330" role="img" aria-label="特征重要性示意图" style="width:100%;height:auto;border-radius:8px;background:#f8fafc;">
    <rect width="900" height="330" fill="#f8fafc"/>
    <text x="450" y="42" text-anchor="middle" font-size="24" font-weight="700" fill="#1f2937">随机森林还能估计特征重要性</text>
    <g font-size="17" fill="#111827">
      <text x="165" y="104" text-anchor="end">是否加入购物车</text>
      <rect x="190" y="82" width="520" height="32" rx="4" fill="#2563eb"/>
      <text x="725" y="105" fill="#475569">0.34</text>
      <text x="165" y="154" text-anchor="end">历史购买次数</text>
      <rect x="190" y="132" width="405" height="32" rx="4" fill="#22c55e"/>
      <text x="610" y="155" fill="#475569">0.26</text>
      <text x="165" y="204" text-anchor="end">浏览次数</text>
      <rect x="190" y="182" width="295" height="32" rx="4" fill="#f97316"/>
      <text x="500" y="205" fill="#475569">0.19</text>
      <text x="165" y="254" text-anchor="end">最近访问间隔</text>
      <rect x="190" y="232" width="220" height="32" rx="4" fill="#14b8a6"/>
      <text x="425" y="255" fill="#475569">0.14</text>
      <text x="165" y="304" text-anchor="end">年龄</text>
      <rect x="190" y="282" width="110" height="32" rx="4" fill="#64748b"/>
      <text x="315" y="305" fill="#475569">0.07</text>
    </g>
  </svg>
  <figcaption>特征重要性不是绝对真理，但能帮助我们快速发现模型主要依赖哪些信息。</figcaption>
</figure>

## 6. 随机森林的优点

**效果稳定。**  
相比单棵决策树，随机森林通常更不容易过拟合，泛化能力更好。

**使用门槛低。**  
它不要求特征必须标准化，也能处理非线性关系，对新手很友好。

**适用范围广。**  
分类、回归、特征筛选、异常检测等场景中都能看到它。

**能评估特征重要性。**  
训练完成后，我们可以查看哪些特征对预测更有帮助。

## 7. 随机森林的局限

**模型体积可能较大。**  
树越多，模型越大，预测也可能更慢。

**解释性不如单棵树。**  
单棵决策树可以直接画出来看规则，但随机森林由很多树组成，整体解释会更难。

**对高维稀疏数据不一定最优。**  
比如文本分类中常见的高维稀疏特征，线性模型或深度学习方法有时更合适。

**无法自然外推。**  
在回归任务中，随机森林通常不擅长预测训练数据范围之外的趋势。

## 8. 常见参数怎么理解？

如果你使用 Python 的 scikit-learn，常见参数包括：

| 参数 | 含义 | 新手建议 |
| --- | --- | --- |
| `n_estimators` | 森林里有多少棵树 | 常从 100、300、500 试起 |
| `max_depth` | 每棵树最大深度 | 数据少时可限制深度，防止过拟合 |
| `max_features` | 每次分裂可看的特征数量 | 分类常用 `sqrt`，回归可尝试 `1.0` 或 `sqrt` |
| `min_samples_leaf` | 叶子节点最少样本数 | 适当增大可让模型更平滑 |
| `random_state` | 随机种子 | 固定后结果可复现 |
| `n_jobs` | 并行训练数量 | 可设为 `-1` 使用所有可用核心 |

一个简单经验是：先用默认参数训练一个基线模型，再重点调 `n_estimators`、`max_depth`、`max_features` 和 `min_samples_leaf`。

## 9. Python 快速示例

下面是一个分类任务的最小示例：

```python
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

data = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    data.data,
    data.target,
    test_size=0.2,
    random_state=42,
)

model = RandomForestClassifier(
    n_estimators=100,
    max_features="sqrt",
    random_state=42,
)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Feature importance:", model.feature_importances_)
```

这段代码做了四件事：

1. 加载鸢尾花数据集。
2. 把数据分成训练集和测试集。
3. 训练一个随机森林分类器。
4. 在测试集上评估准确率，并查看特征重要性。

## 10. 新手容易踩的坑

**只看训练集分数。**  
训练集分数高不代表模型真的好。一定要看测试集、验证集，或者使用交叉验证。

**树越多不一定越值得。**  
增加树的数量通常能提升稳定性，但到一定程度后收益会变小，训练和预测成本却会继续上升。

**忽略类别不平衡。**  
如果正负样本比例差异很大，准确率可能会骗人。此时要关注召回率、精确率、F1、AUC 等指标，也可以考虑设置 `class_weight`。

**把特征重要性当因果结论。**  
特征重要性只能说明模型在预测时依赖了哪些特征，不代表这些特征一定是结果发生的原因。

## 11. 适合使用随机森林的场景

随机森林很适合用作表格数据任务的强基线模型，比如：

- 用户流失预测
- 信贷风险评估
- 商品销量预测
- 医疗风险初筛
- 设备故障预测
- 特征重要性分析

如果你不确定该从哪个模型开始，随机森林通常是一个稳妥选择。它也常被用来和逻辑回归、XGBoost、LightGBM、神经网络等模型对比。

## 12. 最后总结

随机森林的核心思想可以记成三句话：

1. 用很多棵决策树一起做判断。
2. 每棵树看到的数据和特征都有一点随机差异。
3. 分类靠投票，回归靠平均，从而得到更稳定的预测。

对新手来说，随机森林最大的价值是：它把“容易理解”和“效果稳定”结合得很好。掌握它之后，你会更容易理解后续的集成学习方法，比如 Bagging、Boosting、XGBoost 和 LightGBM。
