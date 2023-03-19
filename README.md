# When is one classifier enough? ICML 2023 submission

## Examples when ensembles fail / The simple bound is actually tight.

Perhaps the simplest bound on the majority-vote classifier, which we introduce as Inequality (1) on the paper, only guarantees that the majority vote classifier is no worse than twice the average error rate.

$$
\mathbb{L}(h_{\mathrm{MV}}) \leq 2\mathbb{E}_h [\mathbb{L}(h)]
$$

While this naive bound that the majority vote classifier is no worse than twice the average error rate looks loose, it is actaully a tight bound. There are  cases where majority vote classifier is actually worse than the average error rate. To see this, we introduce two examples.

#### Example 1 (The upper bound is tight)
Consider a classification problem with two classes. For given $\epsilon > 0$, suppose slightly less than half, $0.5-\epsilon$ fraction of classifiers are the perfect classifier, correctly classifying test data with probability 1, and the other $0.5+\epsilon$ fraction of classifiers are completely wrong, incorrectly predicting on test data with probability 1. With this composition of classifiers, average error rate is $0.5+\epsilon$ and the majority-vote error rate is $1$. Taking $\epsilon \to 0$ concludes that the upper bound is tight. The left plot of the figure below provides a visual illustration of the composition of classifiers.

#### Example 2 (The upper bound is tight _even when_ the average test error is low)
We again consider a classification problem with two classes. For given $\epsilon > 0$, as in Example 1, slightly less than half, $0.5-\epsilon$ fraction of classifiers are the perfect classifier. All of the other $0.5+\epsilon$ fraction of classifiers, on the contrary, now correctly predicts on the same $1-2\delta$ fraction of the test data and incorrectly predicts on the other $2\delta$ fraction of the test data. With this composition of classifiers, the majority-vote error rate is $2\delta$ even when the average error rate is $\delta(1+2\epsilon)$. Taking $\epsilon \to 0$ concludes that the upper bound is also tight **_even when_** the average test error rate is arbitrarily low. The right plot of the figure below provides a visual illustration of the composition of classifiers.

<!---
![figs/First_order_bound_tight_combined.png]
-->

#### Relation to the _competence_ assumption (Assumption 1 on the paper)
These two examples suggest that it is inevitable to take an additional assumption to obtain a better upper bound of the majority vote classifier error rate, $\mathbb{L}(h_{\mathrm{MV}})$. In our paper, we come up with an assumption, which we call _competence_, that guarantees $\mathbb{L}(h_{\mathrm{MV}}) \leq \mathbb{E}_h [\mathbb{L}(h)]$.

## Competence plots

Below we show what we call "competence plots" for a variety of experimental settings, verifying that Assumption 1 is satisfied. To do this, we estimate both $P(W_\rho \in [t,1/2))$ and $P(W_\rho \in [1/2, 1-t])$ on test data. To do this, given a test set of examples $(x_1,y_1),\dots,(x_m,y_m)$ and classifiers $h_1,\dots,h_N$ drawn from $\rho$, we construct the estimator

$$
\widehat{W}_\rho^{(j)} = \frac{1}{N}\sum_{n=1}^N \mathbb{1}(h_n(x_j)\neq y_j)
$$

and calculate $P(W_\rho \in [t,1/2))$ and $P(W_\rho \in [1/2, 1-t])$ from the empirical CDF of 

$$
\widehat{W}_\rho^{(1)},\dots,\widehat{W}_\rho^{(m)}.
$$

![](figs/competence_plots_v4.png)

We observe that the red curves lie above the blue curves for each setting, showing that $P(W_\rho \in [t,1/2)) \geq P(W_\rho \in [1/2, 1-t])$ for all $t\in (0,1/2)$, as required by competence.

## Bayesian versus deep ensembles

Below, we compare the behavior of deep ensembles to Bayesian ensembles on two tasks:

- ResNet20 architectures on the CIFAR-10 task
- A CNN-LSTM architecture on the IMDB task

For the Bayesian ensembles, we use samples provided openly with the paper [Izmailov, 2021], and train deep ensembles (20 models trained from independent initialization) on the same architectures and datasets.

We remark that the ResNet20 models used here are actually significantly smaller than the larger ResNet18 models used in other experiments in the paper (e.g. Figure 3); the ResNet20 model uses a width factor of 16.

![](figs/bayes_vs_deep_ensemble_v2.png)

We make a few interesting observations regarding these results.

- The Bayesian ensembles offer higher ensemble improvement across both tasks, as measured by the EIR, than the deep ensembles -- by a significant margin the case of the IMDB task. 
- This is well-captured by the DER, which we find is above the threshold of 1 for the ResNet20 using both ensembling methods and the CNN-LSTM Bayesian ensemble, and significantly below the threshold of 1 for the CNN-LSTM deep ensemble.
- The average error rate of the Bayesian models is very bad; it's clear that ensembling is necessary when generating individual models in this way (though this is not surprising). 
- It is not definitive whether Bayesian ensembling is beneficial in general. For the CNN-LSTM/IMDB task, the Bayesian ensemble performs better than any one model on average, and better than the deep ensemble. For the ResNet20/CIFAR-10 task, the Bayesian ensemble performs only slightly better than a single model SGD-trained model on average, and significantly worse than a deep ensemble.  

## Fine-tuned BERT models on GLUE tasks

Here, we try a different type of deep ensembling. We use 20 pre-trained BERT models included with the paper [Sellam, 2022], each trained from an independent intialization, and then fine-tuned on the GLUE tasks. Each of the fine-tuned models are used to form an ensemble.

![](figs/glue_dis_avg_ens.png)

In the above plot, we show the disagreement, average error rate, and majority-vote error rates for these ensembles across the 7 GLUE classification tasks. These experiments highlight an important point regarding the disagreement as a metric characterizing ensemble improvement: the disagreement can be nominally very high (e.g. the RTE task, in which it is almost 30%), but see very low improvement from ensembling. However, the DER captures this much better: the disagreement is nominally high, but _small relative to the average error rate_. Indeed, the DER is small ($<1$) for all of these tasks, as is the improvement from ensembling.

## References

[Izmailov, 2021] Pavel Izmailov, Sharad Vikram, Matthew D Hoffman, and Andrew Gordon Gordon Wilson.
What are bayesian neural network posteriors really like? In Marina Meila and Tong Zhang,
editors, Proceedings of the 38th International Conference on Machine Learning, volume 139 of
Proceedings of Machine Learning Research, pages 4629–4640. PMLR, 18–24 Jul 2021.

[Sellam, 2022] Thibault Sellam, Steve Yadlowsky, Ian Tenney, Jason Wei, Naomi Saphra, Alexander D’Amour,
Tal Linzen, Jasmijn Bastings, Iulia Raluca Turc, Jacob Eisenstein, Dipanjan Das, and Ellie
Pavlick. The multiBERTs: BERT reproductions for robustness analysis. In International
Conference on Learning Representations, 2022.
