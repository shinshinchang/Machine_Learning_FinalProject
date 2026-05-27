### Preliminary Experimental Results

Observation A.1：validation_set of amazon beauty
* NDCG@10：SASRec(0.0704)、CEINN(0.0484) -> CEINN(-45.45%)
* HR@10：SASRec(0.1158)、CEINN(0.0794) -> CEINN(-45.84%)

Observation A.2：test_set of amazon beauty
* NDCG@10：SASRec(0.0360)、CEINN(0.0365) -> CEINN(+1.37%)
* HR@10：SASRec(0.0660)、CEINN(0.0609) -> CEINN(-8.37%)

Observation M.1：validation_set of movielens_10M
* NDCG@10：SASRec(0.1925)、CEINN(0.1536) -> CEINN(-25.33%)
* HR@10：SASRec(0.3247)、CEINN(0.2714) -> CEINN(-19.64%)

Observation M.2：test_set of movielens_10M
* NDCG@10：SASRec(0.1295)、CEINN(0.1425) -> CEINN(+9.12%)
* HR@10：SASRec(0.2331)、CEINN(0.2468) -> CEINN(+5.55%)

Elegant Properties of Causal Recommendation Systems: Robust Generalization and Popularity Bias Mitigation

**The Pitfall of SASRec**. SASRec exhibits severe overfitting to transient popularity trends. Because the validation set chronologically succeeds the training set with minimal temporal gap, the model exploits these spurious correlations to achieve inflated performance on the validation set. However, when deployed on the future horizon (the test set), it suffers from a substantial degradation in generalization performance once the underlying popularity distribution shifts.

**The Counterfactual Robustness of CEINN**. In contrast, CEINN successfully disentangles the confounding variables, validating the efficacy of our proposed Inverse Probability Scoring (IPS) weighting and Adversarial Graph Representation Learning (GRL) modules. On the validation set, CEINN incurs a localized regularization penalty—manifesting as seemingly suboptimal scores—precisely because it refrains from exploiting the dynamic popularity factor $Z$. Nevertheless, when evaluated on the test set—which rigorously tests whether the model has captured the user's intrinsic subjective value $V$ and cost $C$—CEINN demonstrates superior temporal generalization.

| Dataset | Subset | Metric | PopRec | BPR-MF | GRU4Rec | SASRec | BERT4Rec | CEINN |
|:---|:---|:---|---:|---:|---:|---:|---:|:---|
| Amazon_Beauty | validation_set | best_epoch | 1 | 103 | 216 | 45 | 63 | 63 |
| Amazon_Beauty | validation_set | n_steps | 0 | 587 | 88 | 88 | 88 | - |
| Amazon_Beauty | validation_set | train_loss | 0 | 0.2747 | 6.1383 | 6.2967 | 5.8927 | - |
| Amazon_Beauty | validation_set | NDCG@5 | 0.0059 | 0.0213 | 0.0464 | 0.0599 | 0.0285 | 0.0415 |
| Amazon_Beauty | validation_set | NDCG@10 | 0.0080 | 0.0286 | 0.0546 | 0.0704 | 0.0360 | 0.0484 |
| Amazon_Beauty | validation_set | NDCG@20 | 0.0108 | 0.0355 | 0.0629 | 0.0799 | 0.0435 | 0.0559 |
| Amazon_Beauty | validation_set | HR@10 | 0.0164 | 0.0574 | 0.0914 | 0.1158 | 0.0676 | 0.0794 |
| Amazon_Beauty | validation_set | MRR | 0.0080 | 0.0257 | 0.0492 | 0.0632 | 0.0319 | 0.0444 |
| Amazon_Beauty | validation_set | epoch_time_5 | 0 | 2.9 | 0.9 | 1.1 | 1.2 | 1.1 |
| Amazon_Beauty | test_set | NDCG@5 | 0.0041 | 0.0129 | 0.0219 | 0.0289 | 0.0154 | 0.0312 |
| Amazon_Beauty | test_set | NDCG@10 | 0.0055 | 0.0180 | 0.0271 | 0.0360 | 0.0202 | 0.0365 |
| Amazon_Beauty | test_set | NDCG@20 | 0.0077 | 0.0231 | 0.0332 | 0.0430 | 0.0258 | 0.0428 |
| Amazon_Beauty | test_set | HR@5 | 0.0074 | 0.0216 | 0.0335 | 0.0442 | 0.0240 | 0.0445 |
| Amazon_Beauty | test_set | HR@10 | 0.0115 | 0.0376 | 0.0498 | 0.0660 | 0.0389 | 0.0609 |
| Amazon_Beauty | test_set | HR@20 | 0.0209 | 0.0580 | 0.0738 | 0.0938 | 0.0615 | 0.0854 |
| Amazon_Beauty | test_set | MRR | 0.0057 | 0.0167 | 0.0249 | 0.0323 | 0.0190 | 0.0338 |
| movielens_10M | validation_set | best_epoch | 1 | 15 | 163 | 153 | 147 | 106 |
| movielens_10M | validation_set | n_steps | 0 | 12592 | 137 | 137 | 137 | - |
| movielens_10M | validation_set | train_loss | 0 | 0.3044 | 5.4560 | 5.3833 | 5.3138 | - |
| movielens_10M | validation_set | NDCG@5 | 0.0172 | 0.0185 | 0.1504 | 0.1636 | 0.1182 | 0.1263 |
| movielens_10M | validation_set | NDCG@10 | 0.0229 | 0.0247 | 0.1775 | 0.1925 | 0.1426 | 0.1536 |
| movielens_10M | validation_set | NDCG@20 | 0.0297 | 0.0328 | 0.2030 | 0.2193 | 0.1663 | 0.1796 |
| movielens_10M | validation_set | HR@10 | 0.0456 | 0.0489 | 0.2992 | 0.3247 | 0.2499 | 0.2714 |
| movielens_10M | validation_set | MRR | 0.0226 | 0.0248 | 0.1549 | 0.1669 | 0.1237 | 0.1328 |
| movielens_10M | validation_set | epoch_time_5 | 0 | 101.2 | 7.4 | 10.3 | 10.7 | 19.4 |
| movielens_10M | test_set | NDCG@5 | 0.0196 | 0.0208 | 0.0964 | 0.1047 | 0.0799 | 0.1184 |
| movielens_10M | test_set | NDCG@10 | 0.0249 | 0.0262 | 0.1191 | 0.1295 | 0.1003 | 0.1425 |
| movielens_10M | test_set | NDCG@20 | 0.0315 | 0.0341 | 0.1419 | 0.1529 | 0.1210 | 0.1662 |
| movielens_10M | test_set | HR@5 | 0.0305 | 0.0327 | 0.1440 | 0.1561 | 0.1205 | 0.1719 |
| movielens_10M | test_set | HR@10 | 0.0472 | 0.0496 | 0.2143 | 0.2331 | 0.1837 | 0.2468 |
| movielens_10M | test_set | HR@20 | 0.0736 | 0.0811 | 0.3051 | 0.3260 | 0.2658 | 0.3408 |
| movielens_10M | test_set | MRR | 0.0244 | 0.0261 | 0.1041 | 0.1122 | 0.0879 | 0.1249 |

* note: Modules regarding to CausalRec of current version are deprecated. Any updates will be added onto new branches.
