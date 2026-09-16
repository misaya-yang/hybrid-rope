# RoPE/位置编码论文审稿全文合集（10 篇）

This file concatenates the per-paper Markdown archives listed in `INDEX.md`.


---

# Round and Round We Go! What makes Rotary Positional Encodings useful? — OpenReview 审稿全文归档

- Venue: **ICLR 2025 Poster**
- OpenReview forum: [https://openreview.net/forum?id=GtvuNrk58a](https://openreview.net/forum?id=GtvuNrk58a)
- Official paper page: [https://proceedings.iclr.cc/paper_files/paper/2025/hash/e6d58fc68c0f3c36ae6e0e64478a69c0-Abstract-Conference.html](https://proceedings.iclr.cc/paper_files/paper/2025/hash/e6d58fc68c0f3c36ae6e0e64478a69c0-Abstract-Conference.html)
- Reviewer handles are the public OpenReview pseudonyms; no attempt is made to identify individuals.
- Source: public OpenReview review dump; fields are preserved as released, including review text, rebuttal comments, meta-review, and decision where available.

## Paper Abstract

Positional Encodings (PEs) are a critical component of Transformer-based Large Language Models (LLMs), providing the attention mechanism with important sequence-position information. One of the most popular types of encoding used today in LLMs are Rotary Positional Encodings (RoPE), that rotate the queries and keys based on their relative distance. A common belief is that RoPE is useful because it helps to decay token dependency as relative distance increases. In this work, we argue that this is unlikely to be the core reason. We study the internals of a trained Gemma 7B model to understand how RoPE is being used at a mechanical level. We find that Gemma learns to use RoPE to construct robust `positional' attention patterns by exploiting the highest frequencies. We also find that, in general, Gemma greatly prefers to use the lowest frequencies of RoPE, which we suspect are used to carry semantic information. We mathematically prove interesting behaviours of RoPE and conduct experiments to verify our findings, proposing a modification of RoPE that fixes some highlighted issues and improves performance. We believe that this work represents an interesting step in better understanding PEs in LLMs, which we believe holds crucial value for scaling LLMs to large sizes and context lengths.

## Review Inventory (5 Official Reviews, 23 Discussion/Comment Notes)

- Final decision: **Accept (Poster)**
- Official review ratings (review order): `5, 5, 8, 5, 8`

## Official Reviews

### Official_Review — Reviewer_DtVE

- Note ID: `R38mxyPD8S`
- Discussion number: `1`
- Invitation: `ICLR.cc/2025/Conference/Submission8344/-/Official_Review`


#### Summary

This paper studied the inner workings of rotary positional embedding. The authors started by challenging the common belief that RoPE decays with distance. Then, the authors showed in Gemma 7B that most RoPE usages appear in low frequencies. The authors explained that high frequencies are for positional information while low frequencies are for semantic information. Finally, the authors observed that the low-frequency components are not robust. Based on this observation, the authors proposed p-RoPE to remove (1-p)*100% of the low-frequency component and showed that this improved the performance of Gemma 2B models.

#### Soundness

`3`

#### Presentation

`3`

#### Contribution

`2`

#### Strengths

* The paper is well-written and the key points are delivered clearly.
* The authors made a good point that RoPE doesn't necessarily decay with distance.
* The figures clearly showed that most attentions happen in low frequencies, and some high-frequency heads and frequency bands exist.

#### Weaknesses

* The claim that high-frequency components are for positional information is unclear.
  - The reasoning seems to be: (a) RoPE can learn a diagonal or off-diagonal pattern. (b) NoPE cannot learn a diagonal or off-diagonal pattern. So RoPE can learn positional information. However, it doesn't necessarily mean the high-frequency components contribute to the diagonal/off-diagonal pattern. So the function of high-frequency components remains unclear.
  - Some prior works provide evidence that NoPE can still encode positional information [1][2]. So the fact that NoPE cannot learn a diagonal or off-diagonal pattern may not imply it cannot encode positions.
* The authors experimentally show that truncating the lowest frequencies can help RoPE learn better. However, it is based on a claim that "RoPE lacks robust semantic channels". 
  - This claim comes from an analysis on a 2-dimensional case. However, it is possible that the 2D case is too restrictive and a higher dimensional case could have a different result.

[1] Haviv et. al. "Transformer language models without positional encodings still learn positional information," EMNLP 2022 Findings.
[2] Chi et. al. "Latent positional information is in the self-attention variance of transformer language models without positional embeddings," ACL 2023.

#### Questions

* Is your p-RoPE having a similar flavor as the partial rotary?
  - Partial rotary means the whole dimension is divided into rotary part and non-rotary part. The rotary part will be treated by RoPE while the non-rotary part is treated by NoPE.
  - Partial rotary embedding has been known by the community for a while (see https://github.com/lucidrains/x-transformers/issues/40). It is known that the partial rotary is slightly better than rotary positional embedding.
  - Because the proposed p-RoPE is interpolating between NoPE (p=0) and RoPE (p=1), it is possible that p-RoPE is similar to partial rotary.

#### Flag For Ethics Review

- No ethics review needed.

#### Rating

`5`

#### Confidence

`3`

#### Code Of Conduct

Yes

### Official_Review — Reviewer_7eRV

- Note ID: `stJiuAMm0w`
- Discussion number: `2`
- Invitation: `ICLR.cc/2025/Conference/Submission8344/-/Official_Review`


#### Summary

The paper provides practical insights into positional encoding for decoder-only models. However, further investigation is needed to establish the effectiveness and reliability of the theories presented.

#### Soundness

`2`

#### Presentation

`3`

#### Contribution

`1`

#### Strengths

Pros:

1. The paper offers a fresh discovery of Rotary Positional Encoding (RoPE), challenging the belief that it primarily helps decay attention coefficients with distance.

2. The paper gives an analysis of high and low RoPE frequencies, their roles in positional and semantic attention, and an innovative RoPE modification (p-RoPE) that demonstrates improvements in some cases.

3. Every section has a summary part which makes the whole paper clear and ready to read.

#### Weaknesses

Cons:

1. The figures in the paper are somewhat confusing. For instance, while the paper emphasizes frequency aspects, Figure 1 illustrates vectors with identical frequency differences, which does not fully align with the paper’s focus. Additionally, the results are primarily presented using heat maps, which may appear monotonous and lack of expressiveness. Given that this is a language modeling task, presenting some results in natural language format would enhance readability and interpretability.

2. (MAIN LIMITATION) The experiments are limited to decoder-only models, making it unclear whether the findings can generalize to other transformer architectures. Whether the observed improvements stem from specific model structures (e.g., encoder-only or encoder-decoder models) rather than from a comprehensive enhancement applicable to diverse positional encodings of transformer. Additionally, the authors did not consider the impact of long-range dependencies or how the choice of positional encoding influences masked attention performance of decoder-only model. It would be valuable if the authors could report comparative results using unmasked attention (e.g., with encoder-only transformers) to examine these effects.

3. (MAIN LIMITATION) The proposed modification p-RoPE appears to primarily integrate elements of RoPE and NoPE, which may lack novelty. While the study advances understanding of RoPE, its applicability to other positional encoding methods, such as Alibi, remains limited, potentially constraining its relevance for models with alternative encoding schemes. The authors should consider incorporating other positional encoding methods and propose more innovative improvements to enhance the usage of lower and higher frequencies.

4. The paper states that higher frequencies correspond to positional attention, while lower frequencies correspond to semantic attention. However, the authors provide an ablation study only for the lower frequencies, without addressing both low and high frequencies. This aspect is insufficiently explained in the ablation experiments.

#### Questions

Why did the authors provide an ablation study only for the lower frequencies rather than for both low and high frequencies?

#### Flag For Ethics Review

- No ethics review needed.

#### Rating

`5`

#### Confidence

`4`

#### Code Of Conduct

Yes

### Official_Review — Reviewer_rv9e

- Note ID: `xIZr2627Lb`
- Discussion number: `3`
- Invitation: `ICLR.cc/2025/Conference/Submission8344/-/Official_Review`


#### Summary

Authors explore how transformer models use RoPE frequencies to learn semantic and positional information. Authors provide a theoretical proof that RoPE does not force the decay in attention coefficient with distance, but instead can create specific patterns. In their experimental study of the Gemma 7B model, authors show that transformers mostly rely on low frequencies, while some heads display high frequency bands, mostly in 1st and last layers of the model. Authors further show that high-frequencies in RoPE provide a mechanism to encode positional information. Low frequencies are used as information channels that are not robust over long context. Finally, authors propose p-RoPE encoding that cuts low frequencies and can help to improve model's performance.

#### Soundness

`3`

#### Presentation

`3`

#### Contribution

`4`

#### Strengths

- The authors conduct a novel theoretical and empirical study of RoPE encodings in transformer models.
- They provide detailed proofs of their main claims.
- The paper is clear and well-written.
- This study can help researchers better understand the underlying mechanisms of popular transformer architectures and encourage research into alternative improved solutions.

#### Weaknesses

- The empirical study is limited to a single Gemma architecture. While this is unlikely, some results may be artifacts of the specific model selected.
- In Section 3 authors train 2B model and show improvements on validation perplexity. While these results are positive, perplexity improvements do not always results in overall improvements in model's abilities. Authors could provide evaluation results on popular benchmarks* to build more convincing picture.


* see, for ex, Section 2.3 in https://arxiv.org/pdf/2307.09288

#### Questions

1. Pre-training model from scratch is expensive and not always feasible. I was wondering if you can instead continuously train other (Gemma-7B, Llama-3 1/3/8B, etc) models for fewer steps but with p-RoPE approach? Do you think it would work or no and why?

2. Please, correct me if I'm wrong, but in the proof of Proposition 3.1:
for k>1, g_k=theta^{-2(k-1)/d} is not necessarily rational, but algebraic. Therefore, Lemma A.1 should be stated not for rational g, but for algebraic g, and it should use the fact that pi is transcendental.
line 680, cos(j-i-r) -> cos(j-i+r)
line 680, the comma in the displayed equation should be a period.
line 684, cos(j-i-r) -> cos(j-i+r)

#### Flag For Ethics Review

- No ethics review needed.

#### Rating

`8`

#### Confidence

`4`

#### Code Of Conduct

Yes

### Official_Review — Reviewer_eLgp

- Note ID: `Bra9GES0eY`
- Discussion number: `4`
- Invitation: `ICLR.cc/2025/Conference/Submission8344/-/Official_Review`


#### Summary

This paper investigates the role of Rotary Positional Encodings (RoPE) in Transformer-based Large Language Models (LLMs). The authors challenge the common belief that RoPE's usefulness comes from its ability to decay token dependency with increasing relative distance. Instead, they explore how different frequencies in RoPE are utilized, particularly within the Gemma 7B model. The paper provides both theoretical and empirical analyses, proposes a modified RoPE, and highlights the importance of understanding positional encodings for scaling LLMs.

#### Soundness

`2`

#### Presentation

`3`

#### Contribution

`3`

#### Strengths

1. The paper provides a fresh perspective on RoPE, questioning existing assumptions and offering new explanations for its effectiveness.
2. The authors present mathematical proofs to support their claims, enhancing the credibility of their findings.
3. The use of the Gemma 7B model for empirical analysis adds practical relevance to the theoretical insights.

#### Weaknesses

1. Although the observed phenomena and mathematical proofs can support the paper's point of view, the experimental performance does not seem good enough. The paper hopes to adapt to any context length, but the actual experimental results only have one result on 8K. And the evaluation of PPL is not comprehensive enough.
2. At the semantic level, the results of the models in Table 2 should be compared on the general benchmark or other tasks that are more representative of semantics, which will be more convincing.
3. At the positional level, it should be compared with similar experiments such as needle in a haystack or Ruler on long contexts to prove its long context expansion ability.

#### Questions

1. I have some doubts about the display of Figure 2. Normally, for the same qk, different relative distances should have a gradually decreasing effect. But does different qk introduce different variables, making the comparison unfair?
2. Is there any empirical experiment on how many frequencies are cut off for the best effect?
3. Will removing low-frequency RoPE make the effect on short text worse?

#### Flag For Ethics Review

- No ethics review needed.

#### Rating

`5`

#### Confidence

`4`

#### Code Of Conduct

Yes

### Official_Review — Reviewer_T6HH

- Note ID: `raH0YWgspE`
- Discussion number: `5`
- Invitation: `ICLR.cc/2025/Conference/Submission8344/-/Official_Review`


#### Summary

This paper delves into the role of Position Embedding (PE) in LLMs, challenging the traditional view that RoPE primarily attenuates attention weights as the relative distance between words increases. It proposes a new hypothesis that RoPE constructs position-attention patterns (e.g., diagonal or previous-token focus) using high-frequency components while leveraging low frequencies to convey semantic information. Several case studies and theoretical analyses support this hypothesis.

Overall, the paper provides insightful findings and hypotheses on a key component of LLMs: position encoding. However, the arguments mainly rely on case studies with gemma-7B, and the experiments lack diversity in the foundation models (e.g., missing LLaMA series) and tasks (e.g., language model vs. QA vs. code...). Additionally, some parts of the proofs contain errors.

#### Soundness

`3`

#### Presentation

`3`

#### Contribution

`3`

#### Strengths

1. This paper successfully challenges, both theoretically and empirically, the traditional view that RoPE attenuates attention weights as relative distance between tokens increases. 
2. The authors' hypothesis about the roles of the high-frequency and low-frequency components of RoPE is novel and insightful.

#### Weaknesses

1. The experimental validation lacks diversity in both foundation models and datasets.
2. Some perspectives and proofs regarding NoPE contain errors, while they don't affect the main conclusion, they may mislead readers.
3. The discussion of related work is insufficiently thorough, and few papers are cited (only about one page).
4. Throughout Section 4, the authors conceal a core assumption: that attention scores are interpretable or meaningful. Higher attention scores for certain tokens imply a meaningful preference in the model, giving special significance to the larger norms in Equation at Line 257. I want to point out that this assumption is still being debated, and I suggest that the authors make it explicit.
> Is Attention Explanation? An Introduction to the Debate (Bibal et al., ACL 2022)

#### Questions

1. In practice, do the phenomena observed on gemma-7b apply to other LLM backbones, such as llama-2 or llama-3?
2. The phenomena observed in this paper are based on what scale and type of data? Could the findings be validated across various tasks, such as language modeling, code, or QA?
3. In Line 190, the authors claim that NoPE has strong OOD capabilities, but Kazemnejad et al., 2024 only validated this for sequences of length up to 50, where NoPE performed slightly better than other PEs, without exhibiting exceptionally strong OOD performance (refer to their Figure 3). The following works explored NoPE's extrapolation, and it can be seen that the generalization ability of the NoPE baseline is limited:
> Length Generalization of Causal Transformers without Position Encoding (Wang et al., Findings 2024)
> 
> [Neurips24 spotlight] Exploring Context Window of Large Language Models via Decomposed Positional Vectors
4. In Line 497, the authors state, "p-RoPE is in spirit similar to the idea of increasing the wavelength of RoPE from 10,000 to 500,000," so I suggest that Table 2 should include RoPE with a base of 500,000.
5. How do the observations in Section 4 change across different RoPE variations, such as 0.25-RoPE, 0.75-RoPE, RoPE_10000, and RoPE_500000? For example, in Figure 3, how does the norm distribution of low frequencies vary across these models?
6. In Line 726, this proof only applies to the first layer of NoPE’s attention heads. Starting from the second layer, the assumption 
$a_{3,3}=<q_3, k_3>=<q_3, k_2>=a_{3,2}$ 
no longer holds.
7. Attention heads with specific patterns have been widely studied. Besides the patterns discussed in this paper (e.g., diagonal or previous-token focus), could the authors observe and analyze other representative attention patterns, such as special token focus, punctuation focus, and locality focus? Please refer to typical patterns in the following paper:
> [ICLR24] Model Tells You What to Discard: Adaptive KV Cache Compression for LLMs

#### Flag For Ethics Review

- No ethics review needed.

#### Rating

`8`

#### Confidence

`4`

#### Code Of Conduct

Yes

## Meta-Review

### Meta_Review — Area_Chair_RdPT

- Note ID: `iSH3ZBnZXn`
- Discussion number: `1`
- Invitation: `ICLR.cc/2025/Conference/Submission8344/-/Meta_Review`


#### Metareview

**Summary:** This paper provides a theoretical and empirical study of Rotary Positional Encodings (RoPE) in transformer-based LLMs. The authors challenge the conventional belief that RoPE facilitates token dependency decay with increasing distance. Instead, the work hypothesizes that RoPE's utility lies in its ability to construct robust positional attention patterns using high frequencies, while low frequencies encode semantic information. The paper validates these hypotheses through mathematical proofs and extensive experiments, including a novel modification of RoPE, named p-RoPE, which demonstrates performance improvements in specific settings. The study advances the community's understanding of positional encodings and proposes actionable insights for future model design.

**Decision:** The paper makes a significant theoretical and empirical contribution by deepening our understanding of RoPE and proposing meaningful modifications. Despite some concerns regarding the experimental breadth (e.g., limited evaluation on downstream tasks and reliance on perplexity as the primary metric), the added ablations and discussions during the rebuttal phase strengthened the case for the paper’s claims. Reviewer 7eRV raised concerns regarding novelty and limited to decoder-only models. However, based on the prevailing techniques in this field, I do not consider these to be weaknesses of this work. 

Overall, the contributions of this paper outweigh its shortcomings, and I recommend its acceptance. I encourage the authors to incorporate the reviewers' feedback and the additional content provided in the rebuttal into the final version to further enhance the quality of the paper.

#### Additional Comments On Reviewer Discussion

The discussion highlighted the paper's strong theoretical contributions and novel insights into RoPE mechanisms, which were well-supported by proofs and experiments. Reviewers appreciated the added ablations, such as analysis across different frequency ranges and results on Llama 3.1 8B, which demonstrated generality. However, some concerns about limited experimental diversity and reliance on perplexity as the primary metric for evaluation remained partially unresolved. Despite this, the reviewers largely agreed that the paper provides valuable understanding of positional encodings and is a significant contribution to the field.

## Rebuttal and Discussion Comments

### Official_Comment — Authors

- Note ID: `LC1Y2BeOAK`
- Discussion number: `1`
- Invitation: `ICLR.cc/2025/Conference/Submission8344/-/Official_Comment`


#### Comment

We thank you for your very interesting questions. We would like to address your points in full.

**Figure 1 illustrates vectors with identical frequency differences, which does not fully align with the paper’s focus[...] heat maps, which may appear monotonous and lack of expressivenes**

We believe that Figure 1 illustrates nicely one of the main mechanisms which we present in Proposition 3.1 and later use in Theorem 5.3. The figure depicts a single frequency as we believe that visualising multiple frequencies might be too cluttered and detract from the main point of the construction. To address this point we have amended the caption to clarify that the figure is focusing on a single frequency of RoPE as we agree with you that this might otherwise not be clear. 

Regarding the heat maps, we believe that they offer a very clear way to visualise our arguments. We also would like to point out that these types of heatmaps are very common in these types of visualisations [1, 2]. As we are mostly interested in visualising activations when we *take a mean* over different sequences, there is often no clear 1:1 correspondence to natural language. We would be happy to accommodate your request and modify our visualisations if you could point out which specific figure(s) you are referring to and how you would improve their message. 

**The experiments are limited to decoder-only models...**

In this work we focus on Large Language Models such as Gemma and Llama that generate text in an auto-regressive manner. These language models do not use encoder-Transformers. As such, it is out of scope of this work to study encoder Transformers. We are further unaware of open-sourced and pre-trained LLMs that use an encoder architecture and use RoPE. We hope that given our comments convince you that this is not in fact a limitation of our work. Finally, it is also quite common for works to focus solely on studying decoder Transformers e.g. [3, 4]. We have made this more clear in the text. 

**...While the study advances understanding of RoPE, its applicability to other positional encoding methods, such as Alibi, remains limited, potentially constraining its relevance for models with alternative encoding schemes...**

We are happy to read that you believe our study advances the understanding of RoPE. This is in fact the main goal of our work. As far as we are aware RoPE is by far the most popular positional encoding used in the training of autoregressive LLMs today – used in Gemma, Llama, and many others. For this reason we choose to focus only on understanding RoPE in this paper. Alibi is definitely a very interesting technique, but as far as we are aware much less used in frontier LLMs and mathematically very different. For these reasons, Alibi falls outside the scope of this work. We do however comment in the Appendix (Section B.2) on how we believe our work applies to different positional encodings.

**The paper states that higher frequencies correspond to positional attention, while lower frequencies correspond to semantic attention. However, the authors provide an ablation study only for the lower frequencies, without addressing both low and high frequencies. This aspect is insufficiently explained in the ablation experiments.**

We thank you for the interesting suggestion. We have added this as an ablation in Table 2 and are happy to report that removing the lowest frequencies indeed seems to outperform by a significant margin the removal of the highest frequencies. We hope that you find this result interesting.

We sincerely thank you for your review and valuable comments. We hope that in light of our revisions (see general comment) and our response and additional ablations that you agree that our work has now improved. We would be grateful if you could consider upgrading your score under this new light. We are more than happy to answer any further questions during the rebuttal period.

[1] Randomized Positional Encodings Boost Length Generalization of Transformers. ACL 2023. 

[2] Penzai + Treescope: A Toolkit for Interpreting, Visualizing, and Editing Models As Data. Johnson. Arxiv 2024

[3] Transformers need glasses! Information over-squashing in language tasks. NeurIPS 2024. Barbero et al.

[4] The expressive power of Transformers with Chain of Thought. ICLR 2024. Merrill et al.

### Official_Comment — Authors

- Note ID: `5eK6UV21wc`
- Discussion number: `2`
- Invitation: `ICLR.cc/2025/Conference/Submission8344/-/Official_Comment`


#### Comment

We are happy to hear that you found our work well-written and clear. We are also happy that you appreciated one of our important claims in the paper which provides evidence against the common claim present in important works that RoPE helps to decay with distance the signal. We would like to answer your questions and comments.

**The claim that high-frequency components are for positional information is unclear...However, it doesn’t necessarily mean the high-frequency components contribute to the off-diagonal pattern. So the function of high-frequency components remains unclear.**


We thank you for the great comment. In our paper we mathematically explain why the highest frequencies are the most effective in constructing the positional attention patterns, but indeed we do not make claims that they are *necessary*. What we instead claim is that given some fixed budget of vector “norm” the highest frequencies are precisely the most useful to have the sharpest patterns. For the details see the paragraph below Theorem 5.3 which explains exactly why the highest frequencies are the most useful when constructing these *sharp* attention patterns. Finally, we also left a discussion of this in the Appendix E.1. 

In practice, we believe that we provide ample evidence for this behaviour in our work – see e.g. Figure 14, Figure 15, and Figure 16, where we show how the heads that are diagonal and off-diagonal have very high frequency usage. The qualitative difference can be contrasted with Figure 17 and Figure 18 in which this high frequency usage is very clearly not as present. In Figure 4 this perhaps is even more clear when looking at a single layer: the two heads that mostly use the highest frequencies are heads 5 and 8 which correspond to a diagonal and off-diagonal head respectively. *We also validated this behaviour in Llama in our new revision in Figure 11*. We hope that this new figure in particular can help to convince you that this is indeed a wide-spread mechanism. We are happy to provide more evidence if you believe more is necessary, but we believe there is already a sufficient amount of evidence in the work (both theoretical and empirical). 

**Some prior works provide evidence that NoPE can still encode positional information [1][2]. So the fact that NoPE cannot learn a diagonal or off-diagonal pattern may not imply it cannot encode positions.**

We agree with this point and in fact we do not make claims that NoPE is unable to encode positional information – we have clarified this in the paper. Our claim is instead that RoPE provides an *efficient* way to construct sharp diagonal or off-diagonal attention patterns (more generally positional attention patterns) through the use of the highest frequencies in particular. The difference between works such as [1, 2] is that we prove that specifically an attention head in isolation with NoPE is unable to learn these specific attention patterns, but we do not make any claims on what a deep model is able to learn. We however believe that studying what a single attention head can do is important because of an argument of efficiency: RoPE allows an attention head to implement something that would instead require more than one layer or component to achieve otherwise without it. 

**This claim comes from an analysis on a 2-dimensional case. However, it is possible that the 2D case is too restrictive and a higher dimensional case could have a different result.**

We believe this 2-dimensional case to be quite applicable because as we show in our experiments, we find that the large norms often focus on distinct 2-dimensional bands. As such, we have indicative evidence that our theorem applies in the practical situations we study over several potent pre-trained LLMs. Of course, proving the more general case would be very interesting, but we believe this to be mathematically quite challenging – although we do not have reason to believe this to be untrue in the general case.

### Official_Comment — Authors

- Note ID: `fvJs5Z9QzW`
- Discussion number: `3`
- Invitation: `ICLR.cc/2025/Conference/Submission8344/-/Official_Comment`


#### Comment

**Is your p-RoPE having a similar flavor as the partial rotary?**

We thank you for pointing out this github issue, we have added this to our work to show that there have been similar ideas. We were unaware of this at the time of writing the paper. p-RoPE is indeed similar in nature to applying this to a part of the embedding, but of course is still different as the partial rotary encoding will still keep the lowest frequencies. We are very happy however to see that this has been tried already as this validates our findings. 

We believe that our work however importantly explains why p-RoPE or partial rotary encodings can work, allowing for the creation of robust semantic channels. We finally note that our proposal of p-RoPE is a small part of our overall work and in our opinion just a clear consequence of our findings. 

We thank you very much for your interesting points. We hope that you appreciate the additional ablations and results in the general comment and that our replies have helped to clarify your doubts. As you are the reviewer with the lowest score for our work, we would truly appreciate it if you could reconsider your opinion of our work given our comments and additional experiments and discussions. We hope that you agree with us that this work provides interesting and novel insights of RoPE, which would be very valuable for the community as a whole. We are of course very happy to answer any further questions.

### Official_Comment — Authors

- Note ID: `t8HvczEj2j`
- Discussion number: `4`
- Invitation: `ICLR.cc/2025/Conference/Submission8344/-/Official_Comment`


#### Comment

We are very happy to see you have enjoyed our work. We are particularly happy to read that you believe “this study can help researchers better understand the mechanism of popular transformer architectures” – which is exactly the point of this paper. 

**The empirical study is limited to a single Gemma architecture. While this is unlikely, some results may be artifacts of the specific model selected.**

We thank you for this comment. We are excited to have now added results with Llama3.1 8B which show similar patterns. Interestingly, Llama3.1 8B has a 500k wavelength parameter and Grouped Query Attention. We find that the findings are very similar to Gemma, helping to support the generality of our work. The results can be found in the Appendix (Section C).

**In Section 3 authors train 2B model and show improvements on validation perplexity. While these results are positive, perplexity improvements do not always results in overall improvements in model's abilities. Authors could provide evaluation results on popular benchmarks to build more convincing picture.**

We have spent some time to add additional ablations to answer other reviewers which we hope you will find valuable. We agree that our evaluation of p-RoPE could be more extensive. The focus of the paper was not that of proposing a new type of positional encoding, but rather that of understanding RoPE. We found that our analysis immediately translated to improvements such as p-RoPE, but we saw p-RoPE more as an interesting ablation to verify our intuition. We are of course happy to see that you have already pointed out that our work helps to better understand RoPE. 

**Pre-training model from scratch is expensive and not always feasible. I was wondering if you can instead continuously train other (Gemma-7B, Llama-3 1/3/8B, etc) models for fewer steps but with p-RoPE approach? Do you think it would work or no and why?**

This is a very interesting point and something which we found to be possible – although we found that performance was better when training from scratch. We believe it is possible as cutting off the lowest frequency rotations provides the “least amount of change” when re-computing the activations.

**Please, correct me if I'm wrong, but in the proof of Proposition 3.1: for k>1, g_k=theta^{-2(k-1)/d} is not necessarily rational, but algebraic. Therefore, Lemma A.1 should be stated not for rational g, but for algebraic g, and it should use the fact that pi is transcendental. line 680, cos(j-i-r) -> cos(j-i+r) line 680, the comma in the displayed equation should be a period. line 684, cos(j-i-r) -> cos(j-i+r)**

Yes you are exactly right. Thanks for catching this and for carefully checking our proofs! We have corrected this. 

We thank you again for endorsing our work and we hope that our additional results on Llama help to increase your confidence in the generality of our results. We are of course more than happy to answer any further questions.

### Official_Comment — Authors

- Note ID: `ko8gY1UTXS`
- Discussion number: `5`
- Invitation: `ICLR.cc/2025/Conference/Submission8344/-/Official_Comment`


#### Comment

Thanks for your excellent points! We would like to address your comments below.

**The paper hopes to adapt to any context length, but the actual experimental results only have one result on 8K. And the evaluation of PPL is not comprehensive enough. ...**

Thanks so much for the great points regarding how to improve our experimental evaluation. We invested time in improving our evaluation of p-RoPE and have added 2 new ablations, which have also been requested by other reviewers. 

We completely agree with your point that the experimental evaluation could be more extensive, but we would like to stress that the point of this work is not that of proposing a new positional encoding, but rather to better understand RoPE. As such we saw the proposal of p-RoPE as a way great way to validate the intuition we develop in our paper. We believe that the 2 new ablations we provide help to support such a claim. 

In other words, while we offer p-RoPE as a practical solution, this is not the main focus of the paper and is only there to validate experimentally our reasoning in Section 6 that the low frequency channels could be removed as they are not robust. We however also believe that our improved experimental section is still rather interesting as we train 2B parameter Gemma models, showing improvements in perplexity. Of course we agree with you that perplexity has its limitations as well and we have covered this in the Appendix (Section B.5).

We are happy to overall see that you seem to appreciate our contributions towards the understanding of RoPE and kindly request you to evaluate our work on whether it provides a better understanding and novel insights on why RoPE is useful. We believe that addressing common misconceptions such as the claim that RoPE helps to decay activations with distance to be important – as pointed out also for example by Reviewer rv9e in their strengths. Many works tend to in fact propose new positional encodings, but very few if any attempt to truly understand why the existing widespread positional encodings we have to date work so well. 

**I have some doubts about the display of Figure 2. Normally, for the same qk, different relative distances should have a gradually decreasing effect. But does different qk introduce different variables, making the comparison unfair?**

We thank you for the great point. We agree with your perspective and we have added in the Appendix (Section B.4 – see Figure 9) another synthetic experiment where the queries and keys are “constant” from a Gaussian – i.e. we sample a *single* query and a *single* key from a Gaussian and then repeat them up to the sequence length. It is clear also in this case that there is no clear decay. We hope that this addresses your point.
 
**Is there any empirical experiment on how many frequencies are cut off for the best effect?**

In our experiments, we found a value of 25% cutoff (0.75-RoPE) to be the best performing. We report in Table 2 results for a cutoff of 0.25 and 0.75, of course we believe that a finer grid search is likely to yield even better results. Please also compare our ablation with 0.75-RoPE_{reversed} in Table 2, where we cut-off the highest frequencies. We show that this does not work as well, aligning with the intuition derived in our work. 

**Will removing low-frequency RoPE make the effect on short text worse?**

Thanks for the interesting question. We have no reason to expect shorter texts to fare worse given our experiments. The datasets we use include a number of short documents with less than $1,000$ tokens.

We would like to thank you for your excellent comments. As most focused on our p-RoPE experiments, we hope that you could appreciate that we believe most of the contributions are actually towards the understanding of RoPE, rather than the proposal of a new type of PE. In this light, we hope that, with our new ablations and the additions we mention in our global comment, that you kindly consider upgrading your score. We are more than happy to keep engaging if you have further questions.

### Official_Comment — Authors

- Note ID: `Q56kfA68NJ`
- Discussion number: `6`
- Invitation: `ICLR.cc/2025/Conference/Submission8344/-/Official_Comment`


#### Comment

Thank you so much for your thorough review and the great suggestions!

We address W3 and W4 first as W1 and W2 are also repeated in the questions. 

**W3 The discussion of related work is insufficiently thorough**

We agree with your point that our related work discussion was a bit limited. We have added a new section in the Appendix (Section B.1) with a pointer in the main text covering a large number of works. We are of course happy to include works you believe we may have missed and hope that our additional literature review addresses your concern. 

**W4 ... the authors conceal a core assumption: that attention scores are interpretable or meaningful.  I want to point out that this assumption is still being debated, and I suggest that the authors make it explicit ...**

We thank you for the additional reference, we have added this as a note in the main text.

**Q1 In practice, do the phenomena observed on gemma-7b apply to other LLM backbones, such as llama-2 or llama-3?**

Thanks for the great comment, we agree that this was missing in the original version of the manuscript. We are excited to have added new results on Llama3.1 8B, showing similar results to Gemma. This new section can be found in the Appendix (Section C). Importantly the Llama model uses a different wavelength of 500k and grouped query attention – leading to very interesting insights which we discuss in the appendix and allowing us to further support our claims. In particular, we find that Llama still prefers the lower frequencies, but now leverages a greater spread of them due to the increased max wavelength. We also find diagonal attention patterns being constructed through the highest frequencies. We hope that you find these new additions valuable. 

**Q2 The phenomena observed in this paper are based on what scale and type of data? Could the findings be validated across various tasks, such as language modeling, code, or QA?**

Our investigations rely on the publicly available pretrained Gemma models (and now also Llama) that are trained on a very large corpus. We believe the details of the training data are not public but for instance in Gemma they report “data from web documents, mathematics, and code” [1]. Llama is also trained on “5% multilingual data” [2]. As such we believe our findings to be rather broad due to the scale, breadth, and amount of training present in these models. 

**Q3 In Line 190, the authors claim that NoPE has strong OOD capabilities, but Kazemnejad et al., 2024 only validated this for sequences of length up to 50...**

We thank you for the additional references, we have now included these as a disclaimer in the main text. We would like to clarify that we do not make claims about NoPE being the best method to generalise to OOD, but simply that NoPE provides a mechanism to construct certain attention patterns in a way that is perfectly robust to relative distance – as by construction NoPE is invariant to relative distance. An example of this is attending to the BOS token robustly, which we have now added a more detailed discussion in the Appendix (Section E.1).

**Q4 In Line 497, the authors state, "p-RoPE is in spirit similar to the idea of increasing the wavelength of RoPE from 10,000 to 500,000," so I suggest that Table 2 should include RoPE with a base of 500,000.**

We agree that this ablation would be valuable in our work. We have now added the results with a base of 500k in Table 2 and are happy to report that p-RoPE outperforms this baseline as well. 

**Q5 How do the observations in Section 4 change across different RoPE variations, such as 0.25-RoPE, 0.75-RoPE, RoPE_10000, and RoPE_500000? For example, in Figure 3, how does the norm distribution of low frequencies vary across these models?**

This is a very interesting question. With the new Llama model, we now are able to see the effect of increasing the wavelength to 500k. In particular, it is precisely what we predicted given our investigation. When comparing Figure 10 (Llama 500k wavelength) and Figure 13 (Gemma 10k wavelength), we see that Gemma is much more limited to the very lowest frequencies, while the increased wavelength allows the model to use many more frequencies in Llama. We have provided a much more detailed discussion in the Appendix (Section C) on why we believe this is the case. Overall, we believe these new results to heavily back up our findings and thank you for the suggestion on this comparison. 

Due to logistical reasons, it is hard for us to perform the same analysis on the models we trained ourselves, but we hope that you can appreciate the comparison we have now added between Gemma at 10k and Llama at 500k. We also believe that these results are likely to be more interesting as these models have of course been trained for much longer.

### Official_Comment — Authors

- Note ID: `w9ccIN0PVN`
- Discussion number: `7`
- Invitation: `ICLR.cc/2025/Conference/Submission8344/-/Official_Comment`


#### Comment

**Q6 In Line 726, this proof only applies to the first layer of NoPE’s attention heads. Starting from the second layer, the assumption  a3,3=<q3,k3>=<q3,k2>=a3,2 no longer holds.**

We thank you for checking our proof. We would like to clarify that this is not an error in our proof. We specifically study what can be implemented in a *single* attention head in isolation and thus in a single layer. The motivation for this is that we care about studying the “efficiency” of a single attention head – i.e. what can a single head by itself implement. For instance, in Gemma 7B one can find previous-token or diagonal heads immediately in the first layer, and our proof shows that these heads would be impossible to implement if one simply used NoPE. This means that RoPE provides additional mechanisms over NoPE to construct heads that the Transformer is finding useful during the learning process. We have made this more clear in the paper as we appreciate that this detail might be missed.

**Q7 Attention heads with specific patterns have been widely studied. Besides the patterns discussed in this paper (e.g., diagonal or previous-token focus), could the authors observe and analyze other representative attention patterns...**

We thank you for the additional reference, which we now cite. We have added a more detailed analysis (Appendix, Section E.1) of the Apostrophe head which is both a “punctuation head” and a “special token head” as it either attends to the BOS “special” token or a punctuation token. We believe this analysis to be rather interesting and likely the first of its kind: reverse engineering the mechanism through which RoPE allows the head to be constructed. 

Further, we would like to point to Figure 18 for an example of a different type of head not added in the main text. We found this to be rather representative of a number of heads, with the high frequencies relatively inactive and “high norm” low frequency bands. We are happy to include more examples if you believe this would be useful to the work, but we believe to already have a large breadth of examples and would prefer to avoid adding too many, in order to preserve clarity and clarity.

We would like to sincerely thank you for your very thorough review and the excellent suggestions, which we believe have helped to strengthen our work. We would be grateful if you could consider upgrading your score to our work if you are satisfied with our answers. Of course we are very happy to answer any further questions. 

[1] Gemma: Open Models Based on Gemini Research and Technology. Gemma Team, 2024.

[2] The Llama 3 Herd of Models. Meta, 2024.

### Official_Comment — Authors — General Comment

- Note ID: `CDmM1GgrPy`
- Discussion number: `8`
- Invitation: `ICLR.cc/2025/Conference/Submission8344/-/Official_Comment`


#### Title

General Comment

#### Comment

We would like to thank the reviewers for their efforts in assessing our work and the valuable feedback. We have accordingly made significant improvements following the suggestions and comments. For convenience, *we have marked in the revised PDF in blue* the added sections and the captions of newly added figures. 

Below we summarise our changes: 

- Llama results: Reviewers T6HH and rv9e requested inclusion of an additional model. We have **added results with Llama 3.1 8B** in the 
Appendix (Section C) showcasing similar patterns to Gemma, which we believe greatly increases the value and generality of our claims.
- Added in Figure 19 an ablation of the usage of RoPE frequencies in different domains (Italian, Chinese, Code, and Arithmetic) to show that our results generalise over different types of domains.

- Added **additional experimental ablation** comparing p-RoPE to increasing the max wavelength parameter in Table 2 as requested by Reviewer T6HH. We are happy to report that p-RoPE outperforms this baseline.
- Added **additional experimental ablation** comparing p-RoPE to removing the *highest* frequency instead of the lowest ones as suggested in p-RoPE in Table 2, as requested by Reviewer 7eRV. We are happy to report that p-RoPE outperforms this baseline.
- Added **additional experimental ablation** comparing p-RoPE to partial-RoPE in Table 2, as requested by Reviewer 7eRV. We are happy to report that p-RoPE outperforms this baseline.

- Added a more **detailed analysis of the “apostrophe head”** in the Appendix (Section E.1)  – reverse engineering how RoPE is used to construct this particular head. 
- **Additional synthetic result** showing that RoPE does not decay over a constant sequence of *repeated* queries and keys sampled *once* from a Gaussian as requested by Reviewer eLgp in Appendix B.4.  
- **Improved related works**: Following the advice from Reviewer T6HH, we have added a more detailed discussion of related works in the Appendix (Section B). 

We hope that our revisions have strengthened our contributions and would like to thank the reviewers for their valuable suggestions.  We look forward to productive rebuttal discussions.

### Official_Comment — Reviewer_rv9e — Thank you for clarifications

- Note ID: `XfSZNecnE3`
- Discussion number: `9`
- Invitation: `ICLR.cc/2025/Conference/Submission8344/-/Official_Comment`


#### Title

Thank you for clarifications

#### Comment

Thank you for clarifications. >> "something which we found to be possible – although we found that performance was better when training from scratch" - was it included in the paper? If the evaluation results were not so good, it is still an interesting point for practical usage.

I went over other reviews, authors' answers, and changes made in the paper. I think paper makes a solid contribution and has sufficient evidence to support main claims. I feel comfortable keeping my score at Accept.

### Official_Comment — Reviewer_T6HH — Thanks for the rebuttal and the detailed responses.

- Note ID: `yqWzepwJ5o`
- Discussion number: `10`
- Invitation: `ICLR.cc/2025/Conference/Submission8344/-/Official_Comment`


#### Title

Thanks for the rebuttal and the detailed responses.

#### Comment

I appreciate the authors' effort during the rebuttal phase in conducting additional experiments and making modifications to the descriptions. Most of my concerns have been resolved. Considering the resolution of key issues (e.g., the use of the LLaMA series as the base, additional baselines, and more analyses of attention patterns), I will raise my score.  

However, there are still three remaining concerns that I would like to address:  

1. The authors seem to have misunderstood my Question 2, which is about the dataset on which these analyses or statistical experiments (e.g., 2-norm) are conducted, not about the pretraining dataset of Gemma.  

2. In response to Question 3, the authors claimed, *"as by construction NoPE is invariant to relative distance."* I would like to point out that this understanding of NoPE is incomplete. NoPE simply does not provide explicit PE information to the Transformer, but it can implicitly learn both absolute and relative positional information through the causal mask. For example, [1] has demonstrated that a single-layer NoPE can learn absolute positional information, and a two-layer NoPE can learn relative positional information. Furthermore, Kazemnejad et al., 2024, showed through similarity analysis of hidden representations that NoPE's representations are highly similar to those of T5. Likewise, the Proposition 5.3 in this paper merely highlights the problem with a single-layer NoPE. However, in practice, single-layer Transformer-NoPE is almost never used, so this conclusion is not particularly exciting.  
[1] *Latent Positional Information is in the Self-Attention Variance of Transformer Language Models Without Positional Embeddings*, ACL2023 Honorable Mentions.  

3. Finally, I would like to discuss p-RoPE and partial-RoPE (as suggested by reviewer DtVE). Let’s assume \(d_{head} = 128\):  
   - The original RoPE uses `inv_freq = 1.0 / (base ** (torch.arange(0, 128, 2) / 128))`, with the lowest frequency being $1.0 / \text{base}^{126/128}$.  
   - 0.5-RoPE uses `inv_freq = 1.0 / (base ** (torch.arange(0, 64, 2) / 128))`, with the lowest frequency being $1.0 / \text{base}^{62/128}$.  
   - Partial-RoPE (50% part) uses `inv_freq = 1.0 / (base ** (torch.arange(0, 64, 2) / 64))`, with the lowest frequency being $1.0 / \text{base}^{62/64}$.  

   Clearly, 0.5-RoPE removes the low-frequency components of positional encoding more significantly, but a comparison between p-RoPE and partial-RoPE remains valuable (if more time and resources are available). The low-frequency components in the original RoPE are effective at learning semantics because these dimensions are less sensitive to positional changes, which is beneficial for semantic learning. In partial-RoPE, while the low-frequency information still exists, half of the dimensions are unaffected by PE. This implies that semantic learning does not necessarily need to rely on low-frequency dimensions. I hope the authors can include this experiment in the next version.  

   Additionally, regarding partial-RoPE, beyond its mention in the GitHub issue, it has also been used in GPT-NeoX and DeepSeek-V2 (which can serve as formal references).

### Official_Comment — Authors — Thank you for supporting our work!

- Note ID: `LXzkpXkekj`
- Discussion number: `11`
- Invitation: `ICLR.cc/2025/Conference/Submission8344/-/Official_Comment`


#### Title

Thank you for supporting our work!

#### Comment

We are happy to hear that you wish to maintain your score and also would like to thank you for your fast reply!

**was it included in the paper? If the evaluation results were not so good, it is still an interesting point for practical usage.**

We have not included it as it was preliminary experimentation, however we will include it in an eventual camera ready in the Appendix, as we believe we lack sufficient time in this rebuttal period to experiment with this appropriately. 

We once again thank you for your efforts in reviewing our work and for your positive score.

### Official_Comment — Authors — Thank you for supporting our work and the great review!

- Note ID: `1nj3XHw31F`
- Discussion number: `12`
- Invitation: `ICLR.cc/2025/Conference/Submission8344/-/Official_Comment`


#### Title

Thank you for supporting our work and the great review!

#### Comment

Thanks for your reply! We are happy to hear that you intend to raise your score. We would like to also thank you for the follow up questions and for your engagement with our work. 

**The authors seem to have misunderstood my Question 2, which is about the dataset on which these analyses or statistical experiments (e.g., 2-norm) are conducted, not about the pretraining dataset of Gemma.**

We apologise as we have indeed misunderstood your original question, thanks for clarifying. This is a very interesting point and we completely agree that this would be an interesting ablation to conduct. We have added in the Appendix a new figure with a number of different domains: Italian, Chinese, Code, and Arithmetic. The results can be found in the newly added Figure 19. The patterns we find for these types of prompts are very similar across the different domains. 

**In response to Question 3, the authors claimed, "as by construction NoPE is invariant to relative distance." I would like to point out that this understanding of NoPE is incomplete. NoPE simply does not provide explicit PE information to the Transformer, but it can implicitly learn both absolute and relative positional information through the causal mask […]**

We thank you for the very interesting comment. We would like to start by agreeing with you that we do not believe that NoPE is unable to learn positional information and perhaps our statement on “NoPE being invariant to relative distance” was not very precise. With that, we meant that over very long context even the slowest frequencies in RoPE will eventually destroy information – but that instead with NoPE, this semantic destruction does not occur because by construction the individual channels *can be used* in a way that is invariant to relative distance. For example, in our explanation of the apostrophe head, we show that a particular channel is being used to detect the BOS token, but that this mechanism will eventually break due to the rotations if the context is long enough. Instead, if this was a non-rotating channel, this mechanism would be much more robust. You are right in pointing out that this does not have to be true if the previous layer learns some positional artefacts, but our claim is more on the possibility of this to occur with NoPE and on the impossibility with RoPE.

You make a great point when you mention that works such as [1] and Kazemnejad et al point out that indeed with NoPE the Transformer can still learn a positional bias – given more than one layer. We still believe that our result is interesting because it provides understanding at a different level. While it is definitely interesting to study the expressive power of a sequence of layers, we believe there is still value in studying mechanisms that can occur in a specific attention head. Not only because we can show that certain mechanisms would not be able to occur with NoPE at the first layer, but also because many of these compositional arguments from a practical perspective will be less robust and less efficient. For instance, the proof of Kazemnejad et al relies on the universal approximation theorem to map 1/t to some desired decay function, which in practice of course points to potential issues with generalisation for instance when t is out of distribution. We have also added a comment on this in the Appendix (in a paragraph at the end of Section A.2).

We hope that this clarifies our position on this and are of course happy to keep discussing this with you.  

**Finally, I would like to discuss p-RoPE and partial-RoPE (as suggested by reviewer DtVE). […]**

We thank you for the additional references – we have now expanded our section on partial RoPE with a discussion on the differences! We expect partial RoPE and p-RoPE to achieve in different ways the same end goal. We believe our work provides a much more solid understanding on why techniques such as partial RoPE are helpful. We are happy to see techniques similar to p-RoPE being used already! 

For completeness, we have now also added an ablation in Table 2 with partial-RoPE showing that p-RoPE seems to show stronger performance. We hope that you can appreciate this ablation.

Overall, we sincerely thank you for the effort you have put in reviewing our manuscript and thank you again for mentioning that you wish to increase your score. We believe your comments have been really useful in improving our work and welcome any more you may have!

### Official_Comment — Reviewer_T6HH — Thanks for the rebuttal.

- Note ID: `WdT6lnGomD`
- Discussion number: `13`
- Invitation: `ICLR.cc/2025/Conference/Submission8344/-/Official_Comment`


#### Title

Thanks for the rebuttal.

#### Comment

Thank you for your detailed response. I have no further questions. I raised my score (from 5 to 8).

### Official_Comment — Authors — Thank you!

- Note ID: `kCqe48VNJR`
- Discussion number: `14`
- Invitation: `ICLR.cc/2025/Conference/Submission8344/-/Official_Comment`


#### Title

Thank you!

#### Comment

Thank you very much for your thorough review and your useful feedback. We found it very helpful. We are happy that you have decided to raise your score! Best, Authors.

### Official_Comment — Authors — Additional ablation

- Note ID: `UTybi6gAE7`
- Discussion number: `15`
- Invitation: `ICLR.cc/2025/Conference/Submission8344/-/Official_Comment`


#### Title

Additional ablation

#### Comment

We wanted to also kindly add that due to our discussion with Reviewer T6HH, we have now added an experimental comparison to partial RoPE in Table 2 and expanded our discussion in the Appendix (Section D) on the similarities and differences. We also for convenience copy here the relevant discussion. We are happy to note that the reviewer has now decided to raise the score as well in support of the paper. We hope that you may find this relevant to your initial comments.

**In response to Reviewer T6HH to a request of comparison with partial RoPE:**

"We have now expanded our section on partial RoPE with a discussion on the differences! We expect partial RoPE and p-RoPE to achieve in different ways the same end goal. We believe our work provides a much more solid understanding on why techniques such as partial RoPE are helpful. We are happy to see techniques similar to p-RoPE being used already!

For completeness, we have now also added an ablation in Table 2 with partial-RoPE showing that p-RoPE seems to show stronger performance. We hope that you can appreciate this ablation."

### Official_Comment — Reviewer_eLgp

- Note ID: `qoKjcM5I40`
- Discussion number: `17`
- Invitation: `ICLR.cc/2025/Conference/Submission8344/-/Official_Comment`


#### Comment

I appreciate the authors' response. While the rebuttal addressed some questions, I still have concerns about the experimental settings and results.

1. The effect of Figure 9 is indeed shocking. But what does the phrase "perfectly aligned" mean? In other words, what is the difference between the settings of Figure 9 and Figure 2 (a)?
2. The author's description of semantics and position in Table 1 is not experimentally verified.
3. Even though the author says the focus is on understanding, I still agree with the theoretical proof in the paper, but only PPL seems less convincing in the experiment.

### Official_Comment — Authors

- Note ID: `MDUrBL1RPC`
- Discussion number: `18`
- Invitation: `ICLR.cc/2025/Conference/Submission8344/-/Official_Comment`


#### Comment

We would like to thank you very much for your response and for your efforts as a reviewer. We are happy to have addressed some of your questions. We would like to address your remaining points.

**The effect of Figure 9 is indeed shocking. But what does the phrase "perfectly aligned" mean? In other words, what is the difference between the settings of Figure 9 and Figure 2 (a)?**

We thank you for the question. To clarify:

*Figure 2 (a)*: the queries and keys are repeated vectors of all-ones. In other words, we take the first queries and keys q_1 and k_1 as all-ones vectors and repeat them n times.

*Figure 9*:  We sample 1 query and 1 key from a Gaussian distribution, i.e. the queries and keys are different vectors now. We then repeat the query n times and the key n times. We then show the effect of RoPE on this repeated sequence. 

The difference is that in Figure 2 (a) they are always perfectly aligned as the base vectors are all vectors of 1s. Since they are the same vector, their initial angle between them is always 0 (what we mean by perfectly aligned). Instead, in Figure 9, while the queries and keys are repeated, they are *different* Gaussian random vectors, so they are not the same vector. We see that therefore the vectors being repeated is not sufficient, but they also have to be aligned (the same vector) for them to decay. 

Just for completeness, the difference in Figure 2 (b) is that here we sample Gaussian vectors for each query and key, so we sample n queries and n keys from a Gaussian, while in Figure 9, we sample only 1 query and 1 key and then repeat them n times.

Please let us know if this is now hopefully more clear! 

**The author's description of semantics and position in Table 1 is not experimentally verified.**

We agree with you and have clarified in the table that these are the discussed *theoretical* properties. Thanks for pointing this out!

**Even though the author says the focus is on understanding, I still agree with the theoretical proof in the paper, but only PPL seems less convincing in the experiment.**

We are happy that you agree with the proofs in our paper! There are many works that propose a new type of positional encoding, but we believe that it is also very valuable to improve our understanding of the positional encodings being used today. We hope that you agree that our paper indeed does contribute to a better understanding of RoPE. In fact, we are happy that in your review you mention that we “question existing assumptions about RoPE”  and that our analysis has “practical relevance” which we believe is really the main point of this paper! We hope that this work can in fact provide the community with a more solid understanding of why RoPE is useful.

The experimental evaluation of our p-RoPE method is mostly to validate our intuitions derived in the sections of this paper. We hope that the 3 added experimental ablations help to solidify to you that p-RoPE is an interesting and valid approach. We completely agree that we could experiment on more tasks, but we believe this to be somewhat outside the main scope of this work which is that of understanding RoPE and not the proposal of a new PE.

We once again thank you for your review and are happy to answer any further questions.

### Official_Comment — Authors

- Note ID: `0jlWLQcsX0`
- Discussion number: `19`
- Invitation: `ICLR.cc/2025/Conference/Submission8344/-/Official_Comment`


#### Comment

Thanks for pointing this out, we have updated line 423 to reference this work.  

We are indeed happy to see evidence for our conjecture. 

Best,
Authors

### Official_Comment — Reviewer_DtVE — Post-rebuttal update

- Note ID: `o2J6f0vXzG`
- Discussion number: `20`
- Invitation: `ICLR.cc/2025/Conference/Submission8344/-/Official_Comment`


#### Title

Post-rebuttal update

#### Comment

I appreciate the authors' thoughtful responses and the additional efforts put into conducting more ablation studies. After reviewing them, I believe these enhance the quality of the paper. Therefore, I have raised my score.

### Official_Comment — Authors — Thank you!

- Note ID: `9L7xi66Suf`
- Discussion number: `21`
- Invitation: `ICLR.cc/2025/Conference/Submission8344/-/Official_Comment`


#### Title

Thank you!

#### Comment

Dear Reviewer DtVE,

Thank you for acknowledging our efforts! We are very happy that you found our responses useful and have upgraded your score.

Could you please let us know what would be required, in your opinion, to bring the paper over the bar of acceptance?
We still have ~1.5 days to make concrete revisions to the paper, and we'd really like to try to make it happen!

Best,
Authors

### Official_Comment — Authors

- Note ID: `5A3S79YNFI`
- Discussion number: `22`
- Invitation: `ICLR.cc/2025/Conference/Submission8344/-/Official_Comment`


#### Comment

Dear Reviewer 7eRV,

We are very grateful for you review and insights on our paper. As the rebuttal period is coming to an end, we would really appreciate if you could let us know if you our responses and additional ablations have helped to improve your opinion on our work? 

We are available for further discussion at any point. 

Best,
Authors

### Official_Comment — Authors

- Note ID: `m4rjkAppjS`
- Discussion number: `23`
- Invitation: `ICLR.cc/2025/Conference/Submission8344/-/Official_Comment`


#### Comment

Dear Reviewer DtVE,

We thank you once again for your engagement and for already increasing your score.

As the rebuttal period is now coming near an end, we would be really interested in hearing if you believe there are outstanding unresolved points in our work. We would really love to have a chance to address them!

Best,
Authors

### Official_Comment — Authors

- Note ID: `bJtATg1qKb`
- Discussion number: `24`
- Invitation: `ICLR.cc/2025/Conference/Submission8344/-/Official_Comment`


#### Comment

Dear Reviewer eLgp,

We thank you once again for your engagement with us and your further questions!

We are wondering if our response has addressed your remaining concerns? As we the rebuttal period is almost over, we would really be happy to have the chance to address any final questions before we cannot anymore.

Best,
Authors

## Decision

### Decision — Program_Chairs — Paper Decision

- Note ID: `KJCe51ATEz`
- Discussion number: `1`
- Invitation: `ICLR.cc/2025/Conference/Submission8344/-/Decision`


#### Title

Paper Decision

#### Decision

Accept (Poster)


---

# Wavelet-based Positional Representation for Long Context — OpenReview 审稿全文归档

- Venue: **ICLR 2025 Poster**
- OpenReview forum: [https://openreview.net/forum?id=OhauMUNW8T](https://openreview.net/forum?id=OhauMUNW8T)
- Official paper page: [https://proceedings.iclr.cc/paper_files/paper/2025/hash/c131c8875c7b1133ffdad2b53cb10e91-Abstract-Conference.html](https://proceedings.iclr.cc/paper_files/paper/2025/hash/c131c8875c7b1133ffdad2b53cb10e91-Abstract-Conference.html)
- Reviewer handles are the public OpenReview pseudonyms; no attempt is made to identify individuals.
- Source: public OpenReview review dump; fields are preserved as released, including review text, rebuttal comments, meta-review, and decision where available.

## Paper Abstract

In the realm of large-scale language models, a significant challenge arises when extrapolating sequences beyond the maximum allowable length. 
This is because the model's position embedding mechanisms are limited to positions encountered during training, thus preventing effective representation of positions in longer sequences.
We analyzed conventional position encoding methods for long contexts and found the following characteristics.
(1) When the representation dimension is regarded as the time axis, Rotary Position Embedding (RoPE) can be interpreted as a restricted wavelet transform using Haar-like wavelets. 
However, because it uses only a fixed scale parameter, it does not fully exploit the advantages of wavelet transforms, which capture the fine movements of non-stationary signals using multiple scales (window sizes). 
This limitation could explain why RoPE performs poorly in extrapolation.
(2)
Previous research as well as our own analysis indicates that Attention with Linear Biases (ALiBi) functions similarly to windowed attention, using windows of varying sizes.
However, it has limitations in capturing deep dependencies because it restricts the receptive field of the model.
From these insights, we propose a new position representation method that captures multiple scales (i.e., window sizes) by leveraging wavelet transforms without limiting the model's attention field.
Experimental results show that this new method improves the performance of the model in both short and long contexts. 
In particular, our method allows extrapolation of position information without limiting the model's attention field.

## Review Inventory (4 Official Reviews, 12 Discussion/Comment Notes)

- Final decision: **Accept (Poster)**
- Official review ratings (review order): `5, 6, 5, 5`

## Official Reviews

### Official_Review — Reviewer_kKUx

- Note ID: `yM57GeN1C3`
- Discussion number: `1`
- Invitation: `ICLR.cc/2025/Conference/Submission13404/-/Official_Review`


#### Summary

This paper conducts a unified analysis of positional encoding methods in large language models and proposes a novel wavelet-based approach. The work begins by demonstrating that RoPE can be interpreted as a restricted wavelet transform using Haar-like filters operating at a fixed scale. The authors then analyze ALiBi, revealing that it functions similarly to windowed attention with varying window sizes but is limited by constraints on the attention mechanism's receptive field. Building on these insights, the paper introduces a new wavelet-based positional representation method that leverages multiple scales through wavelet transforms. The method is designed to capture both local and long-range dependencies without restricting the model's receptive field. The authors validate their approach through experiments on short and long contexts.

#### Soundness

`2`

#### Presentation

`2`

#### Contribution

`2`

#### Strengths

1. The motivation and connection are clear as the authors provide a unified analysis of the positional encoding methods.
2. The paper introduces a new wavelet-based positional representation method that leverages multiple scales through wavelet functions.
3. The proposed wavelet positional embedding shows improvements across tasks.

#### Weaknesses

1. Novelty Concerns: 
- The core idea of wavelet-based positional encoding has been explored in the previous work GMT (Ngo et al., 2023a,b), which uses graph wavelets to generate node positional representations that can capture the structural information of a center node on the graph at different resolutions, though in a different domain. However, there is insufficient discussion of how this approach differs fundamentally from or improves upon prior works.
2. Mathematical Rigor:
- The unified comparison for different positional encodings is not rigorous as the wavelet property needs stronger mathematical justification. The proposed "Haar-like wavelets" in Equation (7) require verification of wavelet admissibility conditions beyond just square integrability.
3. Experimental Design: 
- Limited exploration of wavelet families (only four tested).
- Insufficient analysis of scale parameter choices (only two tested).
4. Minor issues:
- The paper lacks of clear notation definition and problem setup affects readability.
- Inconsistent terminology between abstract ("simple Haar wavelets" in Line 18) and technical content ("Haar-like wavelets" in Sec 3.2).
- The phrase 'single window size' (in Line 19) should be replaced with 'fixed scale parameter' to align with standard wavelet terminology, as it specifically refers to the dilation factor in wavelet transforms.
- The space definition L⊭(R) in Line 135 appears to have a typographical error - it should be L²(R), the space of square-integrable functions.
- Some mathematical notation is not explicitly defined. For example, the superscript T in equation (3) seems to represent the length of a discrete sequence, but this isn't explicitly defined.

#### Questions

1. How does your approach fundamentally differ from MGT's wavelet position encoding?
2. Can you provide formal verification of the wavelet admissibility conditions for the proposed "Haar-like wavelets"? Or what else condition you might need to add to make it a wavelet?
3. Why were these specific wavelet families chosen? Have you explored other wavelet families, particularly discrete wavelets like Daubechies or biorthogonal wavelets?
4. What guided your choice of scale parameter ranges? Have you conducted sensitivity analyses on these choices?
5. What is the theoretical justification for removing the $\frac{1}{\sqrt{a}}$ amplitude term?

[1] Ngo, Nhat Khang, Truong Son Hy, and Risi Kondor. "Multiresolution graph transformers and wavelet positional encoding for learning long-range and hierarchical structures." The Journal of Chemical Physics 159.3 (2023).

#### Flag For Ethics Review

- No ethics review needed.

#### Rating

`5`

#### Confidence

`3`

#### Code Of Conduct

Yes

### Official_Review — Reviewer_DFcs

- Note ID: `Rmom89YIBT`
- Discussion number: `2`
- Invitation: `ICLR.cc/2025/Conference/Submission13404/-/Official_Review`


#### Summary

The paper proposes a wavelet transform-based positional representation for transformer models. It begins with highlighting the properties of existing position representation techniques such as relative position bias, RoPE and ALiBi. The authors then show how RoPE can be interpreted as a single scale wavelet transform with a Haar-like wavelet. Then the authors present a position embedding based on wavelet transform. This multi-scale embedding can be viewed as a generalization of RoPE and possess attractive "multi-window" properties of ALiBi. Experiment results have been reported on short and longer-context scenarios showing improved perplexity over existing methods and better extrapolation properties.

#### Soundness

`3`

#### Presentation

`3`

#### Contribution

`3`

#### Strengths

- The paper is well-written barring some typographical errors. It motivates the problem well, starts with a well-defined goal, describes existing methods clearly, and presents the proposed method in a manner which is easy to appreciate.
- The paper tackles a critical problem of context length extrapolation which often arises in practical settings. The method holds significance, not just for the language modeling community, but also other domains such as time series forecasting where context length extrapolation may improve the performance of models on high-frequency data (see [1]). 
- The proposed method is novel to the best of my knowledge. It creatively relates RoPE to time-frequency analysis and proposes a promising method based on wavelet transforms. 
- The method outperforms alternatives and scales gracefully with extrapolated context lengths. 

[1] Ansari, Abdul Fatir, et al. "Chronos: Learning the language of time series." arXiv preprint arXiv:2403.07815 (2024).

#### Weaknesses

- While the discussion of related methods is generally well done, discussion of RoPE scaling techniques (linear, NTK-aware) is missing. A discussion and comparison with these techniques would significantly improve the positioning of this work.
- The results reported in sections 6 and 7 are excellent proofs of concept but they lack comprehensiveness. Particularly, in section 7, evaluations beyond the CodeParrot dataset would be needed to thoroughly appreciate the proposed method. Furthermore, the paper only evaluates the model in term of the perplexity. For a stronger evaluation, this needs to be augmented with task-based evaluations. Apart from common language tasks, you might also want to consider associative recall tasks [2] and the long range arena [3] since the focus of this work is context length extrapolation. To further demonstrate the robustness of the proposed method, domains beyond natural language may also be considered, for example, DNA modeling [4], audio generation [4] and time series forecasting [1]. 

I am willing to raise my score if the evaluation is strengthened. 

[1] Ansari, Abdul Fatir, et al. "Chronos: Learning the language of time series." arXiv preprint arXiv:2403.07815 (2024).    
[2] Arora, Simran, et al. "Zoology: Measuring and improving recall in efficient language models." arXiv preprint arXiv:2312.04927 (2023).    
[3] Tay, Yi, et al. "Long range arena: A benchmark for efficient transformers." arXiv preprint arXiv:2011.04006 (2020).    
[4] Gu, Albert, and Tri Dao. "Mamba: Linear-time sequence modeling with selective state spaces." arXiv preprint arXiv:2312.00752 (2023).

#### Questions

See above.

#### Flag For Ethics Review

- No ethics review needed.

#### Rating

`6`

#### Confidence

`4`

#### Code Of Conduct

Yes

### Official_Review — Reviewer_fCa9

- Note ID: `ZkIXSYKMqT`
- Discussion number: `3`
- Invitation: `ICLR.cc/2025/Conference/Submission13404/-/Official_Review`


#### Summary

the paper presents a novel positional encoding method leveraging wavelet transforms to address the challenges of limited receptive field and  extrapolation in LLMs. 

The proposed wavelet-based approach aims to overcome these limitations by introducing a multi-scale analysis that captures the fluidity of natural language and enhances the model's ability to extrapolate beyond its training context length.

In extrapolation experiments, model that used Ricker-based wavelet positional embedding had the lowest PPL trained on the WikiText-103 dataset, compared with RoPE, ALibi, Transformer-XL.

#### Soundness

`2`

#### Presentation

`2`

#### Contribution

`2`

#### Strengths

- The authors provide a solid theoretical foundation by drawing parallels between RoPE and wavelet transforms, and by extending this analogy to propose their method.
- The proposed method has the potential to be widely applicable to various transformer-based models
- The paper is easy to read and positions itself clearly with respect to related work.

#### Weaknesses

- The paper primarily evaluates the method on language modeling tasks. It would be valuable to see how the approach generalizes to other NLP tasks such as question answering or text summarization.
- The paper does not provide sufficient experimental evidence to support the authors' claim that Wavelet Transform can capture the dynamic changes in a sequence over positions.（L84-85）
- See my questions/suggestions below

#### Questions

- How did the “shift and scale parameters” been decided? Is there any ablation study or results for this particular setting?
- Did you compare the length extrapolation results with other RoPE-based methods? (eg. relative works you mentioned in L47-L48)
- Both RoPE and Wavelet methods aim to model distances at different scales by placing them in different dimensions [1]. In Figure 3, the diagonal lines in the RoPE plot appear because the model can focus on the features of specific token distances. However, Wavelet functions do not have this stable frequency inductive bias, which is why they do not produce diagonal patterns in the attention scores. Do you think this is a desirable property?
- In the experiment represented by Figure 3, what kind of text is input into the model? Why is the initial word particularly important? Which specific tokens does the model attend to? Is this phenomenon common?

[1] Hong, Xiangyu, et al. "On the token distance modeling ability of higher RoPE attention dimension." arXiv preprint arXiv:2410.08703 (2024).

#### Flag For Ethics Review

- No ethics review needed.

#### Rating

`5`

#### Confidence

`4`

#### Code Of Conduct

Yes

### Official_Review — Reviewer_QwSG

- Note ID: `E6duFTc7ts`
- Discussion number: `4`
- Invitation: `ICLR.cc/2025/Conference/Submission13404/-/Official_Review`


#### Summary

In this work, the authors proposed wavelet-based positional encoding for the length extrapolation problem of language models. The motivation is based on an observation that the widely used Rope is related to wavelet transform via simple Haar wavelet functions with a fixed scale. By further investigating the properties of ALiBi, the authors modified the form of relative positional encoding to use wavelet transform based approaches. Several experiments are conducted to demonstrate the performance of the proposed positional encoding.

#### Soundness

`2`

#### Presentation

`2`

#### Contribution

`2`

#### Strengths

1. The paper is easy-to-follow. 
2. The length extrapolation problem is important for language models.

#### Weaknesses

1. **There exist gaps from the motivation to the proposed approach**. In Section 3 and 4, the authors provide analysis on the relationship between RoPE and wavelet transform, and the properties of ALiBi like positional encodings. The ability of ALiBi to accommodate multiple window sizes is concluded as the key point for better length extrapolation performance, while RoPE is claimed to be worse. Based on these statements, a natural question is, what is the advantage of using wavelet transform? The authors do not provide enough supportive quantitative evidence for it, but directly modify the relative positional encoding by using wavelet transform. If it has superiorities, why don't we follow the form of RoPE to use wavelet transform instead of RPE? These points should be better clarified to make the motivation of the proposed approach more convincing.

2. **The design choice for the proposed approach should be better supported**. In Section 5, the authors propose to use Ricker Wavelet as the base function of wavelet transform, and set the shift and amplititude parameters with predefined strategies. However, the reasons to choose these designs are not well clarified (except lines 294-303, which are rather superficial).

3. **The empirical improvement is marginal**. From Table 2 and 3, we can see that the proposed PE does not show significant improvement compared to compared approaches such as ALiBi, which is proposed in 2022 and is not the state-of-the-art nowadays. The evaluations are also limited in task types and scales. Overall, the empirical results are not convincing to show the superiority of the proposed approach.

Overall, I hope the authors can well address the above concerns, which I think are important for the quality of this work.

#### Questions

See the Weaknesses.

#### Flag For Ethics Review

- No ethics review needed.

#### Rating

`5`

#### Confidence

`4`

#### Code Of Conduct

Yes

## Meta-Review

### Meta_Review — Area_Chair_M7PL

- Note ID: `2LXw0S6n40`
- Discussion number: `1`
- Invitation: `ICLR.cc/2025/Conference/Submission13404/-/Meta_Review`


#### Metareview

The paper proposes a wavelet transform-based positional representation for transformer models. This multi-scale embedding can be viewed as a generalization of RoPE and possess attractive "multi-window" properties of ALiBi. Several experiments are conducted in long and short context scenarios to demonstrate the performance of the proposed positional encoding. They show improved perplexity, better length extrapolation properties and  sometimes even improved question answering performance compared to RoPE.

Strengths: The paper tackles a critical problem of context length extrapolation which often arises in practical settings. The proposed method is novel, and the authors provide a solid theoretical foundation by drawing parallels between RoPE and wavelet transform. Their unified analysis of positional encoding methods sheds more  light on this important topic. 

Weaknesses: Reviewers raised a few concerns regarding novelty, mathematical rigor and experimental evaluation.  In my estimation they have been adequately addressed by the authors during the discussion phase.

Overall I think this is a good paper which adds theoretical understanding, as well as a practical method with potential significance beyond the language learning community. I recommend accepting it.

#### Additional Comments On Reviewer Discussion

The reviewers did not respond to the authors during discussion, so I had to judge the responses myself. 
In response to the concerns regarding mathematical rigor the authors clarified several points and fixed a few shortcomings in their theoretical analysis.
Regarding experimental evaluation, the authors added several new evaluations and a host of additional ablations. Some concerns regarding comparison to additional baselines and larger models remain, but are not critical in my opinion.

## Rebuttal and Discussion Comments

### Official_Comment — Authors — Official Comment by Authors (1/3)

- Note ID: `w2DAItnbV4`
- Discussion number: `2`
- Invitation: `ICLR.cc/2025/Conference/Submission13404/-/Official_Comment`


#### Title

Official Comment by Authors (1/3)

#### Comment

We sincerely appreciate the reviewers for their time, effort, and valuable feedback in evaluating our research.  And, we would also like to thank you for evaluating the readability of our paper. In response to the reviewers' valuable comments, we have conducted additional explanations and experiments, and we would like to respectfully present our findings for your consideration.

> **(Weakness1) There exist gaps from the motivation to the proposed approach. In Section 3 and 4, the authors provide analysis on the relationship between RoPE and wavelet transform, and the properties of ALiBi like positional encodings. The ability of ALiBi to accommodate multiple window sizes is concluded as the key point for better length extrapolation performance, while RoPE is claimed to be worse. Based on these statements, a natural question is, what is the advantage of using wavelet transform? The authors do not provide enough supportive quantitative evidence for it, but directly modify the relative positional encoding by using wavelet transform. If it has superiorities, why don't we follow the form of RoPE to use wavelet transform instead of RPE? These points should be better clarified to make the motivation of the proposed approach more convincing.**

Thank you for presenting such a thought-provoking question!

> Based on these statements, a natural question is, what is the advantage of using wavelet transform?

Initially, we contemplated the application of wavelet transform in accordance with the RoPE format. However, as outlined in Section 3.2, "Theoretical Analysis," it is important to note that RoPE conducts wavelet transform along the dimensional axis rather than the time axis. This approach with RoPE does not fully use the unique characteristics of wavelet transforms, which investigate the information in a signal at a certain time. 

Furthermore, when applying wavelet transformation based on RoPE, there is the issue that the scale parameter can only be set to a value less than or equal to d_head. When applying wavelet transformation based on RPE, the scale parameter can be set to any value within the context length when training. In wavelet transformation, the scale parameter represents resolution, and this characteristic is very important (we believe that the importance of scale can be understood from the ALiBi validation section and the ablation study for each scale parameter that we have added this time. ).  

Additionally, even if positional encoding were to incorporate a wavelet transform based on RoPE, the reliance on absolute position could hinder any potential improvements in extrapolation performance. To truly capitalize on the benefits of wavelet transforms for extrapolation purposes, it is essential that positional encoding shifts focus to performing the wavelet transform along the time axis or position, without adopting absolute position. This adjustment would allow for a more effective utilization of wavelet characteristics.

---

> **(Weakness2) The design choice for the proposed approach should be better supported. In Section 5, the authors propose to use Ricker Wavelet as the base function of wavelet transform, and set the shift and amplititude parameters with predefined strategies. However, the reasons to choose these designs are not well clarified (except lines 294-303, which are rather superficial).**

In response to your suggestions, we have included an ablation study examining the impact of shift and scale parameters, presented in Appendix A.10, as well as an analysis of each wavelet type in Appendix A.11. **In the parameter abration study, we verified 10 patterns of parameters. In the wavelet-type abration study, we used more than 20 wavelets.** Our findings reveal that adjusting the shift and scale parameters leads to a further reduction in perplexity. However, we found that the discrete wavelet approach did not yield the expected results; while we employed an approximation, we believe that a more effective selection strategy is needed. Our primary goal in this study was to establish foundational principles of position encoding using wavelet transforms, so we consider the discrete wavelet approach an area for future exploration. We chose Ricker, Morlet, and Gaussian wavelets due to their status as the most representative wavelets commonly described by mathematical formulas. At this juncture, we are not incorporating wavelets that involve complex numbers. Should we decide to investigate the application of complex numbers to position encoding in the future, we anticipate that a revised strategy would be necessary, as referenced in [1].

### Official_Comment — Authors — Official Comment by Authors (2/3)

- Note ID: `E9ofzxD1A3`
- Discussion number: `3`
- Invitation: `ICLR.cc/2025/Conference/Submission13404/-/Official_Comment`


#### Title

Official Comment by Authors (2/3)

#### Comment

> **(Weakness 3)The empirical improvement is marginal. From Table 2 and 3, we can see that the proposed PE does not show significant improvement compared to compared approaches such as ALiBi, which is proposed in 2022 and is not the state-of-the-art nowadays. The evaluations are also limited in task types and scales. Overall, the empirical results are not convincing to show the superiority of the proposed approach.**

I would like to discuss whether ALiBi is a method of SOTA.
As noted in the introduction of Section 1, our research specifically investigates position encodings used during pre-training. Although ALiBi was published in 2022 [2], it is still considered to be state-of-the-art as a position encoding used in “pre-training” because it is also used in mpt models[3] that can handle long sentences with high-performance. Furthermore, our comparison includes NoPE and RoPE, with θ=500000, as indicated in Table 2. Notably, NoPE is a recent 2023 publication, while RoPE (with θ=0.5m) is featured in a 2024 paper. It's important to highlight that RoPE (θ=0.5m) is employed in the Llama-3 model, which has achieved numerous state-of-the-art benchmarks in various tasks. The majority of current large-scale language models utilize either RoPE or ALiBi. We propose that our method offers distinct advantages, as it effectively captures intermediate words (as detailed in Section 6.3.2 and illustrated in Figure 3) while maintaining strong extrapolation performance (as shown in Section 6.2, Table 2), thereby demonstrating clear superiority over these established position encodings.


**In response to the comments we received, we have changed the notation of the proposed methods in Tables 1 and 2. (e.g., RoPE($θ$ =0.5m) -> RoPE (Xiong et al., 2024))**

**We have also added the results of the LongBench[7] evaluation in appendix A.14. The results show that the proposed method outperforms RoPE in most cases.**


Thank you for your useful comments. We are currently working on a comparison experiment between other tasks and the state-of-the-art RoPE position interpolation method. We will definitely let you know the results during the rebuttal period.

---

[1] Wang+, ICLR2020. Encoding word order in complex embeddings

[2] Press+, ICLR2022. Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation

[3] MosaicML NLP Team, Online2023. Introducing MPT-7B: A New Standard for Open-Source, Commercially Usable LLMs

[4] Kazemnejad+, NeurIPS 2023.The Impact of Positional Encoding on Length Generalization in Transformers

[5] Xiong+, NAACL 2024. Effective Long-Context Scaling of Foundation Models

[6] Dubey+, Arxiv 2024. The Llama 3 Herd of Models

[7] Bai+, ACL2024. LongBench: A Bilingual, Multitask Benchmark for Long Context Understanding

### Official_Comment — Authors — Official Comment by Authors (1/2)

- Note ID: `S5Cr5lpR4i`
- Discussion number: `4`
- Invitation: `ICLR.cc/2025/Conference/Submission13404/-/Official_Comment`


#### Title

Official Comment by Authors (1/2)

#### Comment

We sincerely appreciate the reviewers for their time, effort, and valuable feedback in evaluating our research. We are also deeply grateful for your thoughtful assessment of the readability and the theoretical derivations presented in our paper. In response to the reviewers' valuable comments, we have conducted additional explanations and experiments, and we would like to respectfully present our findings for your consideration.

> **(Q) How did the “shift and scale parameters” been decided? Is there any ablation study or results for this particular setting?**

In response to your suggestions, we have included an ablation study examining the impact of shift and scale parameters, presented in Appendix A.10, as well as an analysis of each wavelet type in Appendix A.11. **In the parameter abration study, we verified 10 patterns of parameters. In the wavelet-type abration study, we used more than 20 wavelets.** Our findings reveal that adjusting the shift and scale parameters leads to a further reduction in perplexity. However, we found that the discrete wavelet approach did not yield the expected results; while we employed an approximation, we believe that a more effective selection strategy is needed.

---

> **(Q) Both RoPE and Wavelet methods aim to model distances at different scales by placing them in different dimensions [1]. In Figure 3, the diagonal lines in the RoPE plot appear because the model can focus on the features of specific token distances. However, Wavelet functions do not have this stable frequency inductive bias, which is why they do not produce diagonal patterns in the attention scores. Do you think this is a desirable property?**

We appreciate the opportunity to address the concerns regarding the presence of diagonal patterns in the attention scores. We firmly believe that the absence of such patterns is a favorable attribute of RoPE. Figure 3 show the attention score in extrapolation. The length of the training is 512 and the inference length is 1012. Notably, Diagonal patterns appear in places longer than 512. When it is shorter than 512, diagonal patterns like this do not exist.This observation leads us to conclude that if Rotary Position Embedding (RoPE) is capable of effectively recognizing positions, diagonal patterns should not manifest. While we acknowledge the importance of further investigating the presence of diagonal patterns, we consider this aspect to be outside the primary focus of our current study. Therefore, we will not be conducting additional verification at this time, but we welcome future exploration of this topic. Thank you for your understanding.

---

> **(Q) In the experiment represented by Figure 3, what kind of text is input into the model? Why is the initial word particularly important? Which specific tokens does the model attend to? Is this phenomenon common?**

**An example of the correspondence between the text and the heat map is shown in Appendix A.13.** The model paid particular attention to special tokens such as <s>. In addition, for some heads, it also paid attention to words corresponding to the subject, which may have captured the characteristics of the sentence. This phenomenon was observed in any text.

We are currently conducting additional experiments in response to the other comments provided. We will respectfully share the results with you within the rebuttal period.

### Official_Comment — Authors — Official Comment by Authors (1/2)

- Note ID: `QvfzeyC6n2`
- Discussion number: `5`
- Invitation: `ICLR.cc/2025/Conference/Submission13404/-/Official_Comment`


#### Title

Official Comment by Authors (1/2)

#### Comment

We would like to express our heartfelt gratitude to the reviewers for their valuable feedback in evaluating our research. We would also like to sincerely thank you for the relatively high evaluation we have received. We are currently conducting additional experiments in response to the comments provided and will respectfully share the results during the rebuttal period.

Although this is not a response to the comments we received, we have included the following points in the appendix.
- Additional investigation of shift and scale parameters (A.9)
- Additional investigation of other wavelet types (A.10)
- Further explanation of the attention map (A.11)

There were also some errors in the mathematical formulae, which I have corrected to the extent possible at this stage. We are also currently conducting a thorough review of Section 3.2. We sincerely apologize for keeping you waiting for such a long time and kindly ask for your patience a little longer.

### Official_Comment — Authors — Official Comment by Authors (1/3)

- Note ID: `UknDsianRL`
- Discussion number: `6`
- Invitation: `ICLR.cc/2025/Conference/Submission13404/-/Official_Comment`


#### Title

Official Comment by Authors (1/3)

#### Comment

We would like to express our heartfelt gratitude to the reviewers for their valuable feedback in evaluating our research. Thank you for your excellent comments, especially regarding wavelet transforms. We would be more than happy to address your comments and provide our responses.

> **(Q1) How does your approach fundamentally differ from MGT's wavelet position encoding?**

This paper is of course included in the references. First of all, we would like to emphasize that we were quite worried about whether or not to mention this paper in the main text.
Although this paper appears to be quite similar to our method, it is actually quite different.

1. **Position Encoding Models**: The models addressing position encoding are entirely different. We approach position encoding within the general Transformer framework, while the paper focuses on position encoding specific to Graph Transformers.
2. **Different Tasks**: The tasks being addressed are distinct. Their work involves tasks that handle graphs and represent macromolecules with multiple edges, whereas we are focused on sequences and extrapolation tasks.
3. **Divergent Objectives**: Their objective is to propose a new position encoding that guarantees locality in both spectral and spatial domains. In contrast, we propose a position encoding that remains valid outside of the learned context.
4. **Representation of Location**: The information regarding location representation diverges as well. They represent the position of each node in a graph, while we represent relative locations within a sequence. Graphs can have multiple edges, while sequences represent one-way paths, leading to fundamentally different ways to depict location.

Despite the similar name and concept, our methods are quite distinct. We chose to include this paper in the references to avoid confusing readers by introducing it in the main text.

---

> **(Q3)Why were these specific wavelet families chosen? Have you explored other wavelet families, particularly discrete wavelets like Daubechies or biorthogonal wavelets?**

Thank you for your excellent point! We have conducted additional verification of other wavelets. The results of the experiment are described in Appendix A.11. 

We experimented with more than **20 additional wavelet types**.
At this juncture, we are not incorporating wavelets that involve complex numbers. Should we decide to investigate the application of complex numbers to position encoding in the future, we anticipate that a revised strategy would be necessary, as referenced in [1].
We found that the discrete wavelet approach did not yield the expected results; while we employed an approximation, we believe that a more effective selection strategy is needed.
We also conducted a survey based on the vanishing moment and found that it may have a certain impact. This discovery would not have been possible without your insightful comments, for which we are sincerely grateful. Thank you very much!
Our primary goal in this study was to establish foundational principles of position encoding using wavelet transforms, so we consider the discrete wavelet approach an area for future exploration.

---

> **(Q4) What guided your choice of scale parameter ranges? Have you conducted sensitivity analyses on these choices?**

In response to your suggestions, we have included an ablation study examining the impact of shift and scale parameters, presented in Appendix A.10. We experimented with more than **10 additional parameter patterns**.

The following was learned from the test results. Increasing the scale parameter while keeping the shift parameter constant generally maintained extrapolation performance, though with some fluctuations. However, increasing the number of shift parameters while decreasing scale parameters led to a decline in performance, highlighting the importance of scale parameters. Conversely, adding more scale parameters while reducing shift parameters improved performance in some cases. Nevertheless, reducing shift parameters to two or none resulted in worse extrapolation performance, indicating that shift parameters are also significant.

### Official_Comment — Authors — Official Comment by Authors (2/3)

- Note ID: `8BQHytYqF7`
- Discussion number: `7`
- Invitation: `ICLR.cc/2025/Conference/Submission13404/-/Official_Comment`


#### Title

Official Comment by Authors (2/3)

#### Comment

> **(Q5) What is the theoretical justification for removing the  amplitude term?**

This is a kind of normalization to make the effects of positional expressions even. This is not a theory, but an implementation technique. In order to make the loss converge, it was necessary to make the amplitude of the wavelet values between -1 and +1. If the amplitude of the wavelet was increased, the loss did not converge.

---
> **(Q2) Can you provide formal verification of the wavelet admissibility conditions for the proposed "Haar-like wavelets"? Or what else condition you might need to add to make it a wavelet?**

> **(Weakness)  Minor issues**

We are currently conducting a thorough review of Section 3.2. We sincerely apologize for keeping you waiting for such a long time and kindly ask for your patience a little longer. We will respectfully share the modified text with you within the rebuttal period.

[1] Wang+, ICLR2020. Encoding word order in complex embeddings

### Official_Comment — Authors — Official Comment by Authors

- Note ID: `SveBtTj7Zj`
- Discussion number: `8`
- Invitation: `ICLR.cc/2025/Conference/Submission13404/-/Official_Comment`


#### Title

Official Comment by Authors

#### Comment

Dear reviewers,

Thank you for taking the time to review this paper. We have added experiments and notes in response to some of the comments, and have reported them. As the discussion period is now coming to an end, we would be grateful if you could discuss them.

We are also currently conducting experiments on state-of-the-art RoPE improvement methods and other tasks. We will report the results of these experiments as soon as they are ready.
We look forward to discussing them with you!

### Official_Comment — Authors — Official Comment by Authors (2/2)

- Note ID: `8zFGJ3vW5Q`
- Discussion number: `9`
- Invitation: `ICLR.cc/2025/Conference/Submission13404/-/Official_Comment`


#### Comment

We sincerely appreciate the reviewers for their time, effort, and valuable feedback in evaluating our research. We have completed additional experiments and would like to share the results.
In addition, we have added a discussion on the method of position interpolation.

> **(Weakness 1) While the discussion of related methods is generally well done, discussion of RoPE scaling techniques (linear, NTK-aware) is missing. A discussion and comparison with these techniques would significantly improve the positioning of this work.**

In most cases, the method of position interpolation has been verified using large-scale language models such as llama. **Therefore, we have added a discussion to Section 7.2 (text in red), which contains experiments using llama.**
To summarize, we believe that the position interpolation method such as NTK, PI, YaRN and LongRoPE can be incorporated into our wavelet-based method. Both $\theta$ in RoPE and the scale parameter in our method represent the upper limit of the position representation. Therefore, we believe that $\theta$'s position interpolation can also be used for interpolating our scale parameter. We will consider this verification as a future issue.
Furthermore, the LongRoPE paper [Ding+, Arxiv2024] reports that performance can be improved by avoiding the first position. We think this feature is similar to our shift parameter, and we think that methods like LongRoPE can also be applied to our method.

>(**Weakness 2) The results reported in sections 6 and 7 are excellent proofs of concept but they lack comprehensiveness. Particularly, in section 7, evaluations beyond the CodeParrot dataset would be needed to thoroughly appreciate the proposed method. Furthermore, the paper only evaluates the model in term of the perplexity. For a stronger evaluation, this needs to be augmented with task-based evaluations. Apart from common language tasks, you might also want to consider associative recall tasks [2] and the long range arena [3] since the focus of this work is context length extrapolation. To further demonstrate the robustness of the proposed method, domains beyond natural language may also be considered, for example, DNA modeling [4], audio generation [4] and time series forecasting [1].**

We conducted additional experiments based on the comments we received. Initially, we considered conducting experiments on the Long Range Arena[Tay+, ICLR2021], but due to time and computing resource constraints, we were unable to do so. **Instead, we conducted additional verification on the LongBench task. The details of the experiment are described in Appendix A14.** We conducted verification on the LongBench task, which has relatively long contexts. We used the following datasets: NarrativeQA, Qasper, MultiFieldQA-en, HotpotQA, 2WikiMQA, MuSiQue, TriviaQA, SAMSum, and QMsum, and evaluated them using F1 score or Rouge-L. The experimental results showed that it was more effective than RoPE for most tasks.

---

Furthermore, it was found that there was an error in the experiment in Section 7, so it was re-evaluated. (The length that should have been reported was incorrect.) As a result of the re-evaluation, it was confirmed again that it was more effective than RoPE. The scores in Table 2 have been updated.

We sincerely apologize for the delay in reporting the results of the additional experiments. We also deeply appreciate your extremely valuable comments.
If you have any questions or require further clarification, please do not hesitate to let us know.
We look forward to hearing from you.

[Ding+, Arxiv2024] LongRoPE: Extending LLM Context Window Beyond 2 Million Tokens

[Tay+, ICLR2021] Long Range Arena : A Benchmark for Efficient Transformers ,ICLR 2021

#### Title

Official Comment by Authors (2/2)

### Official_Comment — Authors — Official Comment by Authors (2/2)

- Note ID: `eRnscD22PS`
- Discussion number: `10`
- Invitation: `ICLR.cc/2025/Conference/Submission13404/-/Official_Comment`


#### Comment

We sincerely appreciate the reviewers for their time, effort, and valuable feedback in evaluating our research. We will reply to your next comment.

>**(Q) Did you compare the length extrapolation results with other RoPE-based methods? (eg. relative works you mentioned in L47-L48)**

Although these position interpolation methods need to be discussed, we believe that they are outside the scope of our paper.
NTK, PI, and YaRN are methods that approach RoPE's $theta$, but they require fine-tuning with longer contexts after pre-training. In this paper, we are focusing on position encoding during pre-training, so position interpolation through fine-tuning is not covered.
The same applies to LongRoPE. They are not covered because they perform parameter optimization as well as fine-tuning.

**On the other hand, we think that a discussion related to these position interpolations is necessary, so we have added a discussion to Section 7.2 (text in red).**


---
> **(Weakness) The paper primarily evaluates the method on language modeling tasks. It would be valuable to see how the approach generalizes to other NLP tasks such as question answering or text summarization.**

We have also added the results of the LongBench[7] evaluation in appendix A.14. 
LongBench includes summary and QA tasks, and is evaluated using F1 and Rouge scores. We evaluated  the QA and summary tasks, which have relatively long sequences.
We used the following datasets: NarrativeQA, Qasper, MultiFieldQA-en, HotpotQA, 2WikiMQA, MuSiQue, TriviaQA, SAMSum, and QMsum, and evaluated them using F1 score or Rouge-L.
The results show that the proposed method outperforms RoPE in most cases.

---
> **(Weakness) The paper does not provide sufficient experimental evidence to support the authors' claim that Wavelet Transform can capture the dynamic changes in a sequence over positions.（L84-85）**

The assertion that wavelet transforms can capture dynamic changes in sequences is not specific to our method but rather a general characteristic of wavelet transforms themselves. This feature has been discussed in considerable detail in the book of wavelet [1], and since it is a well-established property, we omit the proof of wavelet transforms' ability to capture dynamic changes in this paper.

---

Furthermore, it was found that there was an error in the experiment in Section 7, so it was re-evaluated. (The length that should have been reported was incorrect.) As a result of the re-evaluation, it was confirmed again that it was more effective than RoPE. The scores in Table 2 have been updated.

We also deeply appreciate your extremely valuable comments.

If you have any questions or require further clarification, please do not hesitate to let us know.
We look forward to hearing from you.

[1] Ingrid Daubechies. Ten Lectures on Wavelets

#### Title

Official Comment by Authors (2/2)

### Official_Comment — Authors — Official Comment by Authors (3/3)

- Note ID: `TqiGE7IB7Y`
- Discussion number: `11`
- Invitation: `ICLR.cc/2025/Conference/Submission13404/-/Official_Comment`


#### Comment

We sincerely appreciate the reviewers for their time, effort, and valuable feedback in evaluating our research. 
We would like to inform you that we have updated the paper in response to the following comments. 

>**(Weakness 1) There exist gaps from the motivation to the proposed approach. In Section 3 and 4, the authors provide analysis on the relationship between RoPE and wavelet transform, and the properties of ALiBi like positional encodings. The ability of ALiBi to accommodate multiple window sizes is concluded as the key point for better length extrapolation performance, while RoPE is claimed to be worse. Based on these statements, a natural question is, what is the advantage of using wavelet transform? ....**

Considering the application of wavelet transformation to RoPE, the simplest formula would be the one in Appendix A13, Formula (25).
(It's a fairly large formula, so it's impossible to include it here. Please refer to the paper.)

**We implemented this approach; however, the computational cost was over five times higher than anticipated, and the pre-training did not complete.** Given the current landscape of large-scale language models, this cost is a significant concern. Additionally, there are key differences between RoPE-based wavelet transformation and RPE-based wavelet transformation, leading us to conclude that RPE-based wavelet transformation is more practical.

The differences between RoPE-based Wavelet and RPE-based Wavelet are as follows: 
- **Number of Scale Parameters:** In RPE-based Wavelet, the scale parameters can be selected up to the maximum sequence length. However, in RoPE-based Wavelet, the selection is limited to a maximum of \( d \). 
- **Memory Usage:** RoPE-based Wavelet requires a wavelet matrix that corresponds to the number of absolute positions \( m \). Consequently, the memory usage is significantly higher. Unlike RoPE-based, RPE-based Wavelet does not necessitate a wavelet matrix that matches \( m \) values, allowing the use of Tip 2 from Appendix A4, which improves memory efficiency. 
- **Absolute and Relative Positions:** When applying wavelet transforms using RoPE-based, it is necessary to use absolute positions. In contrast, RPE-based can utilize relative positions, which enhances extrapolation. 
- **Computational Cost:** Implementing wavelet transforms via RoPE-based requires processing both the query and the key, necessitating two calculations. RPE-based Wavelet only requires one computation since it processes only the query. 

---


Furthermore, it was found that there was an error in the experiment in Section 7, so it was re-evaluated. (The length that should have been reported was incorrect.) As a result of the re-evaluation, it was confirmed again that it was more effective than RoPE. The scores in Table 2 have been updated.

We also deeply appreciate your extremely valuable comments.
If you have any questions or require further clarification, please do not hesitate to let us know. We look forward to hearing from you.

#### Title

Official Comment by Authors (3/3)

### Official_Comment — Authors — Official Comment by Authors (3/3)

- Note ID: `NJC7f5sTQj`
- Discussion number: `12`
- Invitation: `ICLR.cc/2025/Conference/Submission13404/-/Official_Comment`


#### Title

Official Comment by Authors (3/3)

#### Comment

We sincerely appreciate the reviewers for their time, effort, and valuable feedback in evaluating our research. 
We have completed the additional proof and revisions for the points you kindly pointed out.

> **(Weakness) Minor issues: Inconsistent terminology between abstract ("simple Haar wavelets" in Line 18) and technical content ("Haar-like wavelets" in Sec 3.2).**

> **(Weakness) Minor issues: The phrase 'single window size' (in Line 19) should be replaced with 'fixed scale parameter' to align with standard wavelet terminology, as it specifically refers to the dilation factor in wavelet transforms.**

> **(Weakness) Minor issues: The space definition L⊭(R) in Line 135 appears to have a typographical error - it should be L²(R), the space of square-integrable functions.**

> **(Weakness) Minor issues: Some mathematical notation is not explicitly defined. For example, the superscript T in equation (3) seems to represent the length of a discrete sequence, but this isn't explicitly defined.**

Thank you very much for pointing this out. We have corrected the terms you pointed out and the typo.

> **(Q2) Can you provide formal verification of the wavelet admissibility conditions for the proposed "Haar-like wavelets"? Or what else condition you might need to add to make it a wavelet?**

Thank you for your wonderful point! We have revisited and revised the definitions of terms in Section 3.2 to ensure a clearer understanding (text in red).  Additionally, we have reconsidered the conditions for $\psi (t)$ and $\psi (t)$ in Eq.(7) to be wavelets and have corrected them to include the previously missing zero-mean property (in Appendix A15). 
Furthermore, we have added a proof in the appendix A15 to demonstrate the existence of $f(t)$ and $\delta (t)$ that satisfy these conditions.

----

Furthermore, it was found that there was an error in the experiment in Section 7, so it was re-evaluated. (The length that should have been reported was incorrect.) As a result of the re-evaluation, it was confirmed again that it was more effective than RoPE. The scores in Table 2 have been updated.

We sincerely apologize for the delay in reporting the results of the additional experiments. We also deeply appreciate your extremely valuable comments. If you have any questions or require further clarification, please do not hesitate to let us know. We look forward to hearing from you.

### Official_Comment — Authors

- Note ID: `IP3L3IdR3F`
- Discussion number: `14`
- Invitation: `ICLR.cc/2025/Conference/Submission13404/-/Official_Comment`


#### Comment

Dear Reviewers,

Thank you very much for taking the time to review our manuscript. We deeply appreciate your thoughtful comments and suggestions.
In response to the valuable feedback we received, we have revised our manuscript accordingly and conducted additional experiments to address the points raised. We have carefully documented these changes and believe that your insights have significantly improved the quality of our work. We are truly grateful for your constructive input.

If you have any additional comments or further suggestions, we would greatly appreciate hearing from you. We understand that you are very busy, and we sincerely apologize for any inconvenience caused. However, we would be most grateful if you could kindly provide any feedback at your earliest convenience to help facilitate the review process.
Thank you again for your time and effort. We greatly appreciate your guidance and look forward to hearing from you.

## Decision

### Decision — Program_Chairs — Paper Decision

- Note ID: `gCQ9Beo8bl`
- Discussion number: `1`
- Invitation: `ICLR.cc/2025/Conference/Submission13404/-/Decision`


#### Title

Paper Decision

#### Decision

Accept (Poster)


---

# Eliminating Position Bias of Language Models: A Mechanistic Approach — OpenReview 审稿全文归档

- Venue: **ICLR 2025 Poster**
- OpenReview forum: [https://openreview.net/forum?id=fvkElsJOsN](https://openreview.net/forum?id=fvkElsJOsN)
- Official paper page: [https://proceedings.iclr.cc/paper_files/paper/2025/hash/e389b15166cf98966ba058965a8c17e3-Abstract-Conference.html](https://proceedings.iclr.cc/paper_files/paper/2025/hash/e389b15166cf98966ba058965a8c17e3-Abstract-Conference.html)
- Reviewer handles are the public OpenReview pseudonyms; no attempt is made to identify individuals.
- Source: public OpenReview review dump; fields are preserved as released, including review text, rebuttal comments, meta-review, and decision where available.

## Paper Abstract

Position bias has proven to be a prevalent issue of modern language models (LMs), where the models prioritize content based on its position within the given context. This bias often leads to unexpected model failures and hurts performance, robustness, and reliability across various applications. A simple mechanistic analysis attributes the position bias to two components employed in nearly all state-of-the-art LMs: causal attention and position embedding. Based on the analyses, we propose to **eliminate** position bias (e.g., different retrieved documents' orders in QA affect performance) with a **training-free zero-shot** approach. Our method changes the causal attention to bidirectional attention between documents and utilizes model attention values to decide the relative orders of documents instead of using the order provided in input prompts, therefore enabling Position-INvariant inferencE (PINE) at the document level. By eliminating position bias, models achieve better performance and reliability in downstream tasks, including LM-as-a-judge, retrieval-augmented QA, molecule generation, and math reasoning. Notably, PINE is especially useful when adapting LMs for evaluating reasoning pairs: it consistently provides $8$ to $10$ percentage points performance gains, making Llama-3-70B-Instruct perform even better than GPT-4-0125-preview and GPT-4o-2024-08-06 on the RewardBench reasoning set.

## Review Inventory (5 Official Reviews, 25 Discussion/Comment Notes)

- Final decision: **Accept (Poster)**
- Official review ratings (review order): `6, 8, 6, 8, 5`

## Official Reviews

### Official_Review — Reviewer_1nRN

- Note ID: `a8cGPvXfhX`
- Discussion number: `1`
- Invitation: `ICLR.cc/2025/Conference/Submission7925/-/Official_Review`


#### Summary

This work addresses the position bias problem in language models (LMs), where models prioritize information based on its order in the input, affecting performance and reliability across applications. Through a mechanistic analysis, the study identifies causal attention and relative positional encodings as primary contributors to this bias. To mitigate it, the authors introduce Position-INvariant InferencE (PINE), a zero-shot approach that replaces causal attention with bidirectional attention between documents and uses attention values to determine document order. PINE effectively enhances model performance in tasks like QA, molecule generation, and math reasoning, showing notable gains in reasoning benchmarks where it surpasses even state-of-the-art models like GPT-4 in specific evaluations.

#### Soundness

`3`

#### Presentation

`3`

#### Contribution

`3`

#### Strengths

1. This paper proposes a simple yet effective and novel method to mitigate the postion bias problem in LLMs.

2. Extensive experiemnts on Llama-3 and Qwen demostrate the effectiveness of the proposed method.

#### Weaknesses

1. The ablation study of the strategy of re-assign positions is required to better discuss the potential of further improving the method.

2. There is no baseline comparison with other calibration based methods.

#### Questions

1. Do different position assigning strategies have a big effect on the downstream performance?

#### Flag For Ethics Review

- No ethics review needed.

#### Rating

`6`

#### Confidence

`3`

#### Code Of Conduct

Yes

### Official_Review — Reviewer_1Z1E

- Note ID: `fynSBxJyp7`
- Discussion number: `2`
- Invitation: `ICLR.cc/2025/Conference/Submission7925/-/Official_Review`


#### Summary

Proposed work introduces PINE, an attention mechanism that computes the attention scores in a positionally invariant manner. The method does not require any training and can be plugged into an transformer inference setup with minimal costs. Improved results in multiple benchmarks show the effectiveness of the approach.

#### Soundness

`4`

#### Presentation

`4`

#### Contribution

`4`

#### Strengths

- The approach does not require any extra training, and can be used by any attention mechanism.
- Remarkable improvements on rewardbench evals.
- Comparison with relevant baselines and experiments with multiple models performed.
- Positional invariancy is thoroughly ascertained through empirical and theoretical explanations.

#### Weaknesses

- Regarding,
> Hsieh et al. (2024) assumes that the position bias and real relevance are linear combinations and propose solutions accordingly. Different from them, we aim to eliminate the position bias from the mechanical perspective without any assumption at a reasonable cost.

Although it is claimed that this solution is better motivated than the cited one, a comparison is still required since both papers are solving the same problem and show very good improvements in similar yet different tasks.


- [This](https://openreview.net/pdf?id=gEMLMMG0m9) work debiases positional bias in the attention matrix by averaging attention values of documents at different positions. A mention of why such simple methods to debias are less relevant or a comparison is required.


- Regarding line 288,
> the extra big O computation complexity to obtain hidden states is O(nk log k),

From my understanding, for an end-to-end inference time overhead calculation, this should be multiplied by #layers in the model. And further with #heads if computation across heads is not parallelized. The equation does not capture this.
Hence for lines 518-519,
> we find the wall time of PINE is ~2x and ~8x of the vanilla inference

These numbers should depend on the size of the models being considered. Can you tabulate this for each model separately to give a better picture of overhead time? This is significant as benchmarks tend to use large models as evaluators and an inference time $\propto $ #layers limits the practical applicability of the work.

#### Questions

<none>

#### Flag For Ethics Review

- No ethics review needed.

#### Rating

`8`

#### Confidence

`3`

#### Code Of Conduct

Yes

### Official_Review — Reviewer_Qpxn

- Note ID: `69Dy9fGixL`
- Discussion number: `3`
- Invitation: `ICLR.cc/2025/Conference/Submission7925/-/Official_Review`


#### Summary

The paper presents a modification to the causal attention mechanism of decoder-based transformers. The proposed mechanism adds bidirectional attention to each document, in order to make representations position-invariant. New bidirectional attention terms are added in blocks (for the tokens corresponding to each document), which are ordered according to their importance to the document on the diagonal. The proposed importance score is based on a version of the standard attention calculation that ignores positional differences. The paper proves positional invariance and evaluates the empirical performance of the new method on a set of tasks.

#### Soundness

`3`

#### Presentation

`3`

#### Contribution

`3`

#### Strengths

1. The paper is well-written and the graphics are appealing.
2. Position-invariant representations are nice, and you obtain them without k! overhead.
3. The positional bias on RewardBench appears significant and adds credibility to your argument that position bias is an important problem.
4. Strong results for the Qwen models on RewardBench.
5. Strong results for Llama on the reasoning subset of RewardBench.

#### Weaknesses

1. You compare your method to other mechanistic methods with different goals: for example, PCW helps extend the context window; NIA gives a speed boost by making the attention computation sub-quadratic. Thus, these are both *approximate* methods. Your method, by contrast, introduces extra overhead compared to standard sampling. For a fair comparison, it is important to look for methods that are more resource-intensive than standard sampling (for example, if your method has 8x overhead, a simple baseline could involve sampling 8 permutations of the documents and running vanilla sampling 8 times, followed by some aggregation scheme like majority voting).

2. I appreciate the analysis shown in Figure 4b) but I would also like to see an analysis of the choice of importance score (see question 1. below).

3. On both RAG-QA and RewardBench, PINE does not seem to improve over vanilla sampling with the Llama models. To see this, note that the average of the GT-A and GT-B scores should give the expected performance for vanilla sampling (with randomized ordering of the two documents -- please correct me if I am wrong). On molecular generation and math reasoning, you switch between evaluating the Llama and Qwen models, which does not inspire confidence; I would like to see the same list of models evaluated for each task. Overall, your selection of tasks seems somewhat contrived. See question 2. below.

#### Questions

1. Your method gives position invariance as long as the attention orderings for each document depend only on the content of the documents rather than their positions. Thus, mathematically you can consider any function that maps a list of k documents to a permutation on k-1 documents (k-1 since the document on the diagonal is always in the last position). This class of functions is large and includes special cases like variants of your importance score (e.g., variants regarding the choice of normalization). It seems important to assess the sensitivity of PINE with respect to the choice of ordering function.

2. If position bias is as big a problem for our field as the paper argues, it should be possible to show gains on mainstream benchmarks. Consider the generic position-dependent biases in few-shot prompting, as identified by Zhao et al. in their contextual calibration paper. Could you improve, say, the recency bias of few-shot prompting with your approach? If yes, it would be nice to see this demonstrated on a mainstream LLM benchmark.

3. I am not very familiar with the Qwen series of models. Since your method yields essentially no gains for Llama on RewardBench (as noted above – please correct me if my analysis is mistaken), I wonder if the strong performance on the Qwen models has something to do with the architecture of these models. After looking into the matter, I wonder if the Qwen models’ use of sliding window attention could be the reason why the extra bi-directional attention from your method yields a benefit. What do you think? 

4. The ability to permute documents gives you a way to estimate the variance of vanilla sampling performance. Given the wide disparity between the GT-A and GT-B performances on RewardBench, did you look into randomizing the order of premises on R-GSM? This seems especially important because there are so few samples (only 95) in this data set.

#### Flag For Ethics Review

- No ethics review needed.

#### Rating

`6`

#### Confidence

`3`

#### Code Of Conduct

Yes

### Official_Review — Reviewer_K8Th

- Note ID: `CnClJSl4Y2`
- Discussion number: `4`
- Invitation: `ICLR.cc/2025/Conference/Submission7925/-/Official_Review`


#### Summary

Positional bias is a major problem for language models. Several proposals have been made in literature towards mitigating this issue. This paper proposes one method by making the inter-document attention bidirectional. Specifically, they propose their own order such that the effects of order dependence in positional encoding is minimized.

#### Soundness

`2`

#### Presentation

`3`

#### Contribution

`3`

#### Strengths

- I like this paper. they propose a bidirectional attention in the input documents to achieve input order independent attention. 
- Their results are solid. For order dependent benchmarks, they do achieve significant gains.
- While the paper is solid, I would like the authors to revise their claims of position invariance to input-position invariance.

#### Weaknesses

- I tried understanding Figure 2 many times, it was not clear to me. Please see question below. The only sensible explanation is that the attention is simulating the document under question being placed at the end of the sequence, with nearness being provided by importance scores.
- What about the n^2 importance score computations per set of n documents. How to do this when there are multiple documents?
- The position invariance is quite misleading. The correct term I believe would be importance dependent position weighting. i.e. You know the documents far away to the current document under consideration have higher likelihood of error, and hence you place them far away, such that the impact of position is mitigated. 
- Here is a counterexample for position invariance: Let us take a query which involves multi-hop retrieval. The query asks: Retrieve the home pages of all the professors working in the CS department of X university, and list out their topics of interest. Here the retrievals may be all nearly equally important. A full position invariance would require an independence with order whereas that cannot be achieved through this method. What you instead do is to rank documents by importance order and then mitigate positional dependence by placing them in that order of closeness for attention. This is not "invariance". My major concern is with the nomenclature, which claims invariance, and eliminating position bias.
- Lemma 1 may need a revision: sorting by computed importance scores introduces an implicit positional bias. While the function \( f(\text{input order}, \text{content}) \rightarrow \text{output permutation order} \) maintains input order invariance if \( f(\text{io1}, \text{content}) = f(\text{io2}, \text{content}) \) for all input orders \(\text{io1}, \text{io2}\), it imposes a new order-dependent bias in the output, meaning true position invariance is not achieved. Thus, while input-position invariance is maintained, true positional invariance is not achieved, as the sorting process introduces a new, order-dependent bias in the output.

#### Questions

- Figure 2, I understood the last row. 8 is at its original place and (4,5)> (6,7) > (2,3). 
    - Rows 6 and 7 -- (6,7) is at its original place and (2,3) > (4,5)
    - Rows 4 and 5 -- (4,5) should be at their original place, but (6,7) is! Why revise the position of this document to the end when both the documents can be (equally) at small distance from the document under consideration?
- I only understood Figure 2 after going through paragraph 2 in lemma 1. Please consider revising the write up with the document order for computation of each of the rows of the attention.

### Suggestions:
- Move paragraph 2 from Lemma 1 proof from appendix to main paper to explain your method.
- Consider revising the words position invariant to input-position invariant. Please also consider revising the claim that the position bias of LLMs is eliminated (in the title of the paper). From my best understanding, it is mitigated through a reordering and not eliminated.

#### Flag For Ethics Review

- No ethics review needed.

#### Rating

`8`

#### Confidence

`4`

#### Code Of Conduct

Yes

### Official_Review — Reviewer_azrE

- Note ID: `l4cOSOy8Jy`
- Discussion number: `5`
- Invitation: `ICLR.cc/2025/Conference/Submission7925/-/Official_Review`


#### Summary

The authors introduce a method to de-bias a positional dependence for concatenated documents in the context of a LM and prove its supremacy against other de-biasing baselines.

They also compare against vanilla inference, but I cannot tell whether the numbers are actually beneficial.

#### Soundness

`2`

#### Presentation

`1`

#### Contribution

`3`

#### Strengths

- very relevant problem
- the recency and primacy biases are interesting.

#### Weaknesses

- Figure 2 is confusing, since the documents change color when you move them (in the right plot). So it looks like the $D_3$ in the first row is the tokens [2,3], but in the second assignment iteration suddenly $D_3$ is [4,5]? It seems inconsistent with the last row, where importance ordering is $D_2 > D_3 > D_1$, but they are not ordered in that way.
- I don’t think you can claim that you pinpointed the causes of positional bias in transformers. Everybody knows what those are.
- Please Correct me if I am wrong here.
The proof in App B shows that the softmax term is invariant under inter-document permutation before the ordering takes place, but Lemma 1 takes the QKV invariance as a given.
How I think Lemma 1 and the proof should go:
Lemma 1: “The PINE algorithm makes $H_{Pine}$  document-position invariant”
Proof: “The algorithm sorts the documents by a document-position invariant metric and concatenates them. Therefore, the resulting concatenation  is invariant. $H_{Pine}$ is only a function of that concatenation, therefore $H_{Pine}$ is invariant. Proof ends”.
- I can’t tell if the numbers are good. The method outperforms other methods of debiasing, but it does not seem to do super well against Vanilla (shuffle) in Figure 4 for instance. Given that you shuffle reward bench, is Vanilla in Table 2 also implicitly Vanilla (shuffle)? It is overall confusing to me, but I admit that this is an opinion.

#### Questions

- Please discuss the relationship to NoPE (https://arxiv.org/pdf/2305.19466) in your related work
- I find the importance measure that you introduce a little bit weird if you make it independent of positional bias, since you will later calculate attention with positional bias in there, which might change the importance quite a bit. Have you found your measure of document importance, without positional bias, to be monotonic with an importance measure that calculates the attention with positional bias, but putting each document candidate at the same position, right before the currently decoded document for instance?
- I am overall confused by the introduction of bidirectional attention. Given that you calculate importances independently of relative document-position, and you reorder anyway, why do you have to have an attention in the forward direction, instead of simply always putting the document you are decoding at the last position? I might have understood this wrong, the first half of Page 5 is a little confusing to me. When following the concrete example in the proof, there is no bidirectional attention, right?
- Can you comment on the downsides of making keys query dependent (L235)? This seems like you will loose parallelism, at the very least the option to KV-cache.
- Can you comment on your intuition as to why SP and PCW are worse than your approach?
- You say you introduce two other debiasing baselines, permutation and calibration. Is permutation the same as Vanilla (shuffle) in Figure 4? Why are the numbers for it not in any tables?
- Table 2: I don’t fully understand the baselines. IIUC, the GT is either at A or at B, which is why you say 50% random guess baseline, i.e. there is no position C. Why is Vanilla (without qualifier) worse than both Vanilla (GT at A) and (GT at B)? it is not always one of the two, which would lead to some kind of average?
- what do you mean by reasoning “pairs” in 418ff?

#### Flag For Ethics Review

- No ethics review needed.

#### Rating

`5`

#### Confidence

`3`

#### Code Of Conduct

Yes

## Meta-Review

### Meta_Review — Area_Chair_Kcpq

- Note ID: `gHkEXCVYzb`
- Discussion number: `1`
- Invitation: `ICLR.cc/2025/Conference/Submission7925/-/Meta_Review`


#### Metareview

This paper addresses position bias in transformers in situations where generation is supposed to be conditional on a set of input documents.The proposed approach, dubbed PINE, first re-arranges inputs based on relative pairwise document importance scores as measured via aggregate attention scores, which then results in invariant generation given that re-ordering is carried out prior to generation. This comes at the cost of inference overhead since the re-ordering step requires computing importance. PINE operates during inference only and incurs no extra training cost.

The proposal is evaluated in a number of settings generally showing improvements, however it's a bit unclear how inducing position invariance improves so much the performance in reasoning benchmarks such as in the results shown in tables 2 and 3 for RewardBench. If the assumption is that position-dependent models may miss relevant context due to its position, then that could have been verified by exhaustively checking all possible orderings of input documents (perhaps for a subset of the experiments).

The manuscript has some presentation issues, and the method description in section 3.3. as well as the definition of importance scores are a bit confusing and not clear enough. Another limitation worth highlighting is the fact that PINE is limited in scope to settings where input contexts have well defined boundaries, e.g., it is comprised of a set of documents. In a situation where one has a single large document, position bias could still affect performance since relevant information could occur in parts of the document models tend to ignore. Sub-document re-ordering seems to be required in such a case. Moreover, the evaluation, while extensive in a few aspects, has some limitations that are worth noting. For instance, multi-hop settings where information required to answer queries are spread across multiple documents should have been covered.

In summary, the approach offers a strategy to trade compute for order invariance, and consequently boost performance, in settings where generation is conditional on input context and information needs to be retrieved from a set of documents. Although some limitations remain as discussed in the paragraph above, enough evidence is presented to support the authors's claims and the approach would be of value to the community.

#### Additional Comments On Reviewer Discussion

The main overlapping concerns raised by reviewers revolved around presentation issues and lack of clarity. Authors seem to have addressed those concerns well during the discussion phase, and reviewers were mostly satisfied with the improvements, raising their scores. Although some lack of clarity remains as noted above, and the manuscript would benefit of something like a pseudo-code outlining the inference approach.

## Rebuttal and Discussion Comments

### Official_Comment — Authors — Rebuttal (1/2)

- Note ID: `bLtVg5JFJm`
- Discussion number: `7`
- Invitation: `ICLR.cc/2025/Conference/Submission7925/-/Official_Comment`


#### Title

Rebuttal (1/2)

#### Comment

Thanks for your careful review. We are glad that you find our research interesting and relevant. We answer your questions with bullet points:

> W1: Figure 2 is confusing

Numbers in Figure 2 are positions when tokens serve as **keys**, and the numbers of the diagonal are also positions when tokens serve as **queries**. The importance score is computed between **queries** and **keys**. Therefore, the ranking of the documents’ importance differs for different queries, and the claim “The importance ordering is D2 > D3 > D1” is incorrect since this is only the case for Token 8. The [2,3] and [4,5] correspond to the first two formulas in Figure 2, so they are consistent. We mention all these parts in our paragraph of line 219 and 226.

> W2: pinpointing the causes

We aim to emphasize that they are the **only** two parts that cause position bias, which previously no people explicitly discussed to the best of our knowledge. This discussion offers the necessary background to explain why we only focus on these two parts. We do not address this as our core contribution. In the revised paper, we remove words like “our” and use “revisit” in the abstract and the end of the introduction and highlight the modification in the green color.

> W3: Alternative proof

Both yours and ours are correct understanding. The core of Lemma 1 is to say $H\_{pine}$ does not include any new position information, regardless of whether inputs have it. We let input QKV be position-invariant in Lemma 1 to address the premise of mathematical induction, which is used in the proof of theorem 1. 


> W4: The results

Yes, Vanilla means Vanilla(shuffle) in Table 2.  We apologize for the confusion and have changed the annotation accordingly. PINE performs similarly to vanilla inference in Figure 4, which we discuss in line 479 and we hypothesize the reason is that ordering becomes difficult for LMs when there are more documents. However, our results show that PINE consistently performs better in RewardBench, molecule generation, and math reasoning.


> Q1:Discuss NoPE

Thank you for the suggestion. We’ve added it to the updated version to Section 2 (highlighted with green color). NoPE removes the positional encoding, which we find the results are very low in our pilot experiments.


> Q2: The importance measure

We hope we understand your question correctly: Will the importance ranking change before and after applying position encoding? The answer is “sometimes”. The ideal solution is no longer applying position encoding (i.e., NoPE). However, this solution leads to much worse results in our preliminary experiments. Therefore, we keep positional encoding and try to make it not affect the importance ranking: re-order descendingly since positional encoding has recency bias. In this way, RoPE mostly respects the ranking order. Nevertheless, RoPE has an oscillation feature at the microscopic level so the importance ranking is not guaranteed unaffected. This phenomenon does not affect our proof of “elimination.”


> Q3: The bidirectional attention

As shown in Figure 2, the reorder determines the position of documents, and bidirectional attention only applies to query documents (i.e., for token 8, the attention is causal), which lets a query document “see” all other documents when computing attention score. In the proof, bidirectional attention is still used since when computing H_i, query document i need to see all documents.

> Q4: the dependence of keys and queries

This does not affect the KV cache. The conventional KV cache is to cache KV after applying RoPE. In our implementation, we cache KV without RoPE and apply RoPE when needed. In the inference, the bottleneck is IO (loading from HBM to SRAM) instead of compute, and RoPE only brings lightweight computation. Therefore, PINE does not affect KV cache effectiveness and efficiency. The downside is that the implementation requires more engineering, and since we are not experts in writing efficient codes, we still keep the “for” loop in the implementation, which makes the real wall time slower.

> Q5: About PCW and SP

We hypothesize that PCW and SP introduce more OOD operations. For example, PCW and SP assign different tokens to the same positions. However, LM is not trained to handle such operations. Our method will always assign different tokens to different positions when serving as keys in the “eye” of each query. Another reason is that they lose contextual information, while PINE keeps it.

### Official_Comment — Authors — Rebuttal (2/2)

- Note ID: `xd2yXuyBLU`
- Discussion number: `8`
- Invitation: `ICLR.cc/2025/Conference/Submission7925/-/Official_Comment`


#### Title

Rebuttal (2/2)

#### Comment

> Q6: About Permutation Baseline

We apologize for not making this obvious. We did not include this in tables due to space limits, and you can find results in the L459 paragraph. In short, the calibration-based method has non-sense outputs, which we believe is caused by its strong assumption. Permutation-based methods perform worse than ours. We’ve highlighted permutation results in our updated version by adding a bolded sentence at the beginning of the paragraph (L459), and we highlight them with a green color for you to locate.

> Q7: About Vanilla (Shuffle) performance v.s. Vanilla (Gt at A and B)

This is a binary classification problem, but Vanilla (Shuffle) is not necessarily the average of two extreme cases. It could perform even worse than a random guess if the model capability is limited. The key problem is that the model may prefer different positions for different problems. For example, if the dataset has ten questions and the model prefers option A the first five questions and B for the rest, then GT at A and GT at B will all have 50% acc. However, Vanilla(shuffle) may have up to 100% and low to 0%, depending on the concrete shuffle.


> Q8: The meaning of reasoning pairs.

We apologize for the confusion. This is a typo and should be “reasoning problems in RewardBench.” We’ve corrected this in the updated version in L424.


Overall, we sincerely thank you for your suggestions. We are happy to modify the paper further if you have any suggestions for making the presentation more precise. Please let us know if you have follow-up questions, and we will be glad to discuss.

### Official_Comment — Authors — Rebuttal (1/1)

- Note ID: `XWxGkBhK1c`
- Discussion number: `9`
- Invitation: `ICLR.cc/2025/Conference/Submission7925/-/Official_Comment`


#### Comment

We thank you for your insightful advice and suggestions! We are glad that you like our paper and find our results solid. We answer your questions below:

> W1 & Q1: Figure 2 Understanding

Your overall understanding is correct: important documents are placed near to the end of the sequence in PINE. 

For your understanding of row 4,5, your suggestion is a better choice than ours if models are not trained bi-directional (e.g., Prefix LM instead of causal LM). Unfortunately, most modern LMs are trained casually, which means the query token can only see previous tokens. Document 2 is the query document in rows 4,5; suppose its position is (4,5), and Document 3 has the position (6,7). Then, attention computation will make query document 2  see “future” document 3, which is not seen and trained by LMs and causes poor performance.

> W2: Importance score computation

This is the same normal attention computation and therefore can be computed in parallel. We just follow the pipeline of attention score, with the only modification being that Q and K are not encoded with position embedding.

> W3 & W4 & W5: About our claim, proof and terminology

We believe both ours and your understanding is correct. The difference is that we stand on different perspectives. We say “position-invariant” from the input-output perspective: results remain unchanged regardless of input orders. Your proposed claim stands on the method implementation perspective.

The “elimination” also originates from the input-output perspective. Our method still uses position encoding, and we find that removing position encoding such as NoPE yields poor performance in our preliminary experiment.

We agree to add your understanding to the main body of the paper and clarify more about “elimination” and “invariance” in our paper. Please see Section 3.3 green texts.


> Q1 & S1 & S2: presentation and terminology improvement

We thank you for your suggestion and moved the Lemma 1 to our main body. As our response to your W3-5 promises, we’ve added clarification on the terminology on “elimination” and “invariant” is from an input-output perspective and address that internally, the position encoding is still used. Please see Section 3.3 green texts.

#### Title

Rebuttal (1/1)

### Official_Comment — Authors — Rebuttal (1/1)

- Note ID: `K0O5pIkaHQ`
- Discussion number: `10`
- Invitation: `ICLR.cc/2025/Conference/Submission7925/-/Official_Comment`


#### Title

Rebuttal (1/1)

#### Comment

We thank you for your detailed and helpful suggestions and are glad you found our presentation clear and the results strong. We address your concerns below.

> W1: Baseline comparison

PCW and NIA were initially designed for long ICL tasks, but they can all mathematically guarantee the elimination of position bias (similar to our proof). Therefore, they are not approximations of elimination (hopefully, we correctly understand the meaning of approximation in your context; please correct us if it is wrong). We compare our methods with permutation and calibration baselines in the paragraph of L459. In short, the calibration-based method has non-sense outputs, which we believe is caused by its strong assumption. Permutation-based methods perform worse than ours. Note that the permutation baselines are only conducted in RewardBench since, in RAG, the computation cost of O(10!) and O(20!) is intractable. We apologize for not making this clearer, and we’ve highlighted permutation results in our updated version.

> W2 & Q1: the different choice of importance score

This is a very interesting question, and indeed, there are a lot of available mappings. In Figure 4, we compare our importance score with the reverse score, random scores, and positions unchanged. We use attention scores as importance scores, which aligns with research on attention interpretability work such as retrieval head [1]. Since it is not tractable to test all possible functions, we exclude alternatives that lack supporting evidence (for example, overlapping of bigram with question), or need extra efforts (for example, training an NN to do ranking). Therefore, we do not find other plausible importance scores to compare besides Figure 4b. We are willing to add more experiments if you have any suggestions.

[1] Wu, Wenhao, et al. "Retrieval head mechanistically explains long-context factuality." arXiv preprint arXiv:2404.15574 (2024).


> W3 & Q2: Performance

* The shuffle results are average of GT at A and GT at B

Vanilla (Shuffle) is not necessarily the average of two extreme cases. The key problem is that the model may prefer different positions for different problems. For example, if the dataset has ten questions and for the first five it prefers A and second five it prefers B, then GT at A and GT at B will all have 50% acc. However, Vanilla (shuffle) may have up to 100% and low to 0%, depending on the concrete shuffle. In Table 2, the shuffled version is the third line, and our methods show consistent improvements across Llama/Qwen and different sizes.

* Switching models between models in molecule generation

We initially hope to show that our experiments on diverse models and tasks work, and we apologize for causing such confusion. The llama and Qwen models have no differences in the molecule generation tasks because we train from **scratch**, and the two models have indistinguishable architecture differences (if not all the same). For the math dataset, the llama results are shown below, where PINE performs better than the baseline.

|Method| 8B | 70B | 
|-|-| -|
|Vanilla| 63.2 | 84.2 |
|PINE|**64.2** |**86.3** |

* Few-shot prompting experiments

We conducted a few-shot experiment on the ARC dataset with the ICL example selection same as in previous work [2]. Experiments were conducted on the Llama 3 8B model, and PINE showed the best performance.

|Method| Vanilla | Permutation (Cylic) | PINE | 
|-|-| -| -|
|0-shot| 80.34| N/A | N/A |
|3-shots|80.34 | 79.48 | **80.69**|
|5-shots|79.91 | 80.34 | **80.52** |



[2] https://arxiv.org/abs/2309.03882

> Q3: Llama performance and Qwen architecture

We hope our response to W1 clarifies your first subquestion about Llama's performance. In short, we need to compare the last two rows (i.e., PINE and Vanilla), and our methods have noticeable improvements in Llama. The Qwen architecture uses the same architecture as Llama, such as GQA, causal attention, RoPE, layernorm, and MLP. We do not use sliding window features, as all inputs/outputs do not exceed window length.

> Q4: R-GSM variance

We agree that R-GSM has a high variance due to the small data size, and we think this experiment is more like a bonus experiment. Since each problem in R-GSM may have a different number of conditions, and we do not have prior information about the best and worst order of conditions, we are not able to present numbers such as GT at A and GT at B.  If we random shuffle several times and take the average of vanilla inference performance, then we have the following results (PINE vs Avg of Vanilla) :  82.1 vs. 79.5 in Qwen 110B, and 50.5 vs. 46.8 in Qwen 7B.

We conclude that our methods generally have better or on-par performance while maintaining 0 variances, whereas the vanilla inference encounters a high variance ( 6%~7% deviation in accuracy on average).


We hope our responses address your concerns, and we are happy to discuss any follow-up questions.

### Official_Comment — Authors — Rebuttal (1/1)

- Note ID: `CutLC3UGwB`
- Discussion number: `11`
- Invitation: `ICLR.cc/2025/Conference/Submission7925/-/Official_Comment`


#### Title

Rebuttal (1/1)

#### Comment

We thank you for your helpful advice! We sincerely appreciate your favor to our paper and are happy to see you find our paper general and useful.

> W1: Comparison with Hsieh et al.

This is indeed a relevant method. However, we are unable to deliver results since it does not publicly release codes yet. We also face difficulties implementing the method since paper cannot cover every detail due to length limits. 

For example, the paper mentions how to manipulate attention when decoding tokens but does not mention details about encoding documents themselves (i.e., the first forward pass when decoding).


> W2: Averaging importance score

We briefly talked about using averaging over summation in our paper (L231) to prevent putting higher scores on longer documents. In our pilot experiments, we find that summation converts models from position bias to length bias. 

We also try maximum instead of averaging and find this method usually has noticeably worse performance than averaging possibly due to noises brought by unimportant tokens. Therefore, we chose averaging in our final version. 

Thanks for pointing out this discussion. We’ve added this discussion to the paper's Section 3.3 footnote, which is highlighted in green.

> W3: Time cost w.r.t. Layers

We want to point out a slight mistake: The time gain ratio compared with vanilla inference is independent w.r.t. Layers. For example, if each layer of PINE needs 2x time, then in total, PINE still needs 2x time regardless of layers because PINE and vanilla inference have the same number of layers.

However, we agree that the architecture parameters may affect the ratio factor, such as the compute balance between attention and FFN.

In our experiments, we find that if the number of input documents is 2, and time is ~2x across sizes consistently, and if the number of input documents is 20, the time is ~8x across sizes consistently. We can only get a rough ratio since the real ratio depends on factors such as GPU types, connection type, model sharding type, batch size, server burden, etc. 


We hope our response addresses your concerns and are happy to discuss if you have any follow-up questions!

### Official_Comment — Authors — Rebuttal (1/1)

- Note ID: `56JMWLJzka`
- Discussion number: `12`
- Invitation: `ICLR.cc/2025/Conference/Submission7925/-/Official_Comment`


#### Title

Rebuttal (1/1)

#### Comment

We thank you for your advice! We are happy that you find our method simple, effective and useful. We address your concerns below:

> W1 & Q1: The ablation study of position re-assignment

In Figure 4, we compare our re-assignment method with the reversed one,  random shuffle one, and positions unchanged one. Results show that our methods achieve the best. 

Since testing all available re-assignment methods is impossible, we exclude alternatives that lack supporting evidence (for example, re-assigning by counting overlapping of bigram with question), or need extra efforts (for example, training an NN to do re-assignment). Therefore, Figure 4b contains all plausible methods that we can come up with. We are willing to add more experiments if you have any suggestions.

At last, we hope to point out that our re-assignment method aligns with research on attention interpretability work such as retrieval heads [1].

[1] Retrieval Head Mechanistically Explains Long-Context Factuality


> W2: Comparison with calibration-based methods

We thank for pointing out this discussion. First, we compare a calibration method [2] in L459 and find it yields non-sense results due to its strong hypothesis. Another calibration method that is relevant to our method is [3]. However, we are unable to deliver its results since it does not publicly release the codes. We also face difficulties implementing the method ourselves since papers cannot cover every detail: specifically, the paper mentions how to manipulate attention when decoding tokens but does not mention details about encoding documents themselves (i.e., the first forward pass when decoding).



[2] Calibrate Before Use: Improving Few-shot Performance of Language Models

[3] Found in the Middle: Calibrating Positional Attention Bias Improves Long Context Utilization


We hope our explanation addresses your concerns, and we are happy to discuss them with you if you have any follow-up questions.

### Official_Comment — Authors — General Rebuttal

- Note ID: `QF64RzVDl1`
- Discussion number: `13`
- Invitation: `ICLR.cc/2025/Conference/Submission7925/-/Official_Comment`


#### Title

General Rebuttal

#### Comment

We thank all reviewers for their time and hard work in reading and reviewing our paper. We are happy that reviewers find our method interesting (azrE), useful (1Z1E, 1nRN), have strong results (Qpxn, K8Th, 1nRN), and express favor to our paper (1Z1E, K8Th).

We address reviewers' concerns separately and update our paper to incorporate their suggestions. All modifications are highlighted in green so that reviewers can locate them better. Besides, the main modification is to shorten Section 4.1 under the 10-page limit.

### Official_Comment — Reviewer_K8Th — Inc of score

- Note ID: `rXI6W8egpj`
- Discussion number: `14`
- Invitation: `ICLR.cc/2025/Conference/Submission7925/-/Official_Comment`


#### Title

Inc of score

#### Comment

Thanks authors! I was already in favour of the paper, but improving "position-invariance" terminology to eliminating bias, clarifies my major concern.

Increasing my score. Good luck!

### Official_Comment — Reviewer_azrE

- Note ID: `Mj5n8VVtEl`
- Discussion number: `15`
- Invitation: `ICLR.cc/2025/Conference/Submission7925/-/Official_Comment`


#### Comment

Thank you for your clarifications. Some things became clearer.

I maintain my position on the proof, it can be put into a few lines, since it is rather trivial. However, as another reviewer suggested, the example you walk through in the proof can be used to further clarify the method.

Figure 2 remains confusing to me despite your attempts at clarification. Can you explain why there are white boxes on every second line after on the one-off diagonal? Maybe you could write down the resulting attention expressions for one or two of the lines in this figure.

Maybe it helps to compare against a method that seems close, that I have understood: When reading more carefully through the related works, I noticed Peysakhovich & Lerer (2023) also sorts inputs according to their importance. Are you comparing against it? That baseline seems quite relevant and shows good improvements in the paper. If I understand correctly, the biggest difference between this and your work is that they calculate attention __with__ inter-document position information, and then they sometimes re-sort to account for that. It looks like that re-sorting can have a big effect, which relates to my initial Q2.

### Official_Comment — Authors — Follow-up responses

- Note ID: `liMoQ88NLA`
- Discussion number: `16`
- Invitation: `ICLR.cc/2025/Conference/Submission7925/-/Official_Comment`


#### Title

Follow-up responses

#### Comment

We thank you for your prompt reply. We clarify your further questions below.

> The proof

We agree that the proof can be relatively simple. The proof actually does not necessarily need to be written as it is a direct application of the symmetry principle, which is broadly used in natural sciences such as Physics. However, we think the CS community may be unfamiliar with symmetry principle so we write a detailed proof with examples instead. 

> White boxes in our attention

As discussed in L213-L215, we use bidirectional attention across documents and causal attention inside each document. The white box is because it is inside one document.

For line 4, the query is the first token of the second document, and it should see document 1, document 3, and itself when computing attention matrix.

For line 5, the query is the second token of the second document, and it should see document 1, document 3, the first token of document 2, and itself when computing the attention matrix.

> Comparision with Peysakhovich & Lerer (2023)

First, we hope to point out that Peysakhovich & Lerer (2023) cannot eliminate the position bias since their method does not follow the symmetry principle. Another difference besides what you point out is that we use bidirectional attention, whereas they use causal attention as usual.

Second, our method includes re-sorting, too. The difference is that we only do one re-sorting instead of periodically re-sorting. Figure (2) numbers are the results of re-sorting, which we discuss in L226. Figure 4(b) also discusses several variations of re-sorting techniques and shows that ours performs best, which addresses the effectiveness of re-sorting.

Third, that paper does not release codes and lacks some details for us to implement ourselves. For example, how is the document attention normalized? Is the process running for every decoding step or just once? Therefore, we can not faithfully reproduce the results.

Lastly, we want to re-address your Q2. Peysakhovich & Lerer (2023) could fall into local optima due to position encoding, and our importance ranking may be affected by RoPE's local oscillation nature (See RoFormer https://arxiv.org/pdf/2104.09864 Figure 2 for better understanding). Neither of these methods is perfect, and we both show the importance of re-sorting in our paper (Figure 4 b of ours and Figure 5 of Peysakhovich & Lerer (2023) ), though we use different re-sorting techniques.


We hope our responses make things clearer and thanks again for reading our rebuttal.

### Official_Comment — Reviewer_azrE

- Note ID: `RcnDs14EPH`
- Discussion number: `17`
- Invitation: `ICLR.cc/2025/Conference/Submission7925/-/Official_Comment`


#### Comment

The CS community is most certainly familiar with symmetry.

Thanks for the clarification about the white boxes, that helped. I think I also understood the numbers now, and this effectively comes down to re-sorting to calculate the attention for each token (although that would of course be a bad implementation)

Peysakhovich & Lerer (2023) are not far from eliminating position bias, they only need to calculate their sorting metric without RoPE like you do.
Then, the remaining difference to your method is the bidirectional attention. Did you ablate what kind of difference that makes? Apologies, I should have asked this in the first review, but I have just recently realized how close Peysakhovich & Lerer (2023) your method is.

### Official_Comment — Authors

- Note ID: `TpJdkbDQNK`
- Discussion number: `18`
- Invitation: `ICLR.cc/2025/Conference/Submission7925/-/Official_Comment`


#### Comment

Thanks for reading our rebuttal and your quick reply.

> About the symmetry principle

To the best of our knowledge, this term is not frequently used in CS papers as in Physics papers, so we have to be conservative when writing proofs. As a preliminary statistical evidence, only 0.5% (3479 out of 652,604) CS papers in arxiv mention "symmetry", whereas the portion is 7.9% (112,784 out of 1,432,095) in Physics papers.

However, we are more than happy to add a one-sentence illustration about the symmetry principle to our paper after we agree on other points (just to avoid back-and-forth revision).

> Peysakhovich & Lerer (2023) are not far from eliminating position bias, they only need to calculate their sorting metric without RoPE like you do. 

No, they can not eliminate position bias even if they calculate without RoPE. The causal attention breaks the symmetry.

> Is bidirectional useful?

Of course, the red dashed line in Figure 4(b) outperforms the blue dashed line with ~1% Accuracy, showing that simply adding bidirectional attention without our re-sorting (and therefore, the computation overhead of re-sorting is discarded) is useful.

> but I have just recently realized how close Peysakhovich & Lerer (2023) your method is.

Thank you for your question. We hope to address our differences:

* We use bidirectional attention

* Because of the bidirectional attention, our re-sorting is different from their re-sorting, we need to design a position assignment strategy after re-sorting in accordance with our bidirectional attention.

* We need to exclude RoPE when computing importance scores.

* We do not need to periodically re-sort to find a fixed point.

* We have theoretical guarantees.

To sum up, although both methods use "re-sorting," the re-sorting itself and the ways to incorporate the re-sorting results differ greatly.

Thanks again for your feedback; we hope we address your concerns and questions.

### Official_Comment — Authors — Response to Reviewer K8Th

- Note ID: `M6xbeZjiCN`
- Discussion number: `19`
- Invitation: `ICLR.cc/2025/Conference/Submission7925/-/Official_Comment`


#### Title

Response to Reviewer K8Th

#### Comment

Thank you for your helpful advice and for reading our rebuttal! We are glad that our responses address your concerns. 

We will officially incorporate your suggestions into our paper's next revision.

### Official_Comment — Reviewer_Qpxn

- Note ID: `MFyYW6l4iO`
- Discussion number: `21`
- Invitation: `ICLR.cc/2025/Conference/Submission7925/-/Official_Comment`


#### Comment

I appreciate the additional results addressing my previous questions. I'd like to discuss two points:

**Regarding baselines**: when I describe PCW and NIA as "approximate," I'm referring to how they reduce computational requirements by simplifying the attention mechanism - specifically by eliminating inter-document attention. These methods prioritize computational efficiency to handle longer sequences. Your approach, however, maintains full quadratic attention plus additional computational costs, suggesting it's not designed for processing longer sequences. Given that you're testing on standard-length sequences where computational compromises aren't necessary, it would be more appropriate to compare against methods intended for standard lengths.

About permutation baselines: the methods you cite (contextual calibration and PriDe) try to learn order bias in order to remove it at inference-time. I'm suggesting a simpler baseline:

answers = [ ]

Sample k permutations [σ₁, σ₂, ..., σₖ]

**for** permutation in [σ₁, σ₂, ..., σₖ]:

&nbsp;&nbsp;&nbsp;&nbsp;permuted_documents ← permute(documents, permutation)

&nbsp;&nbsp;&nbsp;&nbsp;answer ← run_model(query, permuted_documents)

&nbsp;&nbsp;&nbsp;&nbsp;answers.append(answer)

**return** majority_vote(answers)

This method requires no training and should not give "rubbish" outputs as you observe with the other permutation-based methods. It introduces a k-fold computational overhead, but helps reduce variance from positional bias. You can set k=8 to match your method's computational overhead. When majority voting isn't applicable, you can substitute another aggregation method or random selection of the answer.

**Regarding shuffle performance**: I do not completely follow your explanation. I believe we agree that vanilla shuffle's expected value is (1/2)*E[GT-A] + (1/2)*E[GT-B]. Consider randomly sampling 1,000 examples from the data-generating distribution - approximately 500 will be assigned to GT-A and 500 to GT-B. The expected values for GT-A and GT-B match what you'd get from independent 50% resamples of your original dataset, so E[shuffle] = (1/2)*E[GT-A] + (1/2)*E[GT-B]. Any deviation from this theoretical expectation indicates the impact of finite sample effects. Would you agree with this analysis?

Since your experiments provide performance data for both GT-A and GT-B on each question, you can simulate vanilla shuffle computationally by randomly selecting either GT-A or GT-B results for each question. This eliminates the need for additional model runs - you can simply randomly select from your existing correctness data. For a single simulation, you flip a coin for each example to choose GT-A or GT-B. Running this simulation, say, 10,000 times should provide good estimates of the mean and variance of vanilla shuffle. Could you report the performance of vanilla shuffle as mean +/- standard deviation, rather than as a single number?

### Official_Comment — Reviewer_azrE

- Note ID: `gmsinQ67pa`
- Discussion number: `22`
- Invitation: `ICLR.cc/2025/Conference/Submission7925/-/Official_Comment`


#### Comment

Re bidirectionality:
I don't see the blue dashed line in Figure 4b. Do you refer to the blue dashed line in 4a?

Re position invariance:
I want to decode a new token from my current document. Now I look at all other documents (they are in my context) and calculate a aggregated importance measure that does not depend on their respective positions (i.e. without rope).
Now, I define position invariance (document-wise): f(set of documents D, order σ_j) = f(D, σ_i) for all permutations σ_i,σ_j.
If f includes a sorting S (depends on D implicitly) via position invariant metric (since rope not included), then for all σ_i, f(D, σ_i) = g(D, S(σ_i)) = g(D, σ_0) = g(D, S(σ_0)) = f(D, σ_0). σ_0 is the sorted permutation. I am confused, what am I missing?


The baseline of Peysakhovich & Lerer (2023) is still important to compare against I believe.

### Official_Comment — Authors

- Note ID: `Hc2Kwr5FH5`
- Discussion number: `23`
- Invitation: `ICLR.cc/2025/Conference/Submission7925/-/Official_Comment`


#### Comment

Thanks for reading our rebuttal. We are happy that we addressed your questions, and here are our responses to your new feedback:

> The permutation baseline

We apologize for not making the permutation baseline clear. In the L459, we report two methods: calibration methods does not work, and permutation (which is exactly what you describe) methods that underperform ours. Specifically, the results  on Rewardbench are:

| Method | Llama 3 8B Instruct | Qwen 1.5 7B Instruct |
| - | - | - | 
|Vanilla  | 64.8 | 60.9 |
| PINE| **66.7** | **61.5** |
| Permutation | 65.9 | 61.3 |

> The variance

Thanks for pointing out this discussion, and sorry for not fully understanding your meaning previously. We follow your suggestion and re-compute the variance of vanilla inference under different shuffle. 

Since RewardBench assigns different weights to different subsets, and our originally dumped results do not record the belonging of each sample, we will use an equal average to deliver the results here (therefore, you may find the results are not the same as the Paper reported). Results still show that PINE is effective when all samples are equally averaging.

Llama 3 Series:

| Method |8B | 70 B |
| - | - | - |
|Vanilla (GT at A)  | 70.7 | 80.1 |
|Vanilla (GT at B)   | 65.5 | 76.5 |
|Vanilla (Shuffle)  | 68.1 $\pm$ 0.4 | 78.3 $\pm$ 0.3 |
| PINE| **70.5** | **81.5** |

Qwen 1.5 Series:

| Method | 1.8B |  4B | 7B | 32B | 72B | 72B (Qwen 2.5) | 110B |
| - | - | - | - | - | - | -| -|
|Vanilla (GT at A)  | 39.1 | 32.6 | 60.9 | 75.2 | 80.3 | 90.0 | 88.0 |
|Vanilla (GT at B)   | 63.1 | 72.2 | 57.2 | 74.5 | 69.0 | 82.5  | 74.7 | 
|Vanilla (Shuffle)  | 51.1 $\pm$ 0.5 | 52.4 $\pm$ 0.6 | 59.1 $\pm$ 0.5 | 74.8 $\pm$ 0.4 |  **74.6** $\pm$ 0.4 | 86.2 $\pm$ 0.3 | 81.3 $\pm$ 0.4 |
| PINE| **55.4** | **56.6** | **63.0** | **78.2** | 74.3 | **87.5** |**85.0** |

### Official_Comment — Authors

- Note ID: `9qNoxUEtyY`
- Discussion number: `24`
- Invitation: `ICLR.cc/2025/Conference/Submission7925/-/Official_Comment`


#### Comment

Thanks again for your quick reply. We are delighted you are engaging with us frequently!

> Blue line

Yes, and they have ~1% Acc difference.

> Position invariance

Your most understanding is correct. The minor mistake is "calculate an aggregated importance measure that does not depend on their respective positions (i.e. without rope)." Causal attention implicitly contains position information; therefore, the importance score still depends on positions if causal attention is used instead of bidirectional attention or PCW (masked-out inter-document attention). All other parts you mentioned are correct.

Therefore, that's why we need bidirectional attention (PCW does not show good performance according to our experiments reported in our paper)


> The baseline of Peysakhovich & Lerer (2023) is still important to compare against I believe.

Yes, we agree this baseline is important. However, the paper does not release code and we cannot obtain results.



Again, we thank you for your valuable feedback and hope our response could clarify your questions.

### Official_Comment — Reviewer_1Z1E

- Note ID: `5i31QZfU7K`
- Discussion number: `25`
- Invitation: `ICLR.cc/2025/Conference/Submission7925/-/Official_Comment`


#### Comment

> implementing the method since paper cannot cover

Although the code is not public, the evaluation datasets and code is. Any comparison on the same benchmark should be enough to confirm the gains.

> We briefly talked about using averaging over summation in our paper (L231) to prevent putting higher scores on longer documents.

I understand the need of averaging instead of summation. What I wanted to express is that, the cited work reorders the input several times and computes the score each time for the same document to account for the position bias. This is a simple way to avoid it. I wanted to understand if such a method is enough. The proposed method and this "naive" method, both require more computation than vanilla attention. If we match the number of operations, do you think your method is "better"?  Just wanted a comment on that.

> We want to point out a slight mistake:

My apologies, that was a poorly thought out comment.

### Official_Comment — Authors — Additon experiments on Peysakhovich & Lerer (2023)

- Note ID: `tXFO7UGI45`
- Discussion number: `26`
- Invitation: `ICLR.cc/2025/Conference/Submission7925/-/Official_Comment`


#### Title

Additon experiments on Peysakhovich & Lerer (2023)

#### Comment

We implement the method ourselves to show the difference between PINE and Peysakhovich & Lerer (2023), and find PINE still get a better result on RewardBench.

On LLama 3 8B Instruct:

| Method |  Accuracy |
| - | - |
| Vanilla (Shuffle) | 64.8|
| Peysakhovich & Lerer (2023) [k=1] | 65.2 |
| PINE | **66.7** |

We believe our additional experiments can help you better understand the differences between the two methods.

Since the rebuttal is going to end, please let us know if you have any additional questions and we hope our replies address your concerns. If you find our responses helpful, we would greatly appreciate it if you could consider raising your scores.

### Official_Comment — Authors

- Note ID: `I0WoZ7VeVN`
- Discussion number: `27`
- Invitation: `ICLR.cc/2025/Conference/Submission7925/-/Official_Comment`


#### Comment

Thanks for your additional valuable feedback!

> Implementation of Hsieh et al.

Although the evaluation code is publicly available, we tried and could not reproduce the paper-reported number perfectly on the vanilla inference model, which is probably due to the slight difference in the prompts, etc.

Therefore, we implemented the method ourselves and ran it on the same prompts we used in our experiments (although we admit we can not guarantee the reproduction is perfectly correct since no public codes and some missing details in the paper). Our results show that our approach performs better than Hsieh's, probably because the strong assumption in Hsieh's does not hold well in every case.

On Llama 3 8B Instruct and RewardBench:

| Method |  Accuracy |
| - | - |
| Vanilla (Shuffle) | 64.8|
| Peysakhovich & Lerer (2023) [k=1] | 65.2 |
| Hsieh et al. | 65.0 |
| PINE | **66.7** |

We hope this result can address your concerns.

> If we match the number of operations, do you think your method is "better"?

The method you mention is Peysakhovich & Lerer (2023). We implemented the method by ourselves (since the code is not publicly available) and reported the number in the above table. We find the method cannot beat PINE.

However, we have to admit that the upper bound of such prompt order optimization is higher than PINE since the best case is that the ground-truth documents always appear at the correct position. We believe PINE is still a better choice until such optimization method is discovered, as the above results suggest.

We hope our additional experiments can answer your questions!

### Official_Comment — Authors

- Note ID: `zznUAw4RvS`
- Discussion number: `28`
- Invitation: `ICLR.cc/2025/Conference/Submission7925/-/Official_Comment`


#### Comment

Thank you for engaging with us and giving insightful feedback!

Since the rebuttal is going to end, please let us know if you have any additional questions and we hope our replies address your concerns. 

If you find our responses helpful, we would greatly appreciate it if you could consider raising your scores.

### Official_Comment — Reviewer_azrE

- Note ID: `TFXZCultFB`
- Discussion number: `29`
- Invitation: `ICLR.cc/2025/Conference/Submission7925/-/Official_Comment`


#### Comment

thanks for the additional experiments. i am raising my score

### Official_Comment — Authors

- Note ID: `mTHaQnCRfk`
- Discussion number: `30`
- Invitation: `ICLR.cc/2025/Conference/Submission7925/-/Official_Comment`


#### Comment

Thank you for providing the helpful suggestions! We will incorporate your feedback into our next revision to make the paper clearer and more convincing.

### Official_Comment — Reviewer_Qpxn

- Note ID: `P4Jicr37SI`
- Discussion number: `31`
- Invitation: `ICLR.cc/2025/Conference/Submission7925/-/Official_Comment`


#### Comment

Thank you for your response. I am now more confident that your work improves over the state of the art. As a result, I have raised my score. Good luck!

### Official_Comment — Authors

- Note ID: `YR9aWiRHDX`
- Discussion number: `32`
- Invitation: `ICLR.cc/2025/Conference/Submission7925/-/Official_Comment`


#### Comment

Thank you for your valuable feedback, and engage with us in the rebuttal phase!

We will incorporate your advice into our next revision to make the paper more convincing.

## Decision

### Decision — Program_Chairs — Paper Decision

- Note ID: `2fsgpZvYOQ`
- Discussion number: `1`
- Invitation: `ICLR.cc/2025/Conference/Submission7925/-/Decision`


#### Title

Paper Decision

#### Decision

Accept (Poster)


---

# A Formal Framework for Understanding Length Generalization in Transformers — OpenReview 审稿全文归档

- Venue: **ICLR 2025 Poster**
- OpenReview forum: [https://openreview.net/forum?id=U49N5V51rU](https://openreview.net/forum?id=U49N5V51rU)
- Official paper page: [https://proceedings.iclr.cc/paper_files/paper/2025/hash/928170bcb050fe64a63fe781b82265aa-Abstract-Conference.html](https://proceedings.iclr.cc/paper_files/paper/2025/hash/928170bcb050fe64a63fe781b82265aa-Abstract-Conference.html)
- Reviewer handles are the public OpenReview pseudonyms; no attempt is made to identify individuals.
- Source: public OpenReview review dump; fields are preserved as released, including review text, rebuttal comments, meta-review, and decision where available.

## Paper Abstract

A major challenge for transformers is generalizing to sequences longer than those observed during training. While previous works have empirically shown that transformers can either succeed or fail at length generalization depending on the task, theoretical understanding of this phenomenon remains limited. In this work, we introduce a rigorous theoretical framework to analyze length generalization in causal transformers with learnable absolute positional encodings. In particular, we characterize those functions that are identifiable in the limit from sufficiently long inputs with absolute positional encodings under an idealized inference scheme using a norm-based regularizer. This enables us to prove the possibility of length generalization for a rich family of problems. We experimentally validate the theory as a predictor of success and failure of length generalization across a range of algorithmic and formal language tasks. Our theory not only explains a broad set of empirical observations but also opens the way to provably predicting length generalization capabilities in transformers.

## Review Inventory (5 Official Reviews, 23 Discussion/Comment Notes)

- Final decision: **Accept (Poster)**
- Official review ratings (review order): `6, 8, 6, 8, 6`

## Official Reviews

### Official_Review — Reviewer_H3pw

- Note ID: `PLKezvI7D2`
- Discussion number: `1`
- Invitation: `ICLR.cc/2025/Conference/Submission13115/-/Official_Review`


#### Summary

In the recent years, with the surge of LLMs and transformers, there has been an increasing interest to formally understand the generalization failures of transformer. A notorious and common failure mode that is often mentioned in the literature pertains to length generalization, where the model is trained on sequences of certain length, and then is tested to longer sequences. Many works have shown the failures of modern architectures on a range of tasks (most popular ones being parity) and others have come up with strategies to fix the failures (e.g., changes in positional embeddings). In recent work, Zhou et al., authors proposed an insightful conjecture, the RASP-L conjecture, that aims to dilineates the tasks that are easy for transformers to learn and length generalize on, and then tasks that continue to be hard for transformers. While the conjecture was empirically backed, theoretical justifications for the same have been lacking. In this work, the authors aim to formalize and provide a theoretical justification to the RASP-L conjecture. 

The authors study two types of positional encodings -- no positional encodings, and absolute positional encodings. When dealing with absolute positional encodings the number of parameters grow with length of the input. To address this challenge, the authors introduce a new object, the limit transformer. This transformer encapsulates the behavior of transformers on longer and longer sequences into a single object. The authors define an idealized inference procedure, which searches for the transformer that minimizes the risk along with a regularization constraint. The regularization constraint is a special one, i.e., it is not the standard constraint based on purely say the l2 norm of the weights. With these constraints in place, the authors show that if the ideal function f is expressable in a limit transformer that satisfies two properties -- Local and Periodic, then the transformer is able to learn the function f and length generalize on it. In the second half of the paper, the authors show that for every program C-RASP[phi, Psi) with local and periodic phi and Psi respectively, there exists a limit transformer that accepts the same set of strings that P accepts. The authors also provide communication complexity based arguments to explain the limits of limit transformers. Finally, the authors conduct experiments to match the predictions of the theory.

#### Soundness

`2`

#### Presentation

`3`

#### Contribution

`2`

#### Strengths

1. The authors tackle a hard problem, i.e., providing a formal justification of RASP conjecture for multi-block transformer models. This is both a hard problem and an important one. 
2. The authors have been creative in several aspects of the paper -- i) the regularization constraints that have been imposed seem particularly important to the inference procedure's success, ii) the construction of the limit transformer, iii) the tight connection between the limit transformer and the C-RASP language from Yang and Chiang.
3. The main body of the paper is nicely written and does a nice job of getting to the main results quite fast.

#### Weaknesses

There are quite a few concerns that I have for various parts of the paper. My current score is a reflection of these weaknesses. I would be happy to change my score if the authors can provide satisfactory explanations and no other major concerns appear in the course of discussion. 

1. **No Positional Encoding vs. Absolute Positional Encoding**:  In the current work, the authors went through great detail to construct limit transformer with the idea that limit transformer can encapsulate longer and longer transformers into one limiting object. The first thing that bothers me is that if we take no positional encodings then we do not need this object. The authors do not explain the need of limiting transformer in the context of NoPE as there is no growth in number of parameters anymore that limit transformer needs to cater to. 
The second thing that bothers me is that from an expressivity point of view, the NoPE based transformer can express APE up to a large length. So why can't we use
Theorem 1  (https://proceedings.neurips.cc/paper_files/paper/2023/file/4e85362c02172c0c6567ce593122d31c-Paper-Conference.pdf) from this work. This work essentially is arguing that NoPE can approximate APE (or RoPE). 

2. **About invariance to offsets and regularization** 
     a) The authors introduce the constraint that the transformer should be invariant to offsets, i.e., if the problem appears at different locations in the context window then the solution should not change. This constraint is not explicitly enforced.  
     b)  The authors also have an idealized inference procedure, where the regularizer has been introduced for the purpose of theory. The authors argue that there is an implicit bias towards small values of it at initialization. I don't quite see how. Also, this regularizer is also not explicitly enforced. 
  Since there is quite some gap between the theory and expmts, what do you think is the explanation for this gap? 

3. **Regarding the definition of limit transformers**
     a) The limit transformer is introduced in Definition 2. The term y_i(l) in equation (6), is it the same as how it was defined in equation 4. If so, then does it already have positional encoding in it like in equation (1). If so, then how does the function phi that is introduced additionally absorb the terms that involve inner product of two different positional encodings. It feels if y_i^(0) already had positional encodings in it then won't the first term in equation 6 already take care of the stuff. 

   b) Is the rest of the construction of limit transformer same as standard transformer and the only difference is equation (6)? I ask this because following equation (6), we are not told what happens to attention logits. 

   c) In point 2 in the definition u say that the positional encodings p_i are encoded in finite precision. If that is the case, then when we increase the length to arbitrary large values, the positional encodings start overlapping and we only have finitely many positional encodings.    This does not address the increase in the parameter count issue that the author state was the very reason to define the limit transformer. If we are happy with finitely many positional encodings, then why not just do some periodic encodings in the standard transformer? Also, if we are happy to do everything with finitely many positonal encodings then this goes back to my first concern on NoPE, we can operate with NoPE, express finitely many positional encodings, and simplify the whole story right? 

 
4. **Concerning periodic and local in definition 3.** In Definition 3, you state that phi_l,h is translation-invariant and local. You also stated that phi_l,h expresses the inner product involving positional encodings. This translates into a constraint on the positional encodings. This creates confusions. The authors should be more clear on this whole connection in the main body. 

5. **About the hypothesis class definition 4** In definition 4, you say that each product function involving position encoding is translation invariant. You also say that each function involving exactly one product function is translation invariant. These constraints seem very restrictive. Is there a reason to believe that imposition of these constraints is not over simplifying the problem somehow? Are these constraints implying offset invariant condition? I think offset invariance on its own was reasonable but this seems not very digestable. I would appreciate if authors gave more insights into why these constraints are not unreasonable? Also, some numerical insights into what it means to enforce these constraints? This goes back to my point 2. If you see point 2, I state that there is gap between theory and expmts. In this case, the theory would require some of these constraints, which how do u really enforce? Perhaps these constraints are strong sufficient conditions for length generalization? and far from necessary? Is offset invariance a necessary condition btw for length generalization?

6. **On the regularizer in definition 5** 
    a) After definition 5, you state that the idea of this regularizer is to discourage attention between far-away positions that do not appear together during training, which could hamper length generalization. At what point do you use this insight in the proofs. For instance in Lemma  17 how does it come up? I don't quite see it. Also, since all positions are equally penalized in this regularizer, why would things far off be more penalized? Also, the justification based on initialization making the regularizer small is not fully clear. 

   b) If we use NoPE positional encoding, then p_i is set to zero for all i. As a result, I don't quite understand the role of the regularizer anymore. Since the regularizer is supposed to penalize far away positions, those positions don't seem distinguisable under the regularizer as p_i is set to zero. Further, if p_i is set to zero, then what is the role of the phi function in equation (6), and eventually why do we need the limit transformer? 

7. **On phi function** In the line 260, you indicate that for phi function, we need to only care abt it its values such as phi(1,1), phi(1,2),..phi(1,tau). The values above the diagonal are taken care of by translation invariance, but what abt the values below the diagonal, i.e., phi(2,1)..I don't think you assume symmetry, do you? 

8. **Minor remark on line 326/327** Shouldn't the Q_a in the RHS be Q_a(j) and not Q_a(i)?

9. **Notation remark on Theorem 9**. In an unfortunate use of notation, u call Psi function local and Phi function periodic. Earlier u had used phi for local in the definition 3. 

10. Since the results from the work hold for NoPE positional embeddings. From Theorem 2 in https://proceedings.neurips.cc/paper_files/paper/2023/file/4e85362c02172c0c6567ce593122d31c-Paper-Conference.pdf, the results should extend to relative positional encodings too?

11. Currently the results require a large N_0, which practically speaking can be very large. The authors do mention this limitation. While I am not expecting a bound of any sort in this work, I want to understand the consequences of the results better. The current machinery in this work (if correct), seems to indicate that allowing for a very large N_0 and some strong constraints (periodic, local on hypothesis class), length generalization is achievable for a large N_0 seen during training. If this N_0 from the theory is quite large, then would you say that research in this should try to explain why can transformers do it with a much smaller N_0 than theory predicts? I want to understand the hunch of the authors here. If one tries to bound N_0, then would we run the risk of vaccuous bounds?


12. **Concerns on the proof of Lemma 17**:  
    a) In line 976 you say there are only a finitely many settings traversed by the limit transformer \tilde{T}_i. Why is this the case? Can you properly justify?

      b) In equation 9, you say that we select an R(Tn) that is less than 1/n + inf (R(T)). In line 981, you argue that R(Tn) should converge because inf R(Tn) is bounded and monotonically increasing. This argument only tells that the RHS in equation (9) converges, which is the upper bound. Why does it imply that the LHS converges? You crucially use the existence of this limit in equation (14).

     c) In line 995, you say that lim D_v_i(v_i) is D_0. Why does this limit exist? 

     d) Below equation 17, you state "As this function is monotonically increasing, and as phi_{l,h} has bounded precision, there must be t_infty..." Why does this hold true? I don't quite follow. 

      e) In line 1010-1017, you construct a sequence of T'n satisfying certain properties. Why does this have to exist? For instance why is  D_{vi(n)}(n) = D_{infty}(n) true?

      f) The phi_l,h(i,j) in equation (10) should also bear the index n as it would be different for each limit transformer. This makes the rest of the stuff bit confusing.  For instance, you define D_n(tau) in line 984. Why would it be the case that tau can be larger than n in the summation? Since positional embeddings for that transformer would only be defined up to n right? 

      g) I do not follow the inequality in line 1044?  

      h) In line 1049-1052, you say that set of functions traversed becomes stationary, what do u precisely mean here? 

13. In line 1121 to 1127, how does the C logN fall out. Do you mean that since you partition stuff into N positions and that takes log N bits to compute, we get C log N?


14. I realize that there was one question I had forgotten to add in the above list. In the Definition 4 of your hypothesis class, I do not see any constraint on positional encodings being periodic. Does periodicity fall out as a consequence of the offset independent constraint you add later in the definition? What confuses me quite a bit is that while your transformer is not periodic but your limit transformer is periodic. How is it that a transformer with non-periodic positional encodings with an ever growing context window is captured by another transformer like object with periodic positional encodings? And relatedly why not just have periodic positional encodings on the original transformer to begin with instead of the very strong offset independence conditions in definition 4.

#### Questions

Please see weakness section, where I list both the weaknesses and questions.

#### Flag For Ethics Review

- No ethics review needed.

#### Rating

`6`

#### Confidence

`3`

#### Code Of Conduct

Yes

### Official_Review — Reviewer_odFA

- Note ID: `8x83XrtdWm`
- Discussion number: `2`
- Invitation: `ICLR.cc/2025/Conference/Submission13115/-/Official_Review`


#### Summary

**Update after rebuttal:**
The authors have clarified my questions and made improvements to the presentation of the main manuscript (though it remains a very long paper of course, with some important parts in the appendix, hence I will leave my 'Presentation' rating on 2). I am now more confident that the work is important and is ready to be presented and discussed with the wider ML community. I would now raise my score to a 7, but ICLR does not allow this score this year. I do think that some weaknesses remain, and that the paper does not quite hit an 8. But for the sake of expressing a clear opinion for the decision-making process, I am raising to an 8 (as the only available option), but will keep my confidence low to indirectly indicate that the score is a bit inflated.

---

This paper formally defines a function class, implicitly via the construction of the Limit Transformer, and shows that length generalization with transformers (of arbitrary context length) is provably guaranteed within this function class (this does not include SGD training though). The main part of the proof is the use of an “inference procedure” (Definition 6), which, informally, iterates through all transformers of increasing context size and eliminates the ones that do not fit the data. By construction, the set of functions that these transformers can implement (with increasing long context) is finite, such that eventually, at some finite context size, only the transformer that generalizes correctly to arbitrary length remains (all others have been ruled out by the data at this point; to be precise a complexity regularizer is also required to make the choice unique). This is the main argument in Theorem 7.2 (part of the main result). Keeping the parameters of transformers with increasing context width finite is central to the whole construction, and is reflected in the notions of PERIODIC and LOCAL of the Limit Transformer (informally: the functions implemented via attention only operate on a finite/small context window, and are translation invariant). Finally, the paper shows that a version of C-RASP (with either learned absolute positional encodings or no positional encodings) can be proven to be expressible via Limit Transformers, leading to the conjecture that length generalization with practical transformers is strongly related to whether a solution can be expressed in C-RASP or not. A small set of relevant experiments supports this conjecture well, including a (potential) explanation why some relatively simple regular languages cause trouble with length generalization (for which no C-RASP implementation provably exists).

#### Soundness

`3`

#### Presentation

`2`

#### Contribution

`3`

#### Strengths

* Very timely question. While frontier models show many surprising and unprecedented capabilities, they fail catastrophically on some really simple problems in length generalization. Understanding the underlying reasons is crucial for Safety and Reliability, and may also pave the way to address these issues in future-generation architectures.
* The expressivity result stating that all C-RASP programs can be expressed by a Limit Transformer, and are thus identifiable in theory via the inference process in Def. 6, is a strong result that bridges the theory to very concrete and empirically testable hypotheses.
* Empirical results show that despite a very elaborate construction of a limit process and complex regularizer, the theoretical predictions have actual practical consequences on standard transformers trained in a standard fashion (that differs significantly from the inference process in Def. 6)

#### Weaknesses

* The paper is very extensive (72 pages, 62 are appendix, with 3887 lines in total; given the conference timelines and workload I could not review the appendix). The main paper is thus more a summary of the appendix than a standalone paper. While it makes no sense to have the main results and theorems without the proofs, maybe splitting the publication into a journal- / long-format theory paper and a separate more extensive empirical verification for a mainstream ML conference could be better.
* The construction of the Limit Transformer is quite elaborate (or rather the construction to go to the infinite context limit with non-exploding parameter sets / maintaining a finite function class) and the inference procedure in Def. 6 is completely impractical. Without any empirical results I would have been very skeptical about the practical relevance, and to be fully convinced I would like to see further results (though I believe Fig. 1 is significant and very promising). It is a bit unclear whether the theory had to be this complex and the connection to C-RASP dropped out as a lucky coincidence (which is how the paper is currently written), or whether putting C-RASP on a theoretical footing was the original goal that demanded this level of complexity.
* The inference procedure in Def. 6 seems a bit crude (though it does the job theoretically; same applies to the regularizer which is composed of 8 different complexity terms). The standard approach (e.g., in algorithmic information theory / Solomonoff Induction) is to not try to identify the correct function at some point $N_0$ but bound the overall number of mistakes (which is finite for any finite-length program). $N_0$ would be the point where the “last” mistake happens, which is generally unknown, and usually having bounds and generalization guarantees in terms of numbers of (remaining) mistakes are much tighter and shrink much faster with increasing number of observations. This is probably a question for future work, but it is unclear whether the restriction of using Def. 6 (having to identify the correct function) is implicitly limiting the function class where generalization is possible, and whether this class could be extended by focusing on a theoretical scheme that relies on bounding the mistakes (number of mistakes, and/or their cumulative magnitude).

**Verdict:**
Overall I am a bit ambivalent about the paper. On one hand it tackles a very timely and important problem by starting to make good progress from a theoretical angle (rather than adding even more contradicting experiments). On the other hand the current theory and presentation are quite complex and extensive. If the theory had been published in a journal / long-format paper before, and this paper would solely focus on empirically testing the conjecture that C-RASP expressiveness predicts length generalization, then a 10-page conference format seems like a great fit. Similarly, without the empirical results, I would have strongly doubted the practical relevance of the theory. But, the empirical results in Fig. 1 look very promising; though at this point it is unclear whether any results that contradict the main claims can easily be found or not. I do believe that the ML community needs more exposure to good theory, particularly at conferences, though I am not sure that this paper is the best example (due to its excessive length). I am also quite certain that this paper will spark quite a bit of follow-up work to make the theory simpler and/or more complete (expand the function class), and that publishing it will stir the community to conduct more empirical tests of the theory (which will make overall faster progress than asking the authors to perform more experiments). I am therefore currently slightly in favor of accepting the paper, though I would not be upset if others argue that ICLR may be the wrong venue. My confidence is currently on the low end - I did not have time to go through the extensive appendix, and there are a few bits and pieces of the main paper where I am not fully sure how they work out / will be proven. I am very happy to reconsider my opinion based on the other reviews and authors’ responses, and to make my criticism concrete, I leave some suggestions for improvement in the Questions section below.

#### Questions

**Improvements:**
(I consider all of them optional suggestions, not strict requirements)
 1. Maybe give the reader a better sense of where this is going early in the paper. It should be clear early on that the paper constructs a function class for which generalization can be proven in the theoretical limit, but the theory does not answer how this relates to training actual transformers of fixed context length via SGD (i.e., there may be functions that can theoretically be proven to be length-generalizable, but this may not work in practice). Also state that this function class is likely not complete (i.e., there may be functions where transformers can length-generalize that lie outside this function class).
2. I really liked lines 255-264 in terms of clarifying the paper. Maybe the same information can be qualitatively given early in the paper to prime the reader.
3. Definition 2 can be a bit misleading - it informally may suggest that a Limit Transformer is basically “just a normal transformer with infinite context”. This needs to be clarified. The text already mentions that the Limit Transformer is a *theoretical* construct (maybe consider calling it a Limit Transformer Process or similar, to make sure that the object is not confused with a concrete architecture). The finite precision argument in Def. 2 is theoretically ok, but in practice it just says that the precision is an arbitrarily high natural number (with no upper bound given), which is not implementable “like a standard transformer”. Also state here or earlier in the paper that the Limit Transformer cannot be trained via SGD, but uses a theoretical procedure (Def. 6) that cannot be practically implemented. And finally, the functions $\phi_{l,h}$, whose complexity is not bounded as far as I can tell, do a lot of the heavy lifting and are not just marginal additions to a standard transformer.

**Minor questions**

1. L200-202: This is a very interesting requirement. Together with the requirement for periodicity and locality I am reminded of the pumping lemma for regular languages. But if I understand correctly (some experiments, Fig. 10, and discussion following L 486), C-RASP defines a subset of $TC^0$ and it covers some but not all regular languages and some simple non-regular languages.
2. Showing that C-RASP with a small extension can provably be expressed by a Limit Transformer is very interesting. Can C-RASP potentially still be extended without violating this equivalence, or is the current set of operations (likely) complete?
3. What is the relation of Theorem 7 to standard learnability / language identifiability results (language identifiability is generally not possible from positive examples only)?

#### Flag For Ethics Review

- No ethics review needed.

#### Rating

`8`

#### Confidence

`2`

#### Code Of Conduct

Yes

### Official_Review — Reviewer_6ZTg

- Note ID: `P3W3FtWGm4`
- Discussion number: `3`
- Invitation: `ICLR.cc/2025/Conference/Submission13115/-/Official_Review`


#### Summary

This paper develops a theoretical framework for transformer length generalization, introducing "Limit Transformers" and proving generalization guarantees for functions satisfying PERIODIC and LOCAL properties. They characterize functions that are identifiable with APE and NOPE, and validate the prediction from theory with experiments on algorithmic and formal language tasks.

#### Soundness

`3`

#### Presentation

`3`

#### Contribution

`2`

#### Strengths

1. The paper provides theoretical analysis of length generalization in transformers, including both sufficient conditions for generalization and communication complexity bounds showing certain functions cannot exhibit length generalization. This addresses an important open problem in theory. Also, it is the first to include length generalization theories with positional encodings to my best knowledge.
2. The theoretical framework successfully predicts empirical length generalization behavior across diverse tasks. The C-RASP formalism provides an interpretable way to determine whether a function should exhibit length generalization, making the theory practically useful.
3. The experimental evaluation is thorough with both algorithmic tasks and formal languages.

#### Weaknesses

1. The main contribution compared to the previous C-RASP work and the RASP-L work is to extend the framwork to include positional encodings. However, the positional encodings considered are APE and NOPE, while the more frequently used encodings in practice are relative positional encodings. Also see question 1.
2. The limitation of the proposed theoretical framework is discussed, but it could benefit from more empirical evidence, like where the theory predicts length generalization but empirical performance is poor, or vice versa.

#### Questions

1. Have you tried the performance of RPE although the current theory does not cover it yet? In [1], NOPE is shown to inherently learn some kind of relative positional encoding. Does RPE behave similarly to NOPE in certain tasks? 
2. The model size provided in the appendix is highly dependent on the specific tasks (different tasks use different model depths, embedding space sizes, etc). Do you observe the sensitivity of the generalization performance in terms of model sizes? Or are there other reasons for this?

[1] The Impact of Positional Encoding on Length Generalization in Transformers
Amirhossein Kazemnejad, Inkit Padhi, Karthikeyan Natesan Ramamurthy, Payel Das, Siva Reddy

#### Flag For Ethics Review

- No ethics review needed.

#### Rating

`6`

#### Confidence

`3`

#### Code Of Conduct

Yes

### Official_Review — Reviewer_Vn9i

- Note ID: `oq3Jzp08SY`
- Discussion number: `4`
- Invitation: `ICLR.cc/2025/Conference/Submission13115/-/Official_Review`


#### Summary

The paper studies the length generalization problem in decoder-only Transformers, which refers to the inability of models to deal with longer samples than encountered during the training phase. The paper aims to identify which tasks can achieve length generalization. To this end, the authors introduce Limit Transformer, a theoretical model designed to generalize across varying sequence lengths. Importantly, the authors prove that under an idealized inference procedure, any tasks that can be expressed by Limit Transformer, satisfying Periodic and Local constraints, can provably achieve length generalization. Furthermore, the authors show that any C-RASP programs can be expressed by Limit Transformer, leading to the conclusion that such tasks can also generalize to longer inputs. On the other hand, the paper presents that copying and addition cannot be generalized as these tasks do not exhibit logarithmic communication complexity. Finally, the paper provides experimental results, demonstrating that tasks expressible by C-RASP programs (binary majority, sort, copy unique) easily length-generalize while tasks outside C-RASP (copy repeat, parity, addition) fail to achieve length generalization.

#### Soundness

`4`

#### Presentation

`3`

#### Contribution

`4`

#### Strengths

- The paper addresses an important question in the literature: which tasks can achieve length generalize and which cannot. Prior work on length generalization has mainly focused on improving empirical performance, with fewer studies providing a theoretical understanding. While [1] introduces RASP-L to identify tasks that can achieve length generalize, their argument still remains conjectural. This study goes one step further than the previous RASP-L conjecture, offering a theoretical framework to determine whether certain tasks will provably achieve length generalization or not. Therefore, I believe this paper makes a significant contribution to the literature.
- The FAQ section in the appendix effectively conveys the paper's intuition and enhances the reader’s understanding.
- Overall, the paper is well-structured and provides a detailed analysis to present their findings.

[1] Zhou, Hattie, et al. "What algorithms can transformers learn? a study in length generalization."

#### Weaknesses

I don’t see any significant weaknesses in the paper, but there are a few minor limitations, which I don’t consider critical issues in evaluating this paper.

- The scope of the paper is limited to algorithmic tasks and does not cover length generalization problems in natural language processing tasks.
- As explained in the paper, a key assumption in the framework is the idealized inference procedure, which assumes that we can obtain Transformers that are fitted to reproduce a target function while minimizing a specific regularizer $R(T)$, and thus the current framework is not fully end-to-end. Introducing an analysis of training dynamics to replace this assumption would be a promising direction, achieving a truly end-to-end, complete argument.

#### Questions

- In Figure 1, why do Transformers with NoPE fail even on in-distribution samples (Bin 1) for Copy Unique and Addition? Doesn’t this contradict the observations in [2], which argue that NoPE can length-generalize for these tasks to some extent?

[2] Kazemnejad, Amirhossein, et al. "The impact of positional encoding on length generalization in transformers."

#### Flag For Ethics Review

- No ethics review needed.

#### Rating

`8`

#### Confidence

`3`

#### Code Of Conduct

Yes

### Official_Review — Reviewer_Gpiw

- Note ID: `1ehrSNl582`
- Discussion number: `5`
- Invitation: `ICLR.cc/2025/Conference/Submission13115/-/Official_Review`


#### Summary

This work studies the challenge of enabling transformers to generalize to sequences longer than those seen during training. This capability is often inconsistent across tasks and lacking strong theoretical understanding. The authors present a new theoretical framework to examine length generalization in causal-attention transformers with learnable positional encodings. Using this framework, they analyze the conditions under which transformers can generalize to longer sequences under certain conditions. The theoretical findings are further validated through empirical tests on a set of tasks.

#### Soundness

`3`

#### Presentation

`2`

#### Contribution

`2`

#### Strengths

* Length generalization is a very important problem that is being explored from various perspectives, though primarily through algorithmic and empirical studies, with relatively few theoretical analyses. This work takes a different approach, offering distinct advantages.
* It's encouraging to see that, under certain conditions (as shown in Theorem 7), they identify cases where length generalization is achievable.
* Provides several formalizations (e.g. Limit Transformer) and proofs on this topic that would be valuable for future research.

#### Weaknesses

* The paper’s current writing seems to focus more on demonstrating how to achieve length generalization under specific conditions than on examining the limitations of existing models. For example, it centers on proving Theorem 7 by introducing various modeling assumptions, such as the Limit Transformer and specific inference conditions (like the proposed regularizer). This approach differs somewhat from the expectations set in the introduction, which suggests a more interpretive analysis of current models.
* The practical impact of this paper feels somewhat limited. It would be valuable if the authors could identify practical applications that leverage their findings.
* C-RASP and formal languages are quite distinct from traditional text corpora or other common scenarios. Even though the aim is to theorize the formalism in this field, there seems to be a significant gap between the formalism presented in this paper and that of other widely used settings.

#### Questions

* Could the authors more directly relate their theorem to widely used transformer models? For example, which specific conditions do not hold in these models, and how does this align with the empirical observations in Section 5?
* The FAQ in Appendix A actually was quite useful providing valuable insights and gives readers motivation. Could the authors consider integrating some of these discussions more naturally into the introduction or relevant sections?

#### Flag For Ethics Review

- No ethics review needed.

#### Rating

`6`

#### Confidence

`2`

#### Code Of Conduct

Yes

## Meta-Review

### Meta_Review — Area_Chair_2iVk

- Note ID: `FYK2JNvCmK`
- Discussion number: `1`
- Invitation: `ICLR.cc/2025/Conference/Submission13115/-/Meta_Review`


#### Metareview

Summary:
The paper introduces a theoretical framework analyzing length generalization in causal transformers with learnable absolute positional encodings. The authors introduce "Limit Transformer" to handle growing parameters and show that C-RASP programs (under specified periodic and local constraints) can be translated into Limit Transformers, providing a concrete class of problems where length generalization is guaranteed. The theoretical findings are supported by empirical experiments demonstrating that tasks expressible by C-RASP programs succeed in length generalization, while others, such as copying with repetitions and n-digit addition, fail.

Strengths:
- Addresses an important open problem in transformer theory - length generalization capabilities
- Provides rigorous theoretical analysis with concrete proofs and guarantees
- Formalizes and proves aspects of the previously conjectural RASP-L hypothesis
- Strong empirical validation across diverse tasks showing alignment between theory and practice

Weakness:
- Treatment is limited to absolute positional encodings rather than more popular schemes
- The theoretical setup makes some idealized assumptions that do not reflect practice
- Focus is primarily on algorithmic tasks rather than natural language

Decision:
All the reviewers were in consensus that the paper represents an important step forward in understanding transformer capabilities and limitations, providing both theoretical insights and practical implications for future work. It merits publication at ICLR 2025.

#### Additional Comments On Reviewer Discussion

We thank the authors and reviewers for engaging during the discussion phase towards improving the paper. Below are some of the highlights:

1. NoPE vs APE encodings contradiction:
- Reviewers questioned why NoPE couldn't subsume APE based on prior work and thus alleviate the need for "Limit Transformers"
- Authors clarified with new experiments showing APE's superior performance
- Added Appendix G.7 showing concrete advantages of APE over NoPE
- Convincingly resolved through theoretical and empirical evidence

2. Translation invariance assumptions:
- Concerns about restrictiveness of translation invariance requirement
- Authors added Appendix G.6 showing it emerges naturally in practice
- Demonstrated both theoretical benefits and empirical validation
- Adequately justified the assumption

3. Paper complexity and presentation:
- Reviewers noted elaborate construction and long technical details
- Authors improved introduction and added clarifying content
- While still complex, changes made paper more approachable
- Justified length as necessary for complete treatment

4. Practical impact:
- Questions about real-world applicability
- Authors clarified theoretical foundations enabling future practical work
- Added discussion of potential applications
- Acknowledged limitations while highlighting value for future work

## Rebuttal and Discussion Comments

### Official_Comment — Authors — Response

- Note ID: `88f9Q2zm5m`
- Discussion number: `1`
- Invitation: `ICLR.cc/2025/Conference/Submission13115/-/Official_Comment`


#### Title

Response

#### Comment

Dear reviewer Gpiw,

Thanks for your feedback on our paper!

### Reply Regarding Weaknesses

> (1) The paper does not include interpretive analysis of existing models

We would like to clarify that our paper focuses on a theoretical framework establishing *general criteria* for when length generalization is achievable when training transformers on some problem, and leave interpretive analysis of existing *specific models* out of scope. We have rephrased the introduction to make explicit that we are interested in the setting where transformers are specifically trained on short inputs from some task. 


> (2) What is the practical impact of this paper?

We believe that a theoretical understanding explaining empirical findings (Zhou et al. 2024) can improve current and future applications in NLP. A possible practical consequence is deriving new positional encoding or scratchpad schemes using our theoretical insights to enable stronger length-generalization. For instance, RASP and RASP-L have been used to derive practical advances in the design of transformers for various problems, e.g. [1,2]. Additionally, our framework may be used for more refined methods of interpretability due to the formal connection between our RASP variant and length-generalizing transformers - for instance, decompilation of transformer weights into RASP programs [4]. Hence, we believe our theoretical framework is highly valuable for practical work to build upon.

> (3) C-RASP and formal languages are quite distinct from common scenarios

As described in our response to (2), RASP and RASP-L have had substantial impact in research on transformers. Formal languages are a principled way to model sequences of unbounded length - crucial for understanding length generalization. They provide a formal way to analyze the capabilities of transformers to perform practically relevant tasks, such as induction heads (which we have analyzed in this paper). In fact, formal languages have also been used to validate new positional encoding schemes [3]. 

[1] Fan et al, Looped Transformers for Length Generalization, 2024. https://arxiv.org/abs/2409.15647

[2] Hou et al, 
Universal Length Generalization with Turing Programs, 2024. https://arxiv.org/abs/2407.03310

[3] Ruoss et al, Randomized Positional Encodings Boost Length Generalization of Transformers, 2023. https://arxiv.org/pdf/2305.16843

[4] Friedman et al, Learning Transformer Programs, 2023. https://arxiv.org/abs/2306.01128

### Reply Regarding Questions

> (1) How does the theory relate to widely used transformer models?

As we describe above, our study focuses on general properties of the transformer architecture. We empirically test the predictions by training models on various problems from scratch in Section 5. We expect that the results also have implications for the algorithmic abilities of commonly used specific models (such as GPT-3 or LLaMa-3), but more detailed exploration of this to future work.

> (2) Can more content from the FAQ be included in the main text?
 
We are glad to hear that the FAQ is useful. We have made some changes to the main text to make the relevant information (e.g., about the role of limit transformers) more salient.

### Official_Comment — Authors — Response

- Note ID: `Hk0FkVjBy3`
- Discussion number: `2`
- Invitation: `ICLR.cc/2025/Conference/Submission13115/-/Official_Comment`


#### Title

Response

#### Comment

> I don’t see any significant weaknesses in the paper, but there are a few minor limitations, which I don’t consider critical issues in evaluating this paper.

We thank the reviewer for the positive assessment.

> (1) The scope of the paper is limited to algorithmic tasks and does not cover length generalization problems in natural language processing tasks.

We agree with this, and now make this explicit throughout Section 1. We note that prior empirical work on length generalization (whose results our work aims to give a theoretical foundation to) has largely focused on algorithmic tasks. Expanding to naturalistic language comprehension tasks is an interesting future direction. 


> (2)  Analysis of training dynamics, instead of an idealized inference procedure, would be desirable.


We agree with this limitation, which we acknowledge in Section 6.

### Questions:

> (1) In Figure 1, why do Transformers with NoPE fail even on in-distribution samples (Bin 1) for Copy Unique and Addition? Doesn’t this contradict the observations in [2], which argue that NoPE can length-generalize for these tasks to some extent?


In Copy Unique and Addition (Figure 1), the NoPE results are represented by the red dotted curves, which do start out at 100% in Bin 1 in these tasks [dotted=NoPE, red=not in C-RASP]. Thus, on these tasks, NoPE *does* succeed on in-distribution samples. Please let us know if we should make some aspect of the figure or caption clearer to prevent misunderstanding.

We would also like to point out two differences with [2]. First, [2] report results for Addition on heldout lengths 8-16 (their Figure F.5) , much shorter than our heldout bins 100 and 150. Second, [2] report three "Copy" tasks, where NoPE shows length generalization from length 20 to length 40 only in the first two variants (their Figure F.4), which require matching only the length of the string, but not its character sequence. On their third variant, NoPE does not generalize from length 20 to 40 at all.

[2] Kazemnejad, Amirhossein, et al. "The impact of positional encoding on length generalization in transformers."

### Official_Comment — Authors — Response

- Note ID: `2d3ztVPANh`
- Discussion number: `3`
- Invitation: `ICLR.cc/2025/Conference/Submission13115/-/Official_Comment`


#### Title

Response

#### Comment

> (1) The paper focuses on APE and NoPE, whereas many transformers use relative encodings (RPE)
 
We closely follow Zhou et al 2024 in using APE, as our aim was to formalize the RASP-L conjecture. We were also motivated by [1], which found NoPE to be empirically competitive with relative positional encodings in various algorithmic problems. We believe that extending our theoretical study to relative positional encodings is an exciting next step for future research, as we also mention in Section 6.


[1] Kazemnejad, Amirhossein, et al. "The impact of positional encoding on length generalization in transformers."


> (2) The limitation of the proposed theoretical framework is discussed, but it could benefit from more empirical evidence, like where the theory predicts length generalization but empirical performance is poor, or vice versa.

We have not been able to find examples where the theory predicts length generalization but empirical performance is poor, or vice versa. Any such examples would of course be highly interesting for further refining our theory.

### Questions:

> (1) Have you tried the performance of RPE?
 
As described above (and in Section 6), we agree that expanding our theoretical treatment to RPE is a very interesting question. While NoPE can simulate some amount of positional encoding as shown by [1], this is strictly weaker than what APE can do, as we explain in Appendix G.4. We conjecture that the same applies to RPE, and that NoPE may not subsume general RPE. Hence, we believe treating RPE theoretically will require substantial additional technical work, out of scope for the present paper. 

> (2) Why are the model sizes in the experiments task-dependent?


There may be a minimum model size necessary in order to solve a particular algorithmic task. We conjecture that there is a link between model size and the formula complexity (e.g., length, nesting depth) in C-RASP. Zhou et al conjectured that shorter programs are easier to learn, and we thus conjecture that C-RASP complexity could also be used to predict minimum model size. As proving bounds on the smallest program or formula representing a problem is generally nontrivial, we leave rigorous exploration of this idea to future work.

### Official_Comment — Authors — Response

- Note ID: `TsjRTGO6EU`
- Discussion number: `4`
- Invitation: `ICLR.cc/2025/Conference/Submission13115/-/Official_Comment`


#### Title

Response

#### Comment

> (1) The paper is very long -- should it be multiple papers?

Our paper makes progress on multiple but tightly interleaved   fronts, both establishing a length generalization guarantee in an idealized setup (Theorem 7), settling the (non)membership of many problems in C-RASP (Section 4 and Figure 1), and validating predictions empirically (Section 5). We believe that these theoretical and empirical components are tightly linked, and will be most convincing and impactful if presented together.

> (2) The construction of the Limit Transformer is quite elaborate -- is this necessary?

Our guiding question was to formalize when there exists a single APE transformer-like ''algorithm'' for a problem across input lengths, which Zhou et al 2024 described as a key intuition behind their RASP-L conjecture. Formalizing this led us to the notion of limit transformers, which allowed us both to derive a length generalization guarantee for an idealized inference procedure (Theorem 7), and a new result on the expressiveness of APE transformers (Corollary 26).

> (3) How does the setup compare to bounding the number of errors, as in Solomonoff Induction?
 
Thanks for pointing out the link to Solomonoff induction and similar settings. In fact, our setting is quite similar: In our setting, length generalization in the sense of ultimately converging on generalizing correctly when $n$ is large (Theorem 7) is *equivalent* to the number of mistakes being finite. We now make this explicit in Appendix G.5. 

> (4) Overall, is this paper too extensive and long for this conference?

We would like to remark that ICLR has published papers with similar length. For instance, [1,2,3] have 66 to 74 pages. This applies similarly across Machine Learning conferences, and there have been highly influential papers longer than our submission, such as  [4] with 84 pages and [5] with 93 pages.

[1] Panigrahi et al, Effect of activation functions on the training of overparametrized neural nets, ICLR 2020, https://openreview.net/pdf?id=rkgfdeBYvH

[2] Li et al, Provable Memory Efficient Self-Play Algorithm for Model-free Reinforcement Learning, ICLR 2024, https://openreview.net/forum?id=vNiI3aGcE6

[3] Li et al, Risk Bounds of Accelerated SGD for Overparameterized Linear Regression, ICLR 2024, https://openreview.net/forum?id=AcoXPIPh4A

[4] Allen-Zhu et al, Learning and Generalization in Overparameterized Neural Networks, Going Beyond Two Layers, NeurIPS, https://arxiv.org/pdf/1811.04918

[5] Bai et al,  Transformers as Statisticians: Provable In-Context Learning with In-Context Algorithm Selection, NeurIPS, https://arxiv.org/abs/2306.04637

### Official_Comment — Authors — Answers to Questions

- Note ID: `OWGmQP9tYS`
- Discussion number: `5`
- Invitation: `ICLR.cc/2025/Conference/Submission13115/-/Official_Comment`


#### Title

Answers to Questions

#### Comment

### Questions

> Improvements: (I consider all of them optional suggestions, not strict requirements)

> (1) Make more explicit that the theory applies in an idealized limit, and that the function class may not be complete

We have made edits in Section 1 to make more explicit that the  theory applies to an idealized theoretical limit, and that the function class may not be complete.

> (2) I really liked lines 255-264 in terms of clarifying the paper. Maybe the same information can be qualitatively given early in the paper to prime the reader.

We are glad that this content (a high-level sketch of the proof of Theorem 7) is useful. We will add some of this to section 1.

> (3) The paper should be clearer that Limit Transformers are just a mathematical construct

We agree that it is an important point that Limit Transformers are just a mathematical construct, and are neither trained nor implemented. We have added an explicit statement at line 187.

> (4) The functions $\phi_{l,h}$ can have unbounded complexity and do a lot of the heavy lifting.

It is true that Definition 2 does not constrain $\phi_{l,h}$. However, Theorem 7 concerns Limit Transformers in which the functions $\phi_{l,h}$ are strongly constrained due to the requirements of translation-invariance and locality defined in Definition 3. Indeed, we could have introduced that constraint directly in Definition 2, but chose not to, in order to make the presentation modular, hoping this makes the reader's job easier.

### Minor questions

> (1) Is it true that C-RASP defines a subset of $TC^0$ and covers some but not all regular languages and some simple non-regular languages?

This is true. The ability to perform unbounded counting enables representing certain non-regular languages.

> (2) Showing that C-RASP with a small extension can provably be expressed by a Limit Transformer is very interesting. Can C-RASP potentially still be extended without violating this equivalence, or is the current set of operations (likely) complete?

This is an open question. We have not found functions that are expressed by Limit Transformers but not C-RASP[local, periodic]. We acknowledge this in point 6 of Appendix A.

> (3) What is the relation of Theorem 7 to standard learnability / language identifiability results (language identifiability is generally not possible from positive examples only)?

In applying Theorem 7 to formal languages (such as the 17 regular languages in Figure 1), we assume that the training data provides the set of possible next tokens at each positions (line 443). This implicitly includes negative examples, as negative examples are strings that at some point include a symbol that is impossible given prior context.

### Official_Comment — Authors — Response (Part 1)

- Note ID: `s7wIq9NVpr`
- Discussion number: `6`
- Invitation: `ICLR.cc/2025/Conference/Submission13115/-/Official_Comment`


#### Title

Response (Part 1)

#### Comment

> There are quite a few concerns that I have for various parts of the paper. My current score is a reflection of these weaknesses. I would be happy to change my score if the authors can provide satisfactory explanations and no other major concerns appear in the course of discussion.

We thank the reviewer for the detailed reading and their feedback.

We believe the most important points to revolve around

* Detailed questions about the proof of Lemma 17, which we address in detail
* The role of APE vs NoPE, which we now discuss in Appendix G.4. While NoPE can partially simulate positional information, it is not as powerful as APE. We find in both theory and experiment that NoPE does not perform well on tasks not expressible in C-RASP$[\emptyset]$ (Figure 1 and Appendix G.4).
* The role of offset invariance and translation invariance, which we now discuss in Appendix G.6. We empirically show that translation invariance is often approximately satisfied in trained transformers, both small transformers trained on algorithmic problems, and a real-world LM with APE (GPT-2), suggesting that it is favored by standard training.
We also theoretically show that translation invariance is beneficial for ensuring length generalization on a simple induction head task.

### Point-by-Point Reponse

We address each question point-by-point. We paraphrase the questions for the sake of brevity.

>  (1) Can't NoPE simulate APE? Hence, isn't it sufficient to study NoPE, making the analysis much simpler?

It is true that up to a fixed input length $T$, a NoPE transformer can in principle compute positional information, in the sense of computing an activation with value $1/t$ at position $t$ (as in the proof of Theorem 1 in the Kazemnejad et al paper referenced by the reviewer). Importantly, this is weaker than the ways in which APE transformers can use positional information.
In order to perform this simulation, the transformer would need either rapidly increasing MLP weight values or rapidly increasing width for larger values of $T$. For instance, distinguishing close-by positions, say $t-1$ and $t-2$, in such an encoding requires rapidly increasing parameter values as $t \rightarrow \infty$, as the values get arbitrarily close to $0$. In this sense, even a simple task such as an induction head, which requires attention from $t$ to $t-1$, requires parameters rapidly increasing with the input length Appendix G.4). In our theoretical framework, NoPE is not predicted to length-generalize as well as APE on such a task.
Indeed, empirically, we find NoPE transformers not to perform well on a variety of tasks not expressible in $C-RASP[\emptyset]$ (Figure 1).

In our theory, NoPE is a simple special case of the more powerful APE setup (lines 265), in which special case one can indeed prove Theorem 7 without limit transformers. This is explained in Appendix B.2.

> (2) Regarding assumptions made by the theory
 
> (2a) The theory uses the constraint that all transformers are offset-invariant

We believe that the reviewer refers to ''*In line with the assumed setup, we focus on transformers whose input-output behavior is invariant across offsets: $T(x,o) = T(x,o')$ for any $0 \leq o, o' \leq N(T) - |x|$*''.
We have removed this statement, as it is not needed at this point.
However, we do assume that the Hypothesis Class requires translation invariance, which we discuss under (5).

> (2b) The inference procedure uses an additional regularizer that is not explicitly enforced in standard training

It is true that the additional regularizer (Equation 8) is not explicitly enforced in standard training.
We argue that, nonetheless, it is likely to reflect an inductive bias of standard initialization: We know from Proposition 54 that, if one randomly initializes transformers at context length $N$, and sufficient width, the regularizer will in expectation remain bounded even as $N \rightarrow \infty$. Boundedness of the additional regularizer as $N \rightarrow \infty$ is thus likely to be favored by standard initialization. 

More broadly, we argue that our results are highly interesting even if the theoretical setup makes idealizing assumptions. Despite a lot of empirical research, theoretical understanding of length generalization of transformers is in its infancy. The RASP-L Conjecture has not previously been formalized; there is no existing evidence beyond experiments. Our work already enables a proper formalization of the conjecture, in terms of limit transformers and C-RASP. As a step towards theoretically understanding  length generalization, we study length generalization in an idealized setup abstracting away from training dynamics, as we clearly acknowledge in Sections 1 and 6. We lay groundwork for future work, both through theory (we provided a formal class of languages whose expressivity can be rigorously understood) and experiments (we provide evidence that this language class tracks empirical length generalization behavior).

### Official_Comment — Authors — Part 2

- Note ID: `wKw3Dg2WBc`
- Discussion number: `7`
- Invitation: `ICLR.cc/2025/Conference/Submission13115/-/Official_Comment`


#### Title

Part 2

#### Comment

> (3)   Regarding the definition of limit transformers 

> (3a) Why do limit transfomers need both positional encodings and $\phi_{l,h}$ functions?

Because Limit Transformers have finite width $d$, the positional encodings ${\bf p}_i$ in the Limit Transformer can only absorb some parts of the information from the positional encodings ${\bf p}_i$ of the standard transformer, namely those that can be absorbed into finite-width and finite-precision encodings.  It is desirable to assign Limit Transformers a finite width, so their definition is as close to standard transformers as possible.


Other parts of the positional information cannot be coded in this way -- for instance, it's not possible in a single transformer operating on unboundedly long inputs, to implement attention from position $t$ to position $t-1$ with a fixed-width positional encoding (cf. line 864). Hence, we introduce functions $\phi_{l,h}$ to absorb such remaining positional information. Due to its bounded width,
$y_i$
will generally contain less positional information in the Limit Transformer than in a standard transformer, necessitating the additional term using $\phi_{l,h}$ in Equation 6.




>   (3b) Other than Eq. 6, is the rest of the construction of limit transformer same as standard transformer?



Yes, everything else is identical to the transformers from Section 2. Limit transformers have the same attention mechanism (Eq. 3); they just differ in adding $\phi(i,j)$ to the attention logits.


>  (3c) If we are happy with finitely many positional encodings, then why not just do some periodic encodings in the standard transformer?

Importantly, while the Limit Transformer has only finitely many distinct positional encodings, the standard transformers $T_1, T_2, T_3, ...$ found by the Inference Procedure can have unboundedly many distinct positional encoding vectors, because the width $d$ is not bounded by the definition of the Hypothesis Class. We find it useful to separating, in translating to Limit Transformers, the power of unbounded-width encodings into (i) bounded-width and bounded-precision encodings $p_i$ and (ii) the functions $\phi_{l,h}$. As explained in point 7 of Appendix A (expanded in the revision), $p_i$ capture periodic information, whereas $\phi_{l,h}$ encapsulate local relations. Hence, positional abilities of Limit transformers go beyond the periodic encodings $p_i$.

We study standard transformers with learnable APE encodings, rather than the periodic encodings suggested by the reviewer, to match the setup of Zhou et al 2024 [1]. Extending our theory to hard-coded periodic encodings could be another interesting topic.

We also would like to emphasize that Limit Transformers are a mathematical construct that helps us prove things about standard transformers (as defined in Section 2). There may be other ways of defining these limiting objects that allow proving the same results.

Re *"we can operate with NoPE, express finitely many positional encodings, and simplify the whole story right?"*: Importantly, limit transformers have both bounded-width and bounded-precision positional encodings $p_i$ and $\phi_{l,h}$ functions; jointly, these simulate the power of standard APE transformers, and are substantially more powerful than beyond NoPE, as we explain under point (1).

> (4) The presentation of the translation-invariance and locality constraints imposed on $\phi_{l,h}$ and positional encoding might need to be clearer.

We have rewritten Definition 3 to make this clearer. We would welcome any further advice on how to improve this aspect.

[1] Zhou et al, What algorithms can transformers learn?, ICLR 2024

### Official_Comment — Authors — Part 3

- Note ID: `7U8YYSYc7k`
- Discussion number: `8`
- Invitation: `ICLR.cc/2025/Conference/Submission13115/-/Official_Comment`


#### Title

Part 3

#### Comment

> (5) Why is translation invariance a reasonable constraint in the definition of the Hypothesis Class? Isn't there a gap between theory and experiments?

We agree that translation invariance merits further consideration.
We distinguish two relevant properties:

* (A) offset-invariance of the *input-output behavior*; that is, the transformer's output $T(x,o)$ is independent of $o$;
* (B) translation-invariance of the *product functions*, such as $p_i^T K^T_{l,h} Q_{l,h} p_j$ as assumed in Definition 4 (Hypothesis Class).

Our experimental setup (Section 5) assumes that every training sample is presented with a random offset. This setup is a simplification of the setup in Zhou et al 2024, and aims to mimick how LLMs need to solve reasoning tasks no matter where they appear in a context. Due to this property of the experimental setup, the transformers are trained to be offset-invariant in their input-output behavior (A), in agreement with the theory.

Translation-Invariance (B) is a stronger requirement, and implies (A). We now discuss (B) in Appendix G.6, where we show the following:


- We provide theoretical evidence that translation invariance is beneficial for length generalization (Appendix G.6.1). There, we describe a sequence $T_n$ of transformers violating translation invariance, where $n$ operates at lengths $\leq n$, where $\sup_n \mathcal{R}(T_n)<\infty$ and each $T_n$ describes a target function (a simple induction head task) at lengths $\leq n/2$, but no $T_n$ represents the task at length $n$ -- that is, the models fail to length-generalize. This is in contrast to the translation-invariant setup, where such a situation necessarily leads to $T_n$ length-generalizing correctly for large $n$.


- Translation invariance is approximately satisfied in trained transformers (Appendix G.6.2), both in small transformers trained on algorithmic problems, and in a real-world LM, GPT-2. This suggests that translation invariance, even though not explicitly enforced, is implicitly favored by standard training, presumably because the target function itself is (at least approximately) offset invariant in these setups.

Overall, we conclude that

* standard training tends to implicitly  favor translation invariance, at least in the setups relevant to our experiments
* translation invariance is theoretically beneficial for length generalization



> (6) On the regularizer in definition 5
> 
>  (6a) How does it enter Lemma 17?



The term is key to ensuring that $f$ is represented by a single LOCAL limit transformer.
Specifically, boundedness of the regularizer term, across all context windows n, entails that  $D_0$ (Equation 15) is finite, which is used to derive a contradiction in line 1090. That in turn is used to show that the $\phi$ functions
are local for a uniform $\tau_\infty$. This is needed for concluding that the function $f$ is representable by a single LOCAL limit transformer.

>  Also, since all positions are equally penalized in this regularizer, why would things far off be more penalized? 

The idea is that, when training on lengths $\leq n/2$, the training set constrains the attention behavior between positions at a distance $\leq n/2$, but not at larger distances $>n/2$. The regularizer discourages attention at such greater distances.
For instance, consider an induction head task requiring attention to the previous token in Layer 1. Without an inductive bias discouraging attention (either through a regularizer, or implicitly through initialization), when tested on longer inputs, the attention head in Layer 1 could end up attending both to the preceding position and to some far-away position at distance $>n/2$, because the training data had no information about such distances.


>  Also, the justification based on initialization making the regularizer small is not fully clear.

We refer to our response under Point (2b): Proposition 54 shows that, if one randomly initializes transformers with increasing maximum context length $n$, the regularizer will, in expectation, stay bounded even as $n$ diverges.

> (6b) What is the role of limit transformers and the regularizer in NoPE?

We refer to our response to Point (1), where we explain that APE is substantially more powerful than NoPE. Technically, NoPE is a special case with $p_i \equiv 0$, $\phi(i,j) \equiv 0$ (lines 264-266). Indeed, in this special case, limit transformers are not needed to arrive at our length generalization guarantee. Limit transformers are used to treat APE, which is substantially more powerful than NoPE. We explicitly remark this in Appendix B.2.


> (7) Why are the below-diagonal values of $\phi(i,j)$ not constrained?


The values below the diagonal are irrelevant as we are considering only causally masked transformers (line 91), in line with standard LLMs and with Zhou et al 2024.


>  (8)  Minor remark on line 326/327 Shouldn't the Q_a in the RHS be Q_a(j) and not Q_a(i)?

Agreed, fixed.

### Official_Comment — Authors — Part 4

- Note ID: `b0jzObennx`
- Discussion number: `9`
- Invitation: `ICLR.cc/2025/Conference/Submission13115/-/Official_Comment`


#### Title

Part 4

#### Comment

>  (9)  Notation remark on Theorem 9. In an unfortunate use of notation, u call Psi function local and Phi function periodic. Earlier u had used phi for local in the definition 3.

Thanks, fixed.

>  (10)  Since the results from the work hold for NoPE positional embeddings. From Theorem 2 in https://proceedings.neurips.cc/paper_files/paper/2023/file/4e85362c02172c0c6567ce593122d31c-Paper-Conference.pdf, the results should extend to relative positional encodings too?

As discussed in our response to Point 1, the simulation of positional encoding in NoPE is limited in its capacity, at least compared to APE. We thus conjecture that, similarly, NoPE cannot fully simulate RPE. We believe that theoretical understanding of RPE, analogous to our results for APE, would take substantial additional technical work and is out of scope. 


> (11) What is the role of $N_0$? Would it be interesting to explain why, in practice, transformers can generalize well with modest training lengths?

We agree that predicting and validating realistic bounds on $N_0$ is a very important next direction for research. Deriving an $N_0$ from our proof would give a valid and nonvacuous, but likely overly pessimistic estimate. More realistic bounds on $N_0$ will likely require advances in understanding SGD dynamics on transformers, which remains hard to understand, in particular in the multi-layer setup, needed for many of the functions in Figure 1.


>    Concerns on the proof of Lemma 17:

We thank the reviewer for the close reading, and have expanded the proof to clarify all these aspects.

> a) In line 976 you say there are only a finitely many settings traversed by the limit transformer $\tilde{T}_i$. Why is this the case? Can you properly justify?

Let $A := sup_i \mathcal{R}_\infty(\tilde{T}_i) < \infty$.

The number of limit transformers $\tilde{T}$ with
$R_\infty(\tilde{T}) \leq A$
is finite except for the functions $\phi_{l,h}$, because $A$ bounds  (1) the number of parameters, (2) their magnitudes, (3) the precision at which they are represented.
We make this explicit in line 993 of the new PDF.


>    b) In equation 9, you say that we select an R(Tn) that is less than 1/n + inf (R(T)). In line 981, you argue that R(Tn) should converge because inf R(Tn) is bounded and monotonically increasing. This argument only tells that the RHS in equation (9) converges, which is the upper bound. Why does it imply that the LHS converges? You crucially use the existence of this limit in equation (14).

First, note that
$$R(T_n) \in [\frac{1}{n} + \inf_{T \in U_n} (R(T)), \inf_{T \in U_n} (R(T))]$$ Due to boundedness and monotonicity, $\inf_{T \in U_n} (R(T)))$ converges to a limit, say $\tilde{R}$. 
Since $1/n \rightarrow 0$, the width of the interval converges to 0. The Squeeze Theorem then implies that $R(T_n) \rightarrow \tilde{R}$. We make this explicit in line 1008.


>    c) In line 995, you say that lim D_v_i(v_i) is D_0. Why does this limit exist?

By definition of $R_-$, 
$$D_{\nu_i}(\nu_i) = R(T_{\nu_i}) - R_-(T_{\nu_i})$$ As both terms in the RHS converge, the LHS also has a limit.
We make this more explicit in the text.


>    d) Below equation 17, you state "As this function is monotonically increasing, and as phi_{l,h} has bounded precision, there must be $\tau_{\infty}$..." Why does this hold true? I don't quite follow.

 As $D_\infty(\tau)$ is monotonically increasing and bounded, it converges.  Due to bounded precision, it only takes values in discrete steps (say, only multiples of $2^{-p}$ for some $p$); hence, it must attain the limit at some specific $\tau$, which we refer to as $\tau_\omega$.
We have made this more explicit in line 1040.

>    e) In line 1010-1017, you construct a sequence of T'n satisfying certain properties. Why does this have to exist? For instance why is $D_{\nu_{i(n)}}(n) = D_{\infty}(n)$ true?

We have expanded (line 1042-1058 of the new PDF). The equality in question is $$D_{\nu_{i(n)}}(n) = \liminf_{j\rightarrow\infty} D_{\nu_j}(n) = D_\infty(n)$$ Regarding the first equality, such a $i(n)$ exists because $\phi_{l,h}$ has fixed precision, which entails that the $\lim\inf$ is attained infinitely often. Regarding the second equality, this is the definition of $D_\infty(n)$.



>    f) The phi_l,h(i,j) in equation (10) should also bear the index n as it would be different for each limit transformer.

Thanks for the suggestion. We now use a superscript to indicate this, e.g. $\phi_{l,h}^{(\tilde{T}_n)}(i,j)$ (line 989).


> In the definition of  $D_n(\tau)$, how can $\tau$ exceed $n$?

We now explicitly restrict summation to $\min(n,\tau)$.

### Official_Comment — Authors — Part 5

- Note ID: `aCYca1bz8z`
- Discussion number: `10`
- Invitation: `ICLR.cc/2025/Conference/Submission13115/-/Official_Comment`


#### Title

Part 5

#### Comment

>    g) I do not follow the inequality in line 1044?

This refers to

$$ D_0 =   \limsup_{n\rightarrow \infty} D_{n}(n) \geq \limsup_{n\rightarrow \infty} D_{n}(\tau_\infty) + 2^{-2p} \geq \liminf_{i\rightarrow\infty} D_{\nu_i}(\tau_\infty) + 2^{-2p} = D_0+ 2^{-2p}$$
The first inequality holds because
$D_n(n) \geq D_n(\tau_\infty)$
whenever $n \geq \tau_\infty$, simply because $D_n(\cdot)$ is monotonically increasing for each individual $n$.

The second inequality holds because
$\nu_1, \nu_2, \dots$
is a subsequence of
$1, 2, \dots$;
hence a $\lim \sup ...$ over the larger sequence upper-bounds the $\lim \inf ...$ over the subsequence.
We have made the steps more explicit in line 1091 of the new PDF.

>    (12) In line 1049-1052, you say that set of functions traversed becomes stationary, what do u precisely mean here?

We have made this more explicit and rephrased in line 1100-1107 of the revised PDF. 
It means that after $N_0$, all Limit Transformers traversed must be functionally equivalent to $f$.

>  (13)  In line 1121 to 1127, how does the C logN fall out? Do you mean that since you partition stuff into N positions and that takes log N bits to compute, we get C log N?

We have made this more explicit in lines 1210-1223 of the revised PDF. 
Alice can partition the positions into a constant number of set, and for each of them needs to transfer the number of positions in that partition.



> (14) How does periodicity fall out in the limit transformer?


This is an interesting observation. Indeed, periodicity falls out as a consequence of translation-invariance by Lemma 48: translation-invariant positional relations mediated by finite-rank matries are periodic. Hence, we are able to separate the positional relations in a sequence of transformers $T_n$ with bounded $\mathcal{R}(T_n)$ into local and periodic components. We now explain this better in Appendix A, point (7).

> Why not have periodic positional encodings on the original transformer, instead of the offset independence condition?

We refer to our discussion of offset independence and translation invariance independence above, where we show that offset independence is empirically and theoretically well-motivated. Our aim is to describe the behavior of general learnable APE encodings, as in our experiments and the closely related paper, Zhou et al 2024.

### Official_Comment — Authors — Part 6

- Note ID: `AeD80186Jy`
- Discussion number: `11`
- Invitation: `ICLR.cc/2025/Conference/Submission13115/-/Official_Comment`


#### Title

Part 6

#### Comment

We once again thank the reviewer for the close reading!

Please let us know if there are any remaining questions.

### Official_Comment — Authors — Global Response

- Note ID: `wiuk4JlKeR`
- Discussion number: `12`
- Invitation: `ICLR.cc/2025/Conference/Submission13115/-/Official_Comment`


#### Title

Global Response

#### Comment

We thank all reviewers for their overall positive assessment, extensive comments and constructive criticism. The detailed feedback on both the work's presentation and technical details have allowed us to make the paper more readable and the results more sound. We summarize the reviewers' major concerns and list the changes we have made to address them. 

We have uploaded a revised PDF, with changes highlighted in **blue**.

> The treatment of positional encodings produces seemingly contradictory results with Kazemnejad et al. 2023

The cited paper shows that NoPE transformers can simulate APE transformers up to a large input size. Taken at face value this suggests we don't need limit transformers to prove guarantees for APE transformers. However, there is an important distinction: the parameters in a transformer must increase rapidly to perform the cited simulation on larger inputs. Because of this, our theory predicts that NoPE transformers will length-generalize worse than APE transformers on certain tasks. Indeed, our experimental results confirm that APE transformers can length-generalize on several problems that NoPE transformers do not. Furthermore, the experimental setup in the cited paper differs notably from ours. We have written a treatment of this issue in Appendix G.4, as we believe it is quite important (and thank the reviewers for pointing it out). We also provide further experimental results confirming that APE generalizes better than NoPE in agreement with our theory, and even when performing much broader hyperparameter search for NoPE (Appendix G.7): On a family of induction head tasks, APE achieves ~99% accuracy in generalizing to 3 times the training length, whereas NoPE shows <5% there.

> The requirement of translation-invariant product functions is unrealistic

It is true that standard training of transformers does not enforce translation invariance in the product functions. However, we can *prove theoretically* that translation invariance is helpful for length-generalization (on induction heads for instance), and *show empirically* that length-generalizing transformers often learn translation invariant product functions in practice (both in transformers trained from scratch and in GPT-2). As such, we suggest translation-invariance is favored by standard training even though it may not be explicitly enforced. We have written a new section on this in Appendix G.6.

> The paper is too elaborate

Because theoretical understanding of length generalization of transformers is still in its infancy, building a formal framework for it requires the synthesis of many different ideas. We believe that presenting our simultaneous progress in multiple areas - both theoretical and empirical - is necessary to form a solid theoretical framework. In order to make the paper more approachable, we have rewritten the introduction and changed prose throughout the body of the paper. 

> The practical impact of this paper is not clear

Indeed, our paper focuses on a theoretical framework explaining the length-generalization on algorithmic problems (rather than analyzing transformers in practice on natural language tasks). Nevertheless, we believe that a solid theoretical understanding can enable future work to have a more immediate practical impact. We have given pointers towards potential avenues for this (new positional encodings, improved interpretability) and modified our introduction to make the limitations of our setting clearer. 

Overall, we thank the reviewers for their constructive criticism and support.

### Official_Comment — Authors

- Note ID: `OR3g8PzujH`
- Discussion number: `13`
- Invitation: `ICLR.cc/2025/Conference/Submission13115/-/Official_Comment`


#### Comment

**Addendum:** We have uploaded a new draft version with an added Appendix G.7, in which we report further experiments confirming that **APE generalizes much better than NoPE** on a family of induction head-like tasks, in agreement with our theoretical predictions. On a family of induction head tasks, APE achieves ~99% accuracy in generalizing to 3 times the training length, whereas NoPE shows <5% there. This confirms our theoretical point from Appendix G.4 that the simulation of positional information in Kazemnejad et al 2024 is not powerful enough to allow NoPE to subsume APE. Taken together, we believe that our revision convincingly demonstrates why our theory needs to study the more general (and more complex) case of APE, rather than NoPE.

### Official_Comment — Reviewer_odFA — Thank you for the clarifications and comments

- Note ID: `TUEAMaLAf8`
- Discussion number: `14`
- Invitation: `ICLR.cc/2025/Conference/Submission13115/-/Official_Comment`


#### Title

Thank you for the clarifications and comments

#### Comment

I want to thank the authors for the clarifications and changes and improvements to the paper. I am happy with the authors' responses (though I still believe that very long papers at ML conferences are challenging, due to tight review periods with increased workloads, and should not become the norm; I do acknowledge though that ML has somewhat of a lack of "prestigious" journals, and that authors in general prefer to get a top-tier conference publication out of their work).

[Question (4) ] Thank you for correcting me regarding the complexity of $\phi_{l,h}$.

As I wrote in my original review, I do believe that the work is important and is ready to be presented and discussed with the wider ML community. I would now raise my score to a 7, but ICLR does not allow this score this year. I do think that some weaknesses remain, and that the paper does not quite hit an 8. But for the sake of expressing a clear opinion for the decision-making process, I am raising to an 8 (as the only available option), but will keep my confidence low to indirectly indicate that the score is a bit inflated.

### Official_Comment — Authors

- Note ID: `jUCB9eRyUG`
- Discussion number: `15`
- Invitation: `ICLR.cc/2025/Conference/Submission13115/-/Official_Comment`


#### Comment

We thank the reviewer for their response, and are glad that they are happy with our response.

### Official_Comment — Reviewer_H3pw

- Note ID: `29mBXtxM20`
- Discussion number: `16`
- Invitation: `ICLR.cc/2025/Conference/Submission13115/-/Official_Comment`


#### Comment

I thank the authors for their responses. I do think I understand most things. However, if I have to be honest the previous version was quite sloppy given all the changes. I will increase my rating but decrease my confidence as it is hard to verify all details thoroughly in the long draft.

### Official_Comment — Authors

- Note ID: `NxYWkPouMA`
- Discussion number: `17`
- Invitation: `ICLR.cc/2025/Conference/Submission13115/-/Official_Comment`


#### Comment

We thank the reviewer for their response. Thanks also for taking the time to write an extensive review, which helped us to considerably improve the draft.

### Official_Comment — Reviewer_6ZTg

- Note ID: `mEJPXHXtkY`
- Discussion number: `18`
- Invitation: `ICLR.cc/2025/Conference/Submission13115/-/Official_Comment`


#### Comment

Thank you for the response. I have read it and the responses to other reviewers and will keep my score.

### Official_Comment — Reviewer_Vn9i

- Note ID: `LUjeWNpKOT`
- Discussion number: `19`
- Invitation: `ICLR.cc/2025/Conference/Submission13115/-/Official_Comment`


#### Comment

Sorry for the late comment and thank you for the response. During the extended rebuttal period, I read the paper more carefully and realized that I had some misunderstandings earlier (including the stupid question about Figure 1). I want to ask a few additional questions for further clarification.

Questions:
- I am slightly confused about the statement in Definition 6. Shouldn't each $T_i$ be an element of $\Theta_i$, not $\Theta_n$? Based on the explanation stated between Lines 228 to 231 of the revised version, I guess that $T_1$ is an element of $\Theta_1$. Furthermore, following Definition 4, it seems that $\Theta_n \subset \Theta_1$ holds, which would imply $T_1 \notin \Theta_n$.
- I might have missed this, but can any Limit Transformer be translated into a standard Transformer? If so, can you outline the method?
- For a task that is not expressible by the Limit Transformer satisfying Periodic and Local (e.g., n-digit addition), the authors state that any run of the Idealized Inference Procedure will result in a sequence of Transformers with bad properties (Lines 418 to 421). Does this imply that "for Transformers with fixed depth, number of heads, parameter norms, ranks, MLP dimensions, and precision p, there does not exist a length-generalizable solution for n-digit addition"?

Minor suggestion for improving clarity:
- line 186: R(T) is referenced before it is formally defined.

I would appreciate it if the authors provide response for them, but as the rebuttal deadline is approaching, concise answers would be perfectly fine.

### Official_Comment — Authors

- Note ID: `iAZWNGesZP`
- Discussion number: `20`
- Invitation: `ICLR.cc/2025/Conference/Submission13115/-/Official_Comment`


#### Comment

We thank the reviewer for the close reading and further questions. As the rebuttal deadline is approach, we for now provide concise answers for the questions:

> I am slightly confused about the statement in Definition 6. Shouldn't each $T_i$ be an element of $\Theta_i$, not $\Theta_n$?? Based on the explanation stated between Lines 228 to 231 of the revised version, I guess that $T_1$ is an element of $\Theta_1$ . Furthermore, following Definition 4, it seems that $\Theta_n \subset \Theta_1$ holds, which would imply $T_1 \not\in \Theta_n$.

Thanks for the close reading. The phrasing in Definition 6 is indeed not optimal.
As the reviewer inferred, we intend to say that $T_1 \in \Theta_1$, $T_2 \in \Theta_2$, $T_3 \in \Theta_3$, etc.
We will fix this in our next version.
We believe that this addresses the reviewer's question.
   
>  I might have missed this, but can any Limit Transformer be translated into a standard Transformer? If so, can you outline the method?

Yes, such a translation can be carried out for any finite context length N of the resulting standard transformer, and importantly the translation is uniform in the sense that $\sup_N R(T_N) < \infty$, even though the width of the translation needs to increase. This is formalized in Lemma 47. The outline of the method is as follows; a discussion is also provided in lines 2724-2751 (page 51):
* Intuitively, the positional encodings of $T_N$ consist of those of the Limit Transformer, concatenated with one-hot vectors for the positions from 1 to N. The overall width is thus $d+N$.
* Token embeddings just consist of those of the Limit Transformer, concatenated with the N-dimensional zero vector.
* For each attention head, the $K^T Q$ matrices encode both those of the Limit Transformer (in the first $d$ dimensions) and the $\phi_{l,h}$ functions (in the final $N$ dimensions).
* Complications arise from the fact that this construction (i) does not ensure translation invariance, (ii) does not keep the regularizer in Eq. (8) bounded. This is solved by (i) attending to SOS and computing position relative to that to ensure translation invariance, (ii) routing some of the positional information through the attention mechanism, effectively making it invisible to Eq. (8). This makes the construction more complex, but allows us to prove that translation invariance and  a bounded value for Eq. (8) can indeed be achieved, which in turn is an important insight leading to Theorem 7. This is summarized in lines 2724-2751.

We will add this high-level explanation before the proof of Lemma 47 in the next version.

> For a task that is not expressible by the Limit Transformer satisfying Periodic and Local (e.g., n-digit addition), the authors state that any run of the Idealized Inference Procedure will result in a sequence of Transformers with bad properties (Lines 418 to 421). Does this imply that "for Transformers with fixed depth, number of heads, parameter norms, ranks, MLP dimensions, and precision p, there does not exist a length-generalizable solution for n-digit addition"?

We believe that this is an accurate statement: intuitively, if such a length-generalizable solution existed, the Idealized Inference Procedure should find it; hence, it cannot exist. We will consider adding an explicit statement and proof in the next version; one thing to take care of is to show that this conclusion holds when bounding depth, number of heads, parameter norms, ranks, MLP dimensions, and precision p, but not necessary explicitly bounding Eq. (8).

>  line 186: R(T) is referenced before it is formally defined.

Thanks for catching this. We will fix this in the next version.

### Official_Comment — Reviewer_Gpiw

- Note ID: `lUT7tVNLnh`
- Discussion number: `21`
- Invitation: `ICLR.cc/2025/Conference/Submission13115/-/Official_Comment`


#### Comment

Thank you for your response. I have reviewed it, along with the replies to other reviewers, particularly H3pw, and I will be keeping my score as it is.

### Official_Comment — Reviewer_Vn9i

- Note ID: `887CEg7bfz`
- Discussion number: `22`
- Invitation: `ICLR.cc/2025/Conference/Submission13115/-/Official_Comment`


#### Comment

Thanks for the prompt response. Your response helped improve my understanding of the paper.

I believe this is a good, strong paper for the following reasons:
- In prior literature, the feasibility of length generalization for a given task when trained on Transformer could only be roughly conjectured based on the RASP-L hypothesis. This paper makes a significant advancement by proposing a formal criterion to determine whether length generalization is possible or impossible (under the assumption of an "idealized inference procedure").
- The paper "almost" proves that we cannot expect length generalization for n-digit addition and copy with duplicate tokens when using APE or NoPE. This result is quite impressive, as it aligns well with trends observed in recent empirical studies, which have shifted away from APE and instead focused on developing specialized input format [1, 2] or novel position embedding method [3, 4].

One concern is that, I think the explanation of the Limit Transformer (Section 3.1) is somewhat abstract and difficult to grasp the underlying intuition. The later sections of the paper (section 3.2 and onward) are okay. Maybe the authors can provide additional explanation or example in Section 3.1.

Overall, I will maintain my score.

---
References

[1] Zhou, Hattie, et al. "What algorithms can transformers learn? a study in length generalization." ICLR 2024

[2] Zhou, Yongchao, et al. "Transformers can achieve length generalization but not robustly." arXiv preprint arXiv:2402.09371 (2024).

[3] McLeish, Sean, et al. "Transformers Can Do Arithmetic with the Right Embeddings." NeurIPS 2024

[4] Cho, Hanseul, et al. "Position Coupling: Leveraging Task Structure for Improved Length Generalization of Transformers." NeurIPS 2024

### Official_Comment — Authors

- Note ID: `4oVowRsUCM`
- Discussion number: `23`
- Invitation: `ICLR.cc/2025/Conference/Submission13115/-/Official_Comment`


#### Comment

We thank the reviewer for their feedback and the close reading of the paper. We agree about the link to recent empirical observations regarding addition and copying, and will do our best to increase intuition in Section 3.1.

## Decision

### Decision — Program_Chairs — Paper Decision

- Note ID: `IkOlDUDJ0X`
- Discussion number: `1`
- Invitation: `ICLR.cc/2025/Conference/Submission13115/-/Decision`


#### Title

Paper Decision

#### Decision

Accept (Poster)


---

# Fourier Position Embedding: Enhancing Attention’s Periodic Extension for Length Generalization — OpenReview 审稿全文归档

- Venue: **ICML 2025 poster**
- OpenReview forum: [https://openreview.net/forum?id=ZfDNDkg7Dh](https://openreview.net/forum?id=ZfDNDkg7Dh)
- Official paper page: [https://proceedings.mlr.press/v267/hua25b.html](https://proceedings.mlr.press/v267/hua25b.html)
- Reviewer handles are public OpenReview pseudonyms; no identity resolution is attempted.
- Source: ReviewArena public release (OpenReview-sourced). ICML 2025 uses the venue’s 1–5 Overall Recommendation scale; do not compare these numbers directly with ICLR’s 1–10 scale.

## Review Inventory (4 Official Reviews)

- Final decision: **Accept (poster)**

## Official Reviews

### Review 1 — Reviewer_FqQv

- Review ID: `qjqfqNTfEc`

#### Summary

## update after rebuttal

I read the latest clarification by the authors, and understand q and k in Re[qk*e^{i{m-n}\theta}] are not exactly the 2-dim vector [q_x, q_y]^T but an implicit complex number. I raised my score back.
******

This paper points out that in RoPE, different dimensions correspond to different frequencies,
and suggests viewing the interaction between queries and keys from the perspective of non-Uniform DFT (Eq.2 - Eq.4).
(However, there is an unclear point here: while RoPE is motivated by complex number rotation, its implementation is actually based on vector rotation--—they are not equivalent. The authors' analysis is based on the former, but the actual analysis object is the latter, and the authors' implementation is also the latter. Therefore, at minimum, they should explain this gap.)

Based on this **deconstructive analysis**,
the authors identify two issues:
1. If we consistently view RoPE from a signal perspective, we cannot ignore the spectral leakage caused by activation functions and linear layers;
2. Considering that low frequencies correspond to stable components in long contexts, while the text used in training is not very long, these low-freq components are insufficiently trained.

The authors propose two improvements to RoPE:
1. Allow feature dimensions that previously corresponded to a *single* frequency to now correspond to *multiple* frequencies, while still maintaining a *primary* frequency, enabling more flexible adjustment of the disturbed spectral information.
2. Set the insufficiently trained dimensions directly to 1.

In experiments, the authors primarily validate the benefits of FoPE for length extrapolation capabilities. Experimental scenarios include pre-training, continual pre-training, and fine-tuning.

#### Claims And Evidence

I have concerns about the following claim:

The authors view the interaction between queries and keys from the perspective of non-Uniform DFT (Eq.2 - Eq.4),
and they hope this perspective can be maintained throughout the model, thus proposing two improvements.
The starting point of these approaches is understanding RoPE's formula as Eq.2.
However, **it should be noted that RoPE's actual implementation is not Eq.2.**
Eq.2 can be considered as the heuristic starting point of RoPE,
while the actual implementation rotates each pair of dimensions by a certain angle, which is not equivalent to Eq.2,
therefore, in fact **we cannot write a strict Fourier form equation**.

It can be said that:
**RoPE has both a "heuristic approach" and a "practical approach",
and the authors' analysis is based on the former "heuristic approach" but improves upon the "practical approach".**

I would like to hear other reviewers' and AC's opinions on this point.

#### Methods And Evaluation Criteria

Yes.

#### Theoretical Claims

There are no specific theoretical claims, but the paper's analysis has certain theoretical aspects. Please see my concerns in (Claims And Evidence).

#### Experimental Designs Or Analyses

Yes. 

PS: I have not personally conducted similar experiments and have insufficient understanding of the experimental details.

#### Relation To Broader Scientific Literature

This paper provides insights into understanding Transformer architecture and solving long-range dependency problems from a signal analysis perspective.

#### Questions For Authors

Please refer mainly to the (Claims And Evidence) section,
and also see the (Other Weaknesses) and (Other Comments Or Suggestions) sections

#### Rating

`3`

#### Rebuttal

Thanks for the insightful comments from the reviewer, we are going to make clarifications for the concerns above.

# Clarification for the "heuristic approach" and "practical approach" of RoPE/FoPE
It seems this is the major concern of the reviewer, we acknowledge the importance of clarifying this point:

1. **The Rotary Matrix implies the real part of the complex number in Matrix Space, which is also the main difference between the "heuristic approach" and "practical approach" of RoPE.** The derivation is in Sec 3.4 of Roformer's original paper [1], we also provide a derivation under our understanding: 
In the 2D case of RoPE, the vectors $q$ and $k$ can be viewed either as 2D plane vectors ($\mathbf{q}=(q_x,q_y)^T$) or as vectors in the complex plane ($\mathbf{q}=\|\mathbf{q}\|e^{i\theta_q}$, $\|q\|=\sqrt{q_x^2+q_y^2}$, $\theta_q = \arctan{\frac{q_x}{q_y}}+k\pi$ ). The inner product of the two vectors is equal to $\langle\mathbf{q},\mathbf{k}\rangle=Re[\mathbf{qk}^*]$ (the proof is at the end of this reply).

2. **As taking the real part does not change the frequency-domain property of a vector, thus: ① RoPE's actual implementation still achieves Eq.2; ② we can still write a strict Fourier form equation.**

3. **As a result, our theoretical analysis and implementation are well-aligned.** To provide better clarity, we will explicitly include a derivation in the future revisions.

4. **Our further empirical results also demonstrate that the actual implementation of RoPE properly achieves its theoretical motivation, also well-aligning with our frequency-domain analysis (See https://anonymous.4open.science/r/FoPE-Supplementry-Material/Empirical_Analysis_on_Spectrum_Damage.pdf).** In this experiment, we force every token only has one frequency component in the first layer. Then, we compare the spectrum of the first and second layer with DFT on attention map (aligned with the derivation in Sec 2.2).

# Relationship between fourier/signal modeling and multiple stacked blocks

- Firstly, we want to clarify that, **as the Fourier Transform is a linear transform [2], it does not change the main properties of models.** Thus, the multiple stacked blocks are still in charge of modeling more diverse/complex and higher-order functions/signals, which improves the expression ability of models.
- Secondly, based on our modeling from a signal/frequency-domain perspective, **different layers actually process signals of different frequency distribution.** This is useful for improving the expression ability of models. However, **RoPE wrongly regards the frequency distribution as same in different layers, leading to drawbacks in length generalization**.
- Lastly, our modeling in frequency-domain can analyze the periodicity of attention mechanism, providing insights into the drawback of RoPE in length generalization.

# Typical range of values for $\omega_m$
We have shown these statistics in Appendix D.3 (line 727-728). The frequencies of RoPE are between (0, 1] and samples more densely near 0.

# Comparison with more baselines on more benchmarks 
We also conduct more supplementary experiments, please check our reply to Reviewer LYp1 for detailed results.

---
## Reference
[1] Jianlin Su, et al. RoFormer: Enhanced Transformer with Rotary Position Embedding.

[2] Oppenheim, et al. Signals and Systems.

---
## Additional Proof
$$\langle f_q(\mathbf{x_m},m),f_k(\mathbf{x_n},n)\rangle$$
$$=\langle \mathbf{q}e^{im\theta},\mathbf{k}e^{in\theta}\rangle$$
$$=\text{Re}[\mathbf{qk}^*e^{i(m-n)\theta}]$$
$$=\text{Re}[\|q\|\|k\|e^{i[\theta_q-\theta_k+(m-n)\theta]}]$$
$$=\|q\|\|k\|\cos(\theta_q-\theta_k+(m-n)\theta)$$
$$=\|q\|\|k\|\left[ \cos(\theta_q-\theta_k)\cos(m-n)\theta - \sin(\theta_q-\theta_k)\sin(m-n)\theta\right]$$
$$=\|q\|\|k\|\left[(\cos\theta_q\cos\theta_k+\sin\theta_q\sin\theta_k)\cos(m-n)\theta-(\sin\theta_q\cos\theta_k-\cos\theta_q\sin\theta_k)\sin(m-n)\theta \right]$$
$$=(q_xk_x+q_yk_y)\cos (m-n)\theta - (q_yk_x-q_xk_y)\sin (m-n)\theta$$
$$=[q_x, q_y][\cos (m-n)\theta, -\sin (m-n)\theta; \sin (m-n)\theta, \cos (m-n)\theta][k_x, k_y]^T$$
$$=[q_x, q_y][\cos m\theta, \sin m\theta; -\sin m\theta, \cos m\theta][\cos n\theta, -\sin n\theta; \sin n\theta, \cos n\theta][k_x, k_y]^T$$
$$=(\mathbf{R_{\theta,m}q})^T(\mathbf{R_{\theta,n}k})$$
$$=\mathbf{q^T R_{\theta,m-n}k}$$

#### Other Strengths And Weaknesses

Other Strengths:
1. Provides a deconstructive analysis of existing models；
2. The proposed method is lightweight and effective；
3. Although I am not very familiar with the experimental approaches in this direction, I find the authors' writing of the experimental section to be very fluent.

Other Weaknesses:
- I still have concerns about the signal-based explanation. First, there are the concerns raised in (Claims And Evidence). Additionally, I am concerned about: in the authors' analysis, attention with RoPE becomes an inverse Fourier transform form, which essentially treats the input as a frequency domain response. However, if this is the case, how can we analyze multiple stacked blocks from a signal perspective?

#### Other Comments Or Suggestions

What is the typical range of values for $\omega_m = 1/{\theta}^{(2m/M)}$ in Eq.2? This should be explicitly stated.

#### Essential References Not Discussed

Not sure.

#### Was Revised

`false`

#### Extra Scores

{}

---

### Review 2 — Reviewer_LYp1

- Review ID: `kvdxBmyz69`

#### Summary

This paper analyses the limitations of Rotary Position Embedding (RoPE) in extending language model context length using Discrete Signal Processing theory. It identifies spectral damage from linear layers, activation functions, and insufficient frequency training as key issues affecting RoPE’s periodicity. To address this, the authors propose Fourier Position Embedding (FoPE), which constructs a Fourier Series and removes harmful frequency components to enhance length generalisation.

#### Claims And Evidence

The claims are generally supported  and shows performance (length generalisation) gains across benchmarks. However, it does not address the computational cost of FoPE, which is important for understanding the trade-off between performance and efficiency.

#### Methods And Evaluation Criteria

The proposed methods and evaluation criteria are appropriate. FoPE enhances length generalisation by improving position encoding, addressing spectral issues in long-context attention. The evaluation on diverse benchmarks (e.g., GovReport, MultiNews, TREC) effectively tests performance across varying context lengths. However, including more complex reasoning tasks would provide a more comprehensive assessment.

#### Theoretical Claims

No.

#### Experimental Designs Or Analyses

While the paper demonstrates performance improvements on long contexts, it doesn't provide insights into how the Fourier Position Embedding method impacts computation costs of training and inference.

#### Relation To Broader Scientific Literature

This paper extends Rotary Position Embedding (RoPE) by analyzing its frequency-domain properties using Discrete Signal Processing (DSP) and identifying spectral distortions that hinder long-context generalization. Building on prior work in positional embeddings (Su et al., 2021; Press et al., 2022) it introduces Fourier Position Embedding (FoPE), which filters harmful frequency components to improve attention’s periodicity.

#### Questions For Authors

1) Could the authors include benchmarks on reasoning tasks like MMLU and GSM8K to demonstrate greater task diversity?

2) The authors are asked to provide a comparison of computational analysis.

#### Rating

`3`

#### Rebuttal

Thanks for the detailed comments from the reviewer, we would like to make several clarification below.

# Computation cost of FoPE is similar to RoPE
We agree with the reviewer that the efficiency and computation cost are essential for Position Embedding, thus:
- FoPE keeps the rotary matrix having the similar shape as RoPE.
- FoPE keeps the similar pipeline as RoPE: ① pre-compute the rotary matrix and save the matrix in cache before training; ② take out the matrix for rotation during training. 

Therefore, FoPE is as efficient as RoPE, independent of the model scale. (Reviewer FqQv also acknowledges that FoPE is lightweight and effective in "Other Strengths And Weaknesses")

# Evaluation on more tasks and baselines
It is a nice suggestion to evaluate FoPE on more diverse tasks, thus we supplement the following experiments:
- We validate the effectiveness of FoPE by evaluating on 10+ more benchmarks and comparing with more 3 more baselines. These supplementary experiments are conducted on OLMo-1.2B training on C4 datasets, keeping all of the settings similar to the experiments in Sec 5.2 (line 293-329). We do not show the results on GSM8K as all methods achieve nearly 0 acc, which may be caused by the lack of math data in C4.

| methods | avg acc | basic_arithmetic | social_iqa | winogrande | openbook_qa | sciq  | hellaswag | piqa | commonsense_qa | arc_easy |
|-|-|-|-|-|-|-|-|-|-|-|
| NoPE | 42.14 | 25.67 | 43.71 | 51.86 | 29.80 | 76.70 | 41.83 | 68.83 | 31.61 | 51.40 |
| ALiBi | 42.93 | 24.97 | 42.53 | 53.12 | 31.40 | 77.70 | 43.06 | 69.42 | **33.42** | **53.68** |
| KERPLE | 43.22 | 25.03 | 43.81 | **54.07** | 32.40 | **78.20** | 43.65 | 69.64 | 32.92 | 52.46 |
| FIRE | 42.38 | 25.60 | 42.63 | 49.88 | 33.40 | 77.00 | 42.75 | 69.31 | 32.92 | 50.35|
| RoPE | 42.98 | 24.60 | 43.45 | 51.54 | **33.60** | 77.10 | 43.36 | **70.13** | 33.01 | 52.98 |
| FoPE | **43.37** | **26.17** | **44.12** | 53.20 | 32.20 | 77.80 | **43.83** | 70.08 | 32.92 | 53.33 |

| methods | avg acc | mmlu_stem | mmlu_social_sciences | mmlu_humanities | mmlu_other |
|-|-|-|-|-|-|
| NoPE | 25.81 | 25.47 | 25.38 | 28.01 | 24.39 |
| ALiBi | 25.99 | 25.93 | 27.20 | 26.44 | 24.39 |
| KERPLE | 26.16 | 26.02 | 25.97 | 27.82 | 24.82 |
| FIRE | 26.04 | 26.34 | **27.41** | 26.25 | 24.16 |
| RoPE | 26.68 | 27.01 | 27.26 | 27.87 | 24.53 |
| FoPE | **27.57** | **27.30** | 27.31 | **29.89** | **25.79** |

| methods | avg ce loss | natural_qs_open | trivia_qa_wiki | arc_easy |
|-|-|-|-|-|
| NoPE | 1.4513 |1.4334 | 1.6129 | 1.3077 |
| ALiBi | 1.4242 |1.3879 | 1.6057 | 1.2789 |
| KERPLE | 1.4284 | 1.4149 | 1.5878 | 1.2825 |
| FIRE | 1.4580 | 1.4365 | 1.6258 | 1.3118 |
| RoPE | 1.4225 | 1.4114 | 1.5973 | 1.2588 |
| FoPE | **1.3941** | **1.3818** | **1.5736** | **1.2272** |

- **Although FoPE is primarily designed for length generalization and long-context tasks, it also performs comparably or even surpasses all baseline methods**. But on MMLU, all methods have weak performance, perhaps caused by the training data. In the future, we may conduct more experiments on datasets containing reasoning and math data.
- We also conduct evaluation similar to those in Sec 5.2 (See https://anonymous.4open.science/r/FoPE-Supplementry-Material/Supplementary_Main_Experiments.pdf). **FoPE still demonstrates the best accuracy on passkey retrieval and the second-best ppl on C4 (only slightly behind ALiBi).**

#### Other Strengths And Weaknesses

Strengths:

1) Frequency-domain analysis of RoPE using Discrete Signal Processing (DSP), offering new insights into spectral distortions affecting long-context generalization.  
2) Practical impact with Fourier Position Embedding (FoPE), demonstrating strong improvements in context extension across model scales and tasks.  

Weaknesses:

1) Computational cost of FoPE is unclear, especially for large-scale models.

#### Other Comments Or Suggestions

N/A

#### Essential References Not Discussed

N/A

#### Was Revised

`false`

#### Extra Scores

{}

---

### Review 3 — Reviewer_LcUb

- Review ID: `jILJLHuBRr`

#### Summary

The paper introduces the Fourier positional embedding (FoPE) based on Fourier series. The authors begin by analyzing the rotary positional embedding (RoPE) method in the frequency domain. Further analysis of the feed-forward network yields additional information that linear layers produce spectrum leakage (mix of frequencies) and the activation functions lead to spectrum distortion due to their harmonics generation. These two types of spectrum effects are causes for spectrum damage for context length generalization, hindering RoPE’s effectiveness. Then, the attention mechanism is shown to produce spectrum damage on low frequencies for *undertrained* components. Following this analysis, FoPE is proposed as a multi-frequency representation for each dimension based on Fourier Series, instead of a single frequency like RoPE. The proposed method also includes zeroing the low frequency components to control for undertrained frequencies. Next the authors compare FoPE against RoPE and ALiBi on pre-training, and against YARN extension on continual pre-training of OLMo, and against RoPE on fine-tuning SmolLM-1.7B. They measure perplexity and passkey retrieval to test the context length generalization of the resulting models. In most cases, FoPE shows better generalization when increasing the context length at test time. Further, ablation analysis is performed to test both the zeroing of frequency and without the Fourier series, and various analysis related to other aspects of the language model.

## update after rebuttal
Thanks to the authors for clarifying the various points.  After a thoughtful consideration, primarily given the additional comparisons, I've decided to update my score.

#### Claims And Evidence

The claims made in the submission are mostly supported. A formal definition of *undertrained* components of attention is missing, so it is unclear when does spectrum damage based on this definition matters. An empirical analysis of the potential Spectrum Damage on a well-trained model is lacking.

#### Methods And Evaluation Criteria

The derived methods seem sound. The selection of the low frequency threshold is not analyzed. What is the impact on the method if an incorrect or poor threshold selected? The evaluation criteria for the method seems correct.  The datasets could be selected to test generalization further (e.g., perplexity on SCROLLS).

#### Theoretical Claims

The derivations in the paper seem correct. Please, see comments above about undertrained components.

#### Experimental Designs Or Analyses

The proposed method is evaluated on a pre-training on C4 and Books datasets. The results show that ALiBi performs better than FoPE. In the text, the authors suggest that there is an “issue” with how ALiBi considers the linear declined attention, putting a considerable effort to learn the short-distance information. However, it is well-known that the distribution of mutual information between tokens behaves as a power-law distribution [1]. Namely, this is a property of the data, and the model exploits such prior knowledge (and not the other way around). In addition, pre-trained language models have better perplexity than the ones showed in Section 5.2, and the models are much bigger in size today (using more train tokens) than those used in the paper. 

The passkey retrieval results in section 5.2 show that the bigger the model, the better the retrieval results. If the 3 FoPE model sizes have the same embeddings and trained with the same context length, then why are the smaller models unable to generalize in the same way as the biggest model? Their curves decay much faster. Also, existing models can solve tasks beyond 8k context length today. It is unclear if they can generalize similarly as FoPE or even better. This suggests that there may be an issue when training the models or with the experiment itself.

Last, it is unclear what is the maximum sequence length for section 5.4? Also, what is the performance of the baseline model *before* fine-tuning?

Other positional embedding methods have not been compared. 

[1] “Critical Behavior in Physics and Probabilistic Formal Languages”, Lin and Tegmark, 2017

#### Relation To Broader Scientific Literature

The contribution of FoPE and proposes that context length generalization is of some value to the community. Current LLMs have context lengths that are very large (100k+ tokens), thus the significance is low. The theoretical analysis is well executed but limited (i.e., how do residual connections influence the frequencies? How does the LayerNorm or the specific activation functions?).

#### Questions For Authors

See questions in sections above.
* Can you learn the threshold value for the low frequencies?

#### Rating

`3`

#### Rebuttal

Thanks for the elaborate comments, we will address them according to their proposed order.

# Clarification for "undertrained components" and the impact of poor threshold
- **We have jointly defined the "undertrained components" and "floor frequency" in Fig 2, Sec 3.3 and Sec 4, using both visualization and formula.**
- Also, **these two concepts are well-known in the area of length generalization**, mentioned by many papers [1, 2].
- As for the impact of poor threshold, we add ablations on 60M OLMo trained with 512 context length (similar setting as Sec 5.2). **Our findings suggest that selecting a floor frequency no-less than $2\pi/L$ is necessary, but a higher threshold does not have a significant influence.**
|threshold (f=$2\pi/L$)|0|0.5f|0.75f|f|1.25f|1.5f|
|-|-|-|-|-|-|-|
|**ce loss on 8192 length**|6.86|6.54|5.97|5.84|5.86|5.85|

# Empirical analysis of the Spectrum Damage on well-trained models
- In Sec 5.4, our experiments are based on SmolLM-1.7B, a famous model well-trained by HuggingFace for academic purposes.
- In Sec 5.6, Fig 3 and Fig 7, we have a detailed empirical analysis of the influence of Spectrum Damage on LLaMA-2-7B.
- To further illustrate this phenomenon, we conduct another empirical analysis on LLaMA-2-7B (See https://anonymous.4open.science/r/FoPE-Supplementry-Material/Empirical_Analysis_on_Spectrum_Damage.pdf). We force every token only has one frequency component in the 1st layer. Then, we compare the spectrum of the 1st and 2nd layer after DFT on the attention map. **Many other frequency components appear in the 2nd layer, which demonstrates the happening of Spectrum Damage.**

# Advantage of FoPE over ALiBi and the well-known negative influence of linear declined attention
- **Our results in Sec 5.2 clearly show FoPE's advantages compared to ALiBi, it is unclear why the reviewer got the opposite conclusion.**
- We agree that "the mutual information between tokens behaves as a power-law distribution", and we appreciate this perspective.
- However, this property is only useful for short context modeling, but not for long-context modeling. For example, **the "linear declined attention" hinders the retrieval of long-distance information, thereby lacking the crucial ability for long-context modeling.** 
- **Not only do our results in Fig 1 demonstrate this phenomenon, but many well-known work also have the similar viewpoint [2, 3, 4]** that "linear declined attention" and ALiBi is fall short in long-context applications.

# Clarification for Sec 5.4 (the maximum sequence length and base model performance)
- We have provided the maximum sequence length in Sec 5.4 (line 328 and 373).
- The performance of SmolLM-1.7B-base is as follows, which do not influence our conclusions:
|length|gov_report|multi_news|trec|triviaqa|samsum|
|-|-|-|-|-|-|
|0-4k|9.10|5.06|42.0|68.53|17.97|
|4-8k|9.50|7.06|51.0|69.21|16.70|
|8k+|8.02|6.01|38.0|65.81|20.17|

# Advantage and clarification of "smaller models' performance decay faster than larger ones"
- Firstly, this phenomenon implies FoPE is a scalable method, which should be considered as an advantage.
- Secondly, it is not odd that larger models have better generalization.

# FoPE is valuable for many reasons, although current LLMs achieve 100k+ context length
- It is well-known that current Transformer-based LLMs achieve 100k+ context length using extrapolation as YARN [1]. **But in Fig 4, we have shown FoPE can also be used for extrapolation and has better performance than YARN.**
- With FoPE, models achieve better length generalization trained with a much shorter context length. **Thus, FoPE can significantly improve the training efficiency of models (the longer length, the much slower training).** This is valuable for saving time and money.

# Comparison with more baselines on more benchmarks
Please check our reply to Reviewer LYp1 for detailed results.

# Influence of residual connections, LayerNorm and specific activation functions
- We have modeled the influence of activation functions in Sec 3.2.
- Considering the residual connections and LayerNorm only deliver linear transform in frequency domain, we did not include their analysis in our paper. But for clarity, we will contain them in future revisions.

# Generalization ability of FoPE on non-training data
- **We have evaluated the OOD generalization in Sec 5.2.** (trained on Gutenberg Books and evaluated the ppl on C4)
- Our evaluations on Sec 5.4 include datasets from SCROLLS (See Sec 5.4).

# Clarification for Figure 6
The description is clearly presented just under Fig 6 in Sec 5.5.

---

[1] YaRN: Efficient Context Window Extension of Large Language Models. ICLR 2024.

[2] FIRE: Functional Interpolation for Relative Positions Improves Long Context Transformers. ICLR 2024.

[3] CLEX: Continuous Length Extrapolation for Large Language Models. ICLR 2024.

[4] Mesa-Extrapolation: A Weave Position Encoding Method for Enhanced Extrapolation in LLMs. NeurIPS 2024.

#### Other Strengths And Weaknesses

* Theory and derivations are valuable in the paper. 
* The experiments and results are not conclusive. It’s hard to really conclude the value of FoPE versus RoPE or other not-compared embedding methods.

#### Other Comments Or Suggestions

The descriptions in the text of the experiments related to Figure 6 look disconnected to the Figure itself.

#### Essential References Not Discussed

Not aware of such references.

#### Was Revised

`false`

#### Extra Scores

{}

---

### Review 4 — Reviewer_FbaF

- Review ID: `YhCxrdG5sx`

#### Summary

The paper analyzes how RoPE enables periodic attention patterns and then analyzes the limitation of RoPE in that regard.
The authors argue that the limitation arises from spectral damage prevalent when RoPE is used with typical DL architectures.
The authors propose FoPE, which is based on RoPE, but while RoPE treats each dimension as a single-frequency function, FoPE models each dimension as a Fourier Series, consisting of a dominant frequency component and several harmonic components.
Additionally, FoPE clips low frequencies because the authors argue these are undertrained.
In experiments, the authors show that FoPE maintains a better performance for increased context lengths.

#### Claims And Evidence

The primary claim—that FoPE improves length extrapolation—is well-supported by empirical analysis. The experiments effectively showcase FoPE’s advantages over RoPE, and the inclusion of comparisons with YARN strengthens the argument.

#### Methods And Evaluation Criteria

The evaluation methods and criteria appear appropriate. The experiments test models up to 1.2B parameters, a reasonable scale for assessing generalization. The comparison with YARN adds credibility.

#### Theoretical Claims

I briefly reviewed the theoretical claims presented in the main paper and did not identify any obvious issues.

#### Experimental Designs Or Analyses

The experimental design seems to be valid and sound. Models up to 1.2B parameters are tested, which is already a reasonable size to support generalization to practical settings. The comparison optionally includes YARN, which strengthens the argument.
As I am not deeply familiar with related approaches, I don't know if a comparison to additional methods would be appropriate.
A comparison to non-transformer architectures that provide favourable length extrapolation capabilities would be interesting in addition (e.g. see Figure~7 in [1]).
That said I have one concern regarding the hyperparameters D and sigma. The authors rightly conduct a sensitive analysis which shows that the parameters seem to be important. However, there is no clear guidance on how to select this parameter a-priori.

[1] xlstm: Extended long short-term memory, M Beck et al. Advances in Neural Information Processing Systems 37

#### Relation To Broader Scientific Literature

The paper is closely related to research on positional embeddings for transformers and the broader literature on length extrapolation. The study builds on RoPE, a widely used technique in modern deep learning models.

#### Questions For Authors

- How would you select the sigma and D hyperparameter for a new large-scale models where a hyperparameter search is not feasible because of the scale/cost of the model training? (The sensitivity of these parameters could affect the real-world usability of FoPE)
- Could clipping low frequencies have unintended consequences in certain applications?

#### Rating

`3`

#### Rebuttal

Thanks for the valuable comments from the reviewer, we want to deliver further explanations accordingly.
# Comparison with more baselines on more benchmarks
It is a nice suggestion to compare FoPE with more baseline methods, thus we supplement the following experiments:
- We validate the effectiveness of FoPE by evaluating it on 10+ additional benchmarks and comparing it with 3 more baselines. These supplementary experiments are conducted on OLMo-1.2B training on C4 datasets, with all experimental settings kept consistent with Sec 5.2 (line 293-329).

|methods|avg acc|basic_arithmetic|social_iqa|winogrande|openbook_qa|sciq|hellaswag|piqa|commonsense_qa|arc_easy|
|-|-|-|-|-|-|-|-|-|-|-|
|NoPE|42.14|25.67|43.71|51.86|29.80|76.70|41.83|68.83|31.61|51.40|
|ALiBi|42.93|24.97|42.53|53.12|31.40|77.70|43.06|69.42|**33.42**|**53.68**|
|KERPLE|43.22|25.03|43.81|**54.07**|32.40|**78.20**|43.65|69.64|32.92|52.46|
|FIRE|42.38|25.60|42.63|49.88|33.40|77.00|42.75|69.31|32.92|50.35|
|RoPE|42.98|24.60|43.45|51.54|**33.60**|77.10|43.36|**70.13**|33.01|52.98|
|FoPE|**43.37**|**26.17**|**44.12**|53.20|32.20|77.80|**43.83**|70.08|32.92|53.33|

|methods|avg acc|mmlu_stem|mmlu_social_sciences|mmlu_humanities|mmlu_other|
|-|-|-|-|-|-|
|NoPE|25.81|25.47|25.38|28.01|24.39|
|ALiBi|25.99|25.93|27.20|26.44|24.39|
|KERPLE|26.16|26.02|25.97|27.82|24.82|
|FIRE|26.04|26.34|**27.41**|26.25|24.16|
|RoPE|26.68|27.01|27.26|27.87|24.53|
|FoPE|**27.57**|**27.30**|27.31|**29.89**|**25.79**|

|methods|avg ce loss|natural_qs_open|trivia_qa_wiki|arc_easy|
|-|-|-|-|-|
|NoPE|1.4513|1.4334|1.6129|1.3077|
|ALiBi|1.4242|1.3879|1.6057|1.2789|
|KERPLE|1.4284|1.4149|1.5878|1.2825|
|FIRE|1.4580|1.4365|1.6258|1.3118|
|RoPE|1.4225|1.4114|1.5973|1.2588|
|FoPE|**1.3941**|**1.3818**|**1.5736**|**1.2272**|

- **While FoPE is primarily designed for length generalization and long-context tasks, it also performs comparably or even surpasses all baseline methods**. But on MMLU, all methods have weak performance, perhaps caused by the training data. In the future work, we may conduct more experiments on datasets containing math data.
- We also conduct evaluation similar as Sec 5.2 (See https://anonymous.4open.science/r/FoPE-Supplementry-Material/Supplementary_Main_Experiments.pdf). **FoPE still demonstrate the best accuracy on passkey retrieval  and the second-best ppl on C4 (only slightly behind ALiBi).**
- As time is limited, we have not conducted experiments on RWKV, xLSTM and Mamba. But it is really a nice suggestion to consider the length generalization capabilities of models with different architectures. We would like to cite these papers in related work in our future revisions.

# Clarification of the hyper-parameters' influence on FoPE
We agree with the reviewer that hyper-parameters optimization is crucial for the large-scale training, but we would like to make some clarifications:
1.**In a wide range of parameters we tested, FoPE consistently demonstrates a significantly better performance than RoPE**. 
2. **The performance of FoPE is not highly sensitive to hyper-parameters** (see ablation in Fig 6 and Sec 5.5), although an elaborate hyper-parameter selection of FoPE may lead to a better performance.

Thus, **even without a careful selection of hyper-parameters, FoPE is still competent for replacing RoPE in Transformers**. 

Additionally, we emphasize that hyper-parameter tuning is an inevitable procedure before pre-training nowadays, many other hyper-parameters also need to be selected (i.e. hidden_dim, num_heads, ...). As for FoPE, the ablation studies in Fig 6 and Sec 5.5 suggested that:
- The optimal $\sigma$ for FoPE increases as hidden_dim and num_layers grow, which is expected, as larger models suffer more from spectral damage.
- The best $D$ tends to be slightly bigger than the head_dim of each attention head. This is because too few frequency components cannot represent the spectrum damage well, , while too many exceed the model’s representational capacity.

But it is a nice suggestion to have a better selection strategy, we would like to continually research this problem.

# Influence of clipping low frequencies
It is an insightful suggestion to consider the negative influence of clipping low frequencies, we want to make some clarification below:
- "Clipping low frequencies" is quite similar to the "interpolation" used by many extrapolation methods like YARN[1]. As this operation is mainstream solution for LLMs' length generalization, we suppose "Clipping low frequencies" does not have significant negative influence in most tasks.
- **Also, FoPE has a consistent performance on diverse benchmarks, .** Thus, the answer to this problem is still unclear. But we would like to continually consider this issue.

---

[1] YaRN: Efficient Context Window Extension of Large Language Models. ICLR 2024.

#### Other Strengths And Weaknesses

Strengths:
- Clear motivation with according analysis of the approach (spectrum damage) 
- Evalution includes comparision to length extrapolation technique (yarn)

Weakness:
- No guidance for hyperparameter selection (D and sigma) although the methods seem to be sensitive to these parameters.
- No comparison to non-transformer LLM architectures that provide more favorable length generalization properties like RWKV, xLSTM or Mamba (e.g. see Figure~7 in [1])

[1] xlstm: Extended long short-term memory, M Beck et al. Advances in Neural Information Processing Systems 37

#### Other Comments Or Suggestions

-

#### Essential References Not Discussed

I am not aware of any essential references that are missing.

#### Was Revised

`false`

#### Extra Scores

{}

---

## Meta-Review / Decision Comment

This paper uses insights from discrete signal processing to analyze how RoPE enables periodic attention patterns and its limitations in that regard, identified as a form of "spectral damage". The paper proposes FoPE as a solution, which is differs from RoPE in which it models each dimension not as a single frequency but as a Fourier Series, consisting of a dominant frequency component and several harmonic components. Experiments show that FoPE maintains a better performance for increased context lengths.

This paper provides a valuable contribution which analyzes weaknesses of current models and uses insights from discrete signal processing to propose a solution. Reviewers point out as weaknesses concerns about the signal-based explanation and the computational cost (both of which were clarified in the rebuttal), as well as somewhat inconclusive experiments and results, which does not completely demonstrate the value of FoPE versus RoPE or other embedding methods. The authors addresses most of the concerns. I am leaning towards acceptance.


---

# LongRoPE2: Near-Lossless LLM Context Window Scaling — OpenReview 审稿全文归档

- Venue: **ICML 2025 poster**
- OpenReview forum: [https://openreview.net/forum?id=jwMjzGpzi4](https://openreview.net/forum?id=jwMjzGpzi4)
- Official paper page: [https://proceedings.mlr.press/v267/shang25a.html](https://proceedings.mlr.press/v267/shang25a.html)
- Reviewer handles are public OpenReview pseudonyms; no identity resolution is attempted.
- Source: ReviewArena public release (OpenReview-sourced). ICML 2025 uses the venue’s 1–5 Overall Recommendation scale; do not compare these numbers directly with ICLR’s 1–10 scale.

## Review Inventory (3 Official Reviews)

- Final decision: **Accept (poster)**

## Official Reviews

### Review 1 — Reviewer_Ebnv

- Review ID: `KpynhegvSK`

#### Summary

Maintaining the performance on both long and short benchmarks are a critical challenge for existing long context extension methods. LongRoPE2 is a new approach that extends the effective context window of pre-trained large language models to the target length, while preserving the performance on the original shorter context window.

#### Claims And Evidence

Claims: LongRoPE2 extends context windows to 128k while retaining >97% short-context performance. The key contributions are (1) higher RoPE dimensions are undertrained, (2) evolutionary search for rescaling factors guided by needle-driven perplexity, (3) mixed training with original/rescaled RoPE.

Evidence: Achieves strong results on RULER, and real-world benchmarks (LOFT, LongBench). Outperforms YaRN, NTK, and LongRoPE with much fewer tokens.

#### Methods And Evaluation Criteria

Methods: Evolutionary search for critical dimensions and scaling factors, mixed training (original RoPE for short contexts, rescaled RoPE for long).

Evaluation: Benchmarked on RULER, Needle-in-a-Haystack (retrieval), LOFT/InfiniteBench (real-world), and MMLU/GSM8K (short-context).

#### Theoretical Claims

Challenges prior RoPE OOD theory: insufficient training in higher dimensions extends empirical periods, requiring larger scaling factors than theoretical bounds. 

To be honest I do not carefully check the correctness of all theoretical claims of this paper.

#### Experimental Designs Or Analyses

Ablations confirm needle-PPL’s superiority over standard PPL and mixed training’s necessity.
Adjusted baselines (YaRN-red/NTK-red) show improved but suboptimal performance.

#### Relation To Broader Scientific Literature

Builds on RoPE rescaling (NTK, YaRN) and evolutionary optimization. Different from RAG/agent-based methods, positioning LongRoPE2 as complementary methods.

#### Questions For Authors

1. How does evolutionary search scale to million-token contexts?
2. Does mixed training cause interference between short/long contexts?

#### Rating

`4`

#### Rebuttal

**Response**: Thank you for your valuable feedback and for recognizing the strengths of our work. We appreciate the opportunity to address your concerns.

1) **Affordable evolutionary search computational cost**: we acknowledge that evolutionary search introduces additional costs. To further clarify its feasibility, we conduct additional experiments to evaluate the search cost when scaling from 128k (current context window length) to 1024k. Using vllm0.7.3 as the inference engine and running on an 8*A100(80GB) server setup, we measured the total search time. As shown in the table below, even when scaling to 1M tokens, the total search time remains manageable at 240 hours (10 days). Moreover, this is a one-time *offline* process, and the search time can be linearly reduced by increasing the number of GPUs, due to the nature of evolutionary search. Therefore, it is practical for LLM pretraining teams. 

||128k|512k| 1024k|
|:--:   |:--:   |:--:   |:--:   |
|total search time on 8*80GB a100| 7.5h | 68h | 240h |
   
2) **KV cache recalculation occurs only in specific cases and has minimal overhead**: We acknowledge that KV cache recomputation is required when transitioning from the short context window (using the short factor, i.e., original RoPE) to the long context window (using the long factor). However, this recomputation does not occur in every inference. It happens only when the input length is within the short context window, but the total length (input+generated tokens) exceeds it **for the first time**. After this one-time recomputation, no further recomputation is needed for the rest of the generation. 

In most general inference scenarios, this situation is relatively uncommon, as prompts and completions typically either remain within the short context window or start in the long context mode from the beginning. To quantify the cost, we measured KV recomputation time on a  4x80GB A100 GPU (with vllm 0.7.3) for Phi-3-mini and LLaMA3-8B, comparing it against normal decoding time:

||prefill (kv recompute)|decode-output 512| decode-output 1k| decodetime-output 2k| decodetime-output 4k| decodetime-output 8k | decodetime 16k|
|:--:   |:--:   |:--:   |:--:   |:--:   |:--:   |:--:   |:--:   |
|Phi3-mini (prefill 2k)| 124.1ms| 7.63ms (**16.2**)| 7.66ms (**16.2**)| 7.71ms (**16.1**)| 7.78ms (**15.9**)| 14.29ms (**8.7**)| 23.3ms (**5.3**) |
|LLaMA3-8B (prefill 8k)|613.9ms | 24.11ms (**25.5**)| 24.22ms (**25.3**)| 24.05ms (**25.5**)| 24.18ms (**25.4**) | 23.5ms (**26.1**) | 23.58ms (**26.0**) |

The numbers in () indicate the amount of decoded tokens corresponding to the time spent on KV cache recomputation. These results indicate that the additional recomputation cost is equivalent to generating only ~15 (phi3-mini) and ~25(llama3-8b)  tokens, which is negligible in the context of long-context generation.


>Q2: Does mixed training cause interference between short/long contexts?

**Response**: Thank you for your insightful question. While it’s true that mixed context window training applies two RoPE scaling factors simultaneously during mid-training, which could introduce interference, our empirical results suggest that this "interference" plays a constructive role and hence does not degrade performance. In fact, it not only recovers short-context performance but also enhances long-context performance. To better illustrate this, we refer to Table 7 from our original paper.

||MMLU-(short)|MMLUPro-(short)|GSM8K-(short)| RULER-4k| RULER-8k| RULER-16k | RULER-32k|RULER-64k|RULER-128k|
|:--:   |:--:   |:--:   |:--:   |:--:   |:--:   |:--:   |:--:   |:--:   |:--:   |
|Phi3-mini (**with mixed context window training**)| **70.07**|**40.30**|**73.62**|90.41|**86.87**|**83.33**|**76.51**|**65.37**|**58.81**|
|Phi3-mini (no mixed context window training)|66.56|34.86|64.67|**90.55**|85.77|81.08|73.31|63.75|56.22|
||||||||||
|LLaMA3-8B (**with mixed context window training**)|**65.01**|**34.61**|**50.80**|94.61|**93.68**|**92.31**|**90.49**|**85.62**|**82.03**|
|LLaMA3-8B (no mixed context window training)| 64.57| 33.83| 48.37| **94.67**|93.15|91.24|89.38|83.53|80.18|

A possible explanation for this surprising improvement is that the so-called “interference” actually plays a constructive role in learning. Specifically, the short-context window helps preserve position modeling for non-interpolated positions (e.g., LLaMA3’s native positions 0, 1, 2, ..., 8191), while the long-context window primarily facilitates the adaptation for newly interpolated positions (e.g., LLaMA3’s new positions like 1/16, 2/16, ..., 17/16). This training strategy effectively constrains the model’s adaptation to interpolated positions while maintaining consistency with the original position modeling - a concept similar to the KL divergence constraint in PPO, which prevents large deviation from the original policy model.

We appreciate this insightful question, which has prompted further reflection and discussions.

#### Other Strengths And Weaknesses

Strengths: Efficient scaling (10B tokens), minimal short-context degradation.

Weaknesses: Evolutionary search computational cost; inference requires KV cache recalculation.

#### Other Comments Or Suggestions

N/A

#### Essential References Not Discussed

No

#### Was Revised

`false`

#### Extra Scores

{}

---

### Review 2 — Reviewer_gztC

- Review ID: `YBGS9h8WSG`

#### Summary

This paper proposed LongRoPE2, a RoPE scaling method to extend the context window of LLMs. The primary extension compared to LongRoPE1 is that LongRoPE2 utilizes a needle-based search rather than perplexity-based one for various rope dimension scaling. The experimental results demonstrate the superior performance of LongRoPE2 compared to other RoPE scaling methods.

#### Claims And Evidence

1. The most overclaim is  "LongRoPE2-extended LLaMA3-8B-128k surpasses Meta’s LLaMA3.1-8B-128k in long-context performance with 80x fewer training tokens". This claim is supported by the RULER results in Fig 1. However, LongRoPE adopts a needle-based search for RoPE scaling, which may (over)fit the synthetic tasks in RULER benchmark, hence achieving better results. For other general tasks such as En. MC in InfiniteBench, **LongRoPE2-LLaMA3-8B achieved a score of 46.72 but LLaMA-3.1-8B achieved 65.1**. This means LongRoPE2-LLaMA3-8B may still have a large gap to LLaMA-3.1-8B involving far more training tokens. Note that it's not necessary for LongRoPE2-LLaMA3-8B to surpass LLaMA-3.1-8B, but should fix the claim for clearness.

2.  The mixed context window training is adopted in [1] and LLaMA-3.1 (as well as a common practise in long-context LLM community) to maintain the short-context performance, but the authors claims they propose such a "novel" strategy.

[1]LongAlign: A Recipe for Long Context Alignment of Large Language Models

#### Methods And Evaluation Criteria

The evaluation benchmarks are popular in the long-context understanding field.

#### Theoretical Claims

There are no proofs for theoretical claims.

#### Experimental Designs Or Analyses

I have gone through the ablation studies which have demonstrated the effectiveness of LongRoPE2's designs.

#### Relation To Broader Scientific Literature

This paper is a direct extension of LongRoPE[1].

[1] LongRoPE: Extending LLM Context Window Beyond 2 Million Tokens

#### Questions For Authors

1. Do you think it's better to list the results of LLaMA-3.1-8B in the main table? Since you claimed the proposed method on 10B tokens can surpass LLaMA-3.1-8B's continual training of 800B tokens.

2. What are the criteria for you to choose the evaluation tasks? It seems there are some challenging tasks such as En.QA, En. Sum, etc. in the InfiniteBench and some other tasks of various categories in LongBench. The selected tasks seem to be irregular.

3. The claim regarding the intuition of insufficient high-frequencies RoPE training and the mix context window training should be better to fix.

#### Rating

`3`

#### Rebuttal

>Q1: Clarification on LLaMA3.1-8B long-context evaluation numbers, and the "overclaim" comments

**Response**: We appreciate your feedback and would like to clarify the following points:

1. **65.1 is the En.MC score of the instruct version, not LLaMA3.1-8B.**:  As noted in Table 2 of the LLaMA3.1 tech report, the 65.1 score is for the instruct-tuned version (a detail that can be overlooked). Compared to the fair baseline, LLaMA3.1-8B, our model achieves a higher score (**46.72** vs. **45.85**) on En. MC. Moreover, our model consistently outperforms LLaMA3.1-8B across several long-context benchmarks. Here are additional results:

>InfiniteBench and LongBench:

||avg.|En.MC|En.Sum| KV retrieval | TriviaQA | TREC | LCC | RepoBench-P|
|:--:   |:--:   |:--:   |:--:   |:--:   |:--:   |:--:   |:--:   |:--:   |
|LLaMA3.1-8B|54.28| 45.85| 15.27|16.20| **91.13**| 73.50| 70.24| **67.83**|
|LongRoPE2-LLaMA3-8B|**65.20**|**46.72**| **16.20**|**88.0** |**91.13**| **76.50**| **70.47**| 67.39 |

>LOFT:

||avg.| ArguAna | FEVER | HotPotQA | MS MARCO | NQ | Quora | SciFact|
| :--:  | :--:  | :--:   | :--:   |:--:   |:--:   |:--:   |:--:   |:--:   |
|LLaMA3.1-8B| 53.14|19.0| 90.0| 12.0| 69.0| 78.0| 61.0|43.0|
|LongRoPE2-LLaMA3-8B| **74.28**|**28.0** | **96.0** | **70.0** | **80.0**| **94.0**| **79.0** | **73.0**|


2. **Our RoPE scaling method is designed to improve broad long-context capabilities, not to optimize for any specific benchmark like RULER.** The use of needle data for search is **not** designed to fit RULER but to **better control long-range token dependency distances** in long documents.  E.g., We used only the simplest number needle synthesis method. Our extensive experiments proved the superiority over other methods (e.g., NTK, YaRN) across diverse benchmarks.

>Q2: What are the criteria for you to choose the evaluation tasks? 

**Response**: Our selection follows two key principles:
1) **Effectiveness for evaluating a pre-trained LLM rather than a chat LLM**. Since our method extends a pre-trained LLM without post-training, we prioritize tasks aligned with this setup: (i) completion-based tasks, such as few-shot learning and code completion and En.MC in InfiniteBench. (ii) QA tasks with few-shot examples, such as various text-retrieval QA tasks in LOFT. 

2) **Comprehensive long-context evaluation**. To evaluate multiple aspects of long-context performance, we include tasks covering RULER, needle-in-a-haystack retrieval, real-world text QA, high-difficulty KV retrieval, multi-choice QA, few-shot learning, and code completion, as detailed in our evaluation section. We believe this selection fairly reflects the strengths of our method and provides a well-rounded assessment.

*Additional results on chat-based sub-tasks*.  For your reference, we provide additional results on chat-based LongBench tasks. As shown below, We achieve the highest average score, even surpassing LLaMA3.1-8B.

||Avg.|narrativeqa| Qasper | multifiledQA | hotpotqa | 2wikimqa | musique | gov_report | qmsum| samsum|
| :--: | :--:  | :--:  | :--:   | :--:   |:--:   |:--:   |:--:   |:--:   |:--:   |:--:   |
|LLaMA3.1-8B|22.60| 20.90|12.50|32.72|11.95| 13.98| **8.62**| 29.95|**25.53**|**47.23**| 
|NTK-LLaMA3-8B|20.32|21.14|11.93|29.02|11.91|14.71|7.81|21.50|22.09|42.70|
|LongRoPE2-LLaMA3-8B|**24.31**|**21.79**|**18.13**|**36.25**|**13.85**|**19.42**|8.03|**30.12**|25.41|45.80|

>Q3: The clarifications on the main contributions and claims:

**Response**: We are grateful for your questions and would like to clarify that our main contribution is not the discovery of insufficient training in high-frequency RoPE, but rather the introduction of **a new RoPE OOD hypothesis**. This hypothesis explains why existing RoPE rescaling methods, such as NTK and YaRN, often result in suboptimal long-context performance. This contribution has been acknowledged by the other two reviewers.

Regarding the mixed context window training, we would like to emphasize that the key difference between our approach and those used in LLaMA3.1 and LongAlign is the use of two distinct RoPE scaling factors: a short factor for short contexts and a long factor for long contexts. This **dual-factor** approach is essential in significantly recovering short-context performance, which we have shown through extensive experiments.  

Here, we perform an additional comparison with LLaMA-3.1’s mixed training. As shown below, we significantly improve short-text performance.

||(short)-MMLU|(short)-MMLU pro| (short)-GSM8k | Ruler-128k| 
| :--: | :--:  | :--:  | :--:   | :--:   |
|LLaMA3-8B (**Our mixed context windows training**)|**65.01**|**34.61**|**50.80**|**82.03**|
|LLaMA3-8B (mixed training in LLaMA3.1)| 64.18|32.95 |46.25 |71.83 | 

We hope these responses address your concerns and clarify any confusion, and we will incorporate them in the revisions. Thank you again for your valuable feedback and suggestions, and we kindly ask you to consider re-evaluating our work.

#### Other Strengths And Weaknesses

Strength:
1. Generally the experiments covering long and short contexts are well-designed and can demonstrate the effectiveness of the proposed method.
2. The intuition regarding the insufficient training of high-frequencies RoPE makes sense.
3. I believe the needle-driven PPL search is a better choice for LongRoPE as the pure PPL on normal documents may be orthogonal to long-context performance.

Weaknesses:
1. I feel most designs in the work have been proposed/adopted in previous works. For example, the intuition that high-frequencies RoPE  may be insufficiently trained has been introduced in a popular blog (https://spaces.ac.cn/archives/9706) regarding RoPE scaling. The mixed context window training is adopted in [1] and LLaMA-3.1 to maintain the short-context performance. It may be improper to regard these points as this work's contributions and should give some credit to the related works.

[1]LongAlign: A Recipe for Long Context Alignment of Large Language Models

#### Other Comments Or Suggestions

The related work section is better placed in the main body of the paper to make it self-contained. (This is a suggestion and wouldn't affect my rating.)

#### Essential References Not Discussed

This paper considers a mix context window training strategy as one of its primary contributions. However, such strategies are widely-used in some previous works such as LongAlign[1], which is missing in references.

[1]LongAlign: A Recipe for Long Context Alignment of Large Language Models

#### Was Revised

`false`

#### Extra Scores

{}

---

### Review 3 — Reviewer_ak4t

- Review ID: `GscOWQBEYB`

#### Summary

This paper mainly introduces LongRoPE2, aiming to achieve an effective long context window while preserving short-context performance by context extension. Based on LongRoPE, LongRoPE2 introduces a new needle-PPL guided evolutionary search method for settling the rescaling factors, and proves it to be more effective than the naive PPL-guided one by experiments. For retaining preformance on short contexts, LongRoPE2 proposes a novel mixed context window training method. Compared to YaRN, NTK and LongRoPE,  LongRoPE2 achieves better performance on long contexts while retaining over 98.5% of short-context performance.

---

## update after rebuttal

Thanks to the authors for their response, and I acknowledge that LongRoPE2 is a strong and valuable work. However, there are still some unclear aspects in the paper, such as the lack of a detailed explanation regarding how Figure 3(a) was derived. Due to these unresolved concerns, I have decided not to adjust our initial score. In our view, a score of 3 (borderline accept) remains reasonable and justified.

#### Claims And Evidence

yes

#### Methods And Evaluation Criteria

YES

#### Theoretical Claims

yes
This paper proposes a New RoPE OOD Hypothesis that the empirical RoPE periods in higher dimensions are longer than theoretical values, limiting current methods to fully address RoPE OOD. This implies that the actual optimal rescaling factors may be greater than the theoretical one. Then, by applying needle-PPL-guided search, LongRoPE2 does get rescaling factors larger than the theoretical one, and performs better, which from this point of view can test this hypothesis.

#### Experimental Designs Or Analyses

Yes.
In 4.2, this paper presents reults on RULER, NIAH, LOFT of LongRoPE2-extened, comparing with other SOTA RoPE rescaling methods(YaRN, NTK, LongRoPE) and shows the effectiveness.
In 4.3,  to validate the effectiveness of real critical dimension, one experiment applies d_{rcd} to YaRN and NTK, which is also get improved.  To validate the effectiveness of needle-PPL guided search, one experiment compares it with the naïve one with the same training process on the same test dataset. Finally, the effectiveness of mixed context window training is also validated here.

#### Relation To Broader Scientific Literature

Some scaling methods before LongRoPE(PI, YarN, NTK) ignore the actual errors caused by different parameters of models after training. This issue was preliminarily solved in LongRoPE. LongRoPE2 is based on LongRoPE, and more effective.

#### Questions For Authors

No

#### Rating

`3`

#### Rebuttal

Thank you for your thoughtful review and for recognizing our contributions. We greatly appreciate your acknowledgment of our New RoPE OOD Hypothesis and the role of needle-PPL-guided search in validating this hypothesis through empirical results. We are also glad that you found our extensive experiments in Sections 4.2 and 4.3 valuable in demonstrating the effectiveness of LongRoPE2 and our key design choices, such as the real critical dimension and mixed context window training. Please let us know if there are any specific aspects where we can provide more details. Thank you again for your time and constructive evaluation!

#### Other Strengths And Weaknesses

This paper is clearly demonstrated. The most enlightening contribution may be the New RoPE OOD hypothesis (in 3.1), as it gives directions for optimization of other methods  not limited to this paper.

#### Other Comments Or Suggestions

There is one typo on page 8, subtitle: need-PPL  should be needle-PPL

#### Essential References Not Discussed

No

#### Was Revised

`false`

#### Extra Scores

{}

---

## Meta-Review / Decision Comment

The study presents LongRoPE2, a method designed to significantly extend the context window of pre-trained large language models, such as LLaMA3-8B and Phi3-mini-3.8B, without compromising the performance on shorter context windows. By addressing the out-of-distribution (OOD) issues with existing methods, LongRoPE2 employs a hypothesis-driven evolutionary search algorithm to optimize the RoPE (Random Projections) dimensions, and introduces a mixed training approach. combines original and rescaled RoPE to ensure high performance across various benchmarks. The paper highlights extensive experiments that validate the effectiveness of LongRoPE2, demonstrating its ability to extend the context window to 128k while retaining over 98.5% of the short-context performance using only 10B tokens, a substantial improvement over previous approaches.
Reviewers raised concerns including affordable evolutionary search computational cost, KV cache recalculation, needle-PPL and mixed training necessity，evaluation tasks, effectiveness of the method, etc.  The authors addressed these concerns with clarifies and additional experimental results.
The method has the potential to significantly impact the field of large language models by enabling longer context windows, which is crucial for many real-world applications of LLMs.


---

# LieRE: Lie Rotational Positional Encodings — OpenReview 审稿全文归档

- Venue: **ICML 2025 poster**
- OpenReview forum: [https://openreview.net/forum?id=yMJAYbGcCc](https://openreview.net/forum?id=yMJAYbGcCc)
- Official paper page: [https://proceedings.mlr.press/v267/ostmeier25a.html](https://proceedings.mlr.press/v267/ostmeier25a.html)
- Reviewer handles are public OpenReview pseudonyms; no identity resolution is attempted.
- Source: ReviewArena public release (OpenReview-sourced). ICML 2025 uses the venue’s 1–5 Overall Recommendation scale; do not compare these numbers directly with ICLR’s 1–10 scale.

## Review Inventory (4 Official Reviews)

- Final decision: **Accept (poster)**

## Official Reviews

### Review 1 — Reviewer_SwFq

- Review ID: `9WYobYmMue`

#### Summary

The paper introduces a positional embedding encoding based on Lie Groups. The idea of the paper is to parameterize the positional embeddings using skew symmetric matrices. The authors show the benefit of the proposed method in terms of generalization, data efficiency and compute needed.

Overall, the idea is novel and interesting. The authors have empirical evidence that validates the quality of their work. However, the paper quality needs to be improved, both in terms of presentation as well as writing.

#### Claims And Evidence

The paper claims are well supported by the experiments.

#### Methods And Evaluation Criteria

The proposed method and evaluation criteria make sense.

#### Theoretical Claims

There are not theoretical claims.

#### Experimental Designs Or Analyses

Yes.

#### Relation To Broader Scientific Literature

The paper expands the equivariant work in terms of positional embeddings for Lie groups.

#### Questions For Authors

The authors do not include any equivariant transformer architecture and only compare with 2 other embedding works. Why are the authors not comparing themselves with any architecture of the ones mentioned before. 

I believe that the authors need to compare their results with some equivariant / quasi equivariant work of the ones mentioned before.

#### Rating

`3`

#### Rebuttal

Thank you for the thoughtful and thorough review and writing feedback which has helped strengthen the paper. We have addressed the typos and writing style in the revision based on your comments.

**Equivariant work comparison**: We are excited about the equivariant line of work! We believe it is key to learning sample efficient representations for domains with extensive symmetry. We will include a new section of the related work that relates LieRE to the suggested works on equivariance and other work on lie groups in ML.

As a complimentary architectural change, equivariance transformers do not directly compete with position encodings such as LieRE. Position encodings such as LieRE, RoPE-Mixed and absolute position encodings are minimally invasive modifications to the base transformer architecture, making them compatible with many different architectures. In the case of LieRE, this enables things like finetuning existing LLM weights to become multimodal models capable of handling high dimensional data. 

Combining them would be an exciting area of research. One of the observations present in both this work and the most closely related work [1] was that, sometimes, translation-invariance is helpful, but in other cases having access to the reference coordinate system can help performance. This is supported by the fact that adding absolute position encodings to RoPE-Mixed, and LieRE performs best in the regime where the attention patterns are not necessarily constrained to be translation invariant (recall that the attention patterns are translation invariant only when the block size is constrained to two). Fortunately, this is compatible with many of the designs in the equivariant line of work.

The natural question is how do we let models benefit from both the sample efficiency of equivariance and the fact that sometimes the absolute coordinate system does contain useful information. Combining these methods in the right way is an exciting future direction but has enough technical complexity that it is hard to incorporate into the current paper without losing focus and diluting individual learnings around positional encodings. The contributions in this paper are easiest to understand when comparing to earlier works that modify the same aspect of the transformer architecture. We hope you agree that the revisions to the paper provide a stronger connection to equivariant line of work and an extended related work accelerates future research in the area.

**Equivariance Related work**:A related branch of work encoding problem structure focuses on equivariance. We say that a model T is equivariant with respect to $f$ if T(f(x)) = g(T(x))$ for some $g$ [8]. Where with relative position encoding we often want to be able to encode translation invariance, equivariance provides a more general framework. Equivariance has been applied to improve performance on problems with a wide array of structures, ranging from rotation-invariance [10,13,14], 3D reference frame-invariance [9,12] and many others. The subset of these works that focus on generating equivariant token embeddings for transformers can be combined directly with LieRE or another rotation-based position encoding.

**Lie Groups in Machine Learning**:Lie groups have also had extensive use in machine learning. The range of works is diverse, ranging from algebraic signal processing [15], automated discovery of symmetries [16] to state estimation [18]. Furthermore [17] provides a friendly introduction to differential geometry and lie groups that may be of interest to the reader.

We thank the reviewer for highlighting this connection and helping us improve the paper.

References are replaced with links to respect character limits.
[1] https://arxiv.org/abs/2403.13298

[2] https://arxiv.org/abs/2403.00522

[3] https://arxiv.org/abs/2404.02905

[4] https://arxiv.org/abs/2212.09748

[5] https://arxiv.org/abs/2112.10752

[6] https://arxiv.org/abs/2411.04097

[7] https://arxiv.org/abs/2406.15955

[8] https://arxiv.org/abs/1901.11399

[9] https://arxiv.org/abs/2206.11990

[10] https://arxiv.org/abs/1709.01889

[11]https://proceedings.neurips.cc/paper_files/paper/2021/file/2a79ea27c279e471f4d180b08d62b00a-Paper.pdf

[12] https://arxiv.org/abs/2006.10503 

[13] https://arxiv.org/abs/1711.06721

[14] https://arxiv.org/abs/1612.04642

[15] https://arxiv.org/abs/2305.04431 

[16] https://arxiv.org/abs/2301.05638 

[17] https://link.springer.com/book/10.1007/978-3-030-46040-2 

[18] https://arxiv.org/abs/1903.02958 

[19] https://arxiv.org/abs/1912.12180

[20] https://arxiv.org/abs/2103.03206

[21] https://arxiv.org/abs/2103.14030 

[22] https://arxiv.org/abs/2401.10166

#### Other Strengths And Weaknesses

Comments:

C1  - Use a definition for the property of equations (1) and (2). For example, for eq 2 use
Comm(U,V).

C2 - Page 4, line 216 - Algorithm should be capitalized. All the subsequent calls to the word Algorithm should also be capitalized. 

C3 - Page 5, a lot of blank space. I believe it would improve the quality of presentation to fix this. 

C4 - The text under Figure 1 is confusing and needs a better structure. Some quantities are define but never used. 

C5 - In the experiments, a piece of the text was removed, making the sentence incomplete:

“We train the models on 800,000 examples and observe that they generally converge after the first 400,000. The only exception to this is absolute position encodings, where we have variants trained on 800,000 and”

There is also a missing reference:
“Please refer to the appendix for attention map examples, Figure 13 and Figure ??.”

C6 - In page 15, the text covers the page number.

#### Other Comments Or Suggestions

See before.

#### Essential References Not Discussed

The paper overlooks several works in equivariant transformers and neural networks:

- Equivariant transformer networks, Tai et al
- Equivariant Neural Functional Networks for Transformers, Tran et al
- Equiformer: Equivariant graph attention transformer for 3d atomistic graphs, Liao et al
- Polar transformer networks, Esteves et al
- Efficient equivariant network, He et al, 
- Se (3)-transformers: 3d roto-translation equivariant attention networks, Fuchs et al
- Learning so (3) equivariant representations with spherical cnns, Esteves et al

Also a lot of work in equivariance:
- Harmonic networks: Deep translation and rotation equivariance, Worral et al 

Even in lie groups ML, there are plenty of references missing:
- Lie group algebra convolutional filters, Kumar et al.
- Deep learning symmetries and their Lie groups, algebras, and subalgebras from first principles, Forestano, et al
- Differential geometry and lie groups, Gallier et al
- Reparameterizing distributions on lie groups, Falorsi et al

#### Was Revised

`false`

#### Extra Scores

{'ethical_review_concerns': 'None.'}

---

### Review 2 — Reviewer_udrb

- Review ID: `J3MlWaQAUb`

#### Summary

The authors mainly proposed a new positional encoding method called Lie, to replace the previous wildly used RoPE. It is used to improve the spatial relationship representation, especially in 2D and 3D images. Extensive experiments are conducted on classification tasks, and with the proposed PE, the accuracy values are all improved by a significant margin.

## update after rebuttal
I carefully read the authors' rebuttal, and thanks so much for the responses.
The authors also agree that it currently lacks evidences on image generation tasks, and some of the experiments are limited in design and scope. There issues are not fully resolved actually, and the authors did not clarify how to address them in the final version. The authors mainly used these experiment to "inspire" other future works, which I feel not very informative. 
However, the generalization design of RoPE to Lie group itself is interesting. 
So better AC can make the final decision to balance these factors.

#### Claims And Evidence

The evidences are mostly well supporting the claims. The proposed LiePE greatly improves the transformer-based classification model by a large margin. Figure 9 also shows great generalization capability of the positional encoding to higher resolution. Experiments also show that the compute increment is not significant.

#### Methods And Evaluation Criteria

The idea and the theory behind the proposal is elegant and interesting. Using the exponential of skew-symmetric matrix to generalize RoPE  is intuitive and smart, making the position encoding fully learnable and more expressive. Experiments on classification is a simple yet effective say of validating the idea, and the baseline comparison is clear and fair. 

Given now RoPE is more verified in image generation task, it will be better to show the potential of LiePE on image generation task.

#### Theoretical Claims

Not applied to this paper.

#### Experimental Designs Or Analyses

The main concerns for the experimental analysis is, the newly proposed PE is only used for image classification task and lower-res images. It is not sufficient to prove the effectiveness and expressiveness of the new PE for very long-context modeling. The image understanding tasks on the synthetic data is also very limited to prove its effectiveness. The patch shuffle experiments are interesting, but with random patch shuffle, it's not that meaningful to compare the dropping rate when the accuracy from different approaches are similarly low.

#### Relation To Broader Scientific Literature

The proposed PE is supposed to be a very general approach and a plug-and-play components for all transformer-based models. The idea has its merits and it has great potential to be generalized to high-resolution image generation task. However, the experiments in this paper cannot well support the claims and might not bring significant impacts in the literature.

#### Questions For Authors

See above.

#### Rating

`3`

#### Rebuttal

Dear Reviewer udrb,

We appreciate your recognition of our work's theoretical merits and experimental contributions. We have carefully considered your feedback and would like to address each point.

**Long Context for 1D**: LieRE is primarily focused on inputs with dimensionality greater than one. In fact, there is the [non-obvious] property that, in the 1D setting, LieRE has equivalent representational ability to RoPE with learnable frequencies. This is not the case in higher dimensions. We include a proof of this fact at the end of the response, and will include it in the paper as it has been requested by other readers. 

**Long Context for 2D and 3D**: This implies that the natural equivalent of long-context modeling is evaluating the model at resolutions higher than it was trained at. This is the focus of section 5.6. (Multi-Resolution Classification). We evaluate with up to four times as many inference tokens than during pretraining and finetuning (Figure 9).
Higher Resolution image generation: We agree that high resolution image generation is an exciting application to benchmark position encodings. The long-context image classification examples were motivated in part to create apples to apples experiments when compared to prior works [1, 2], as these are more focused on image classification. We are very supportive of future work in that application. This application is especially exciting in light of recent autoregressive image generation techniques such as VAR [3] that scale more predictably, enabling smaller scale experiments. It would be particularly interesting to see how position encodings could influence global consistency.

**Patch shuffling**: We hope to clarify that the intent of these experiments is to provide insight to the mechanics of why the methods perform differently rather than to show one method is better than the other. Patch shuffling allows us to see whether the model is actually using positional information. We agree with you that it does not identify which method is best at the ultimate applications. For that we depend on the other experiments. The focus of the experiment is to help the reader build an intuition of what is going on with various position encodings. 
Each method compared is distinct in both how and what kind of positional information it is capable of encoding. LieRE has the ability to use both relative and absolute positional information, and, ideally, we would like to rule out there being a simpler method that could perform just as well. The drop in performance when shuffling patches is one limited datapoint consistent with that hypothesis.

**Synthetic task**: We wholeheartedly agree with the assessment that the synthetic task has limitations. Still, we can see clear failures of basic spatial reasoning even in frontier models trained with resources well beyond the reach of any academic lab. Reproducing similar patterns in simple experiments greatly improves the accessibility of studying these issuesIn fact, understanding these failures is an increasingly growing area of research [6,7]. We believe better position encodings are one of the ingredients that will be necessary to resolve these limitations and hope the data point in our paper is suggestive of that. It is important to note that many VLMs are still trained with absolute position encodings that perform substantially worse than the relative position encoding we benchmarked (Table 1). While paper coherence and practical considerations prevent us from fusing this work with a new research project focused on resolving these limitations on spatial reasoning, we hope this experiment serves as a datapoint used to inform future work on spatial reasoning.
Writing and figures: Thank you for the feedback on the writing and figures. We have used it to improve the readability of the paper and make changes such as switching to more readable high contrast color schemes for the figures. (such as Figure 5: https://postimg.cc/D8w2tcKp)

Thank you for the thoughtful review, and we hope we addressed your remaining concerns.

#### Other Strengths And Weaknesses

See above.

#### Other Comments Or Suggestions

The overall writing is not that easy to follow. 
Missing figure numbers in line 307, and the formatting needs improvement. 
Figure5. is too hard to parse.

#### Essential References Not Discussed

Not found.

#### Was Revised

`false`

#### Extra Scores

{}

---

### Review 3 — Reviewer_gAmX

- Review ID: `ynusNde6cA`

#### Summary

LieRE extends the popular RoPE by replacing its block-diagonal 2D rotation matrices with learned, dense, high-dimensional rotation matrices derived from Lie group theory.
The authors show that LieRE addresses key limitations of RoPE, particularly for multi-dimensional data like images and videos. Specifically, while RoPE was originally designed for one-dimensional sequence processing (like text), LieRE generalizes position encoding to higher dimensions through the use of Lie groups. The method involves learning skew-symmetric basis matrices and computing rotation matrices for n-dimensional positions, which are then applied to keys and queries in the attention mechanism.
The paper evaluates LieRE against other positional encoding methods on several tasks, the results show that LieRE outperforms competing methods across these tasks, with particular advantages in data efficiency, resolution invariance, and when processing limited training data.

#### Claims And Evidence

The claims made in the paper are well-supported by empirical evidence, but i just want to point out that the assertion that LieRE provides a "unified approach" for handling different dimensionalities is supported by experiments on 2d and 3d data, but testing on additional dimensionalities (like 1d sequences, higher-dimensional data) would make this claim more robust.

#### Methods And Evaluation Criteria

The methodology and evaluation criteria seem well designed, even though the paper focuses on classification tasks, which may not fully showcase the advantages of better positional encodings. Tasks requiring fine-grained spatial understanding (like segmentation or object detection) would provide a more comprehensive evaluation.

#### Theoretical Claims

There's no formal proof that LieRE can handle "exponentially many relative positions for n-dimensional data" better than alternatives, though the empirical results are supportive.

#### Experimental Designs Or Analyses

The experimental design is sound.

#### Relation To Broader Scientific Literature

The paper builds directly upon RoPE and its variants (RoPE-Mixed, VisionLlama), clearly identifying limitations and proposing extensions. The application of Lie group theory to positional encodings is a new connection between abstract algebra and deep learning architectures.

The paper doesn't extensively discuss connections to other approaches for handling multi-dimensional data in transformers, such as axial attention or perceiver architectures. 

Overall, while the paper makes a significant contribution to positional encoding research.

#### Questions For Authors

N/A

#### Rating

`5`

#### Rebuttal

Dear Reviewer gAmX,

Thank you for your thorough and supportive review. We particularly appreciate your recognition of our mathematical foundations and empirical results. We have built upon your feedback to further improve the paper. In addition to the changes below, we have also expanded the paper to better relate our work to other architectural work such as perceivers and axial attention.

**On sensitivity to initialization**: We originally discovered the sensitivity as we were iterating to reproduce the results of RoPE-Mixed. RoPE-Mixed seems to be particularly sensitive to the scale of the initial weights. LieRE is about half as sensitive. We have added a section to the appendix that explores this in greater detail, but in short, RoPE-Mixed shows up to a 2% drop in CIFAR100 performance while  LieRE shows a 1% drop. Outside of the new initialization sensitivity experiment, all experiments for both LieRE and RoPE-Mixed are performed with the setting where RoPE mixed performs best. The key table is presented below.
| Metric | LieRE_8 (2π vs 1 init) | RoPE-Mixed (2π vs 1 init) |
|---------|------------------------|---------------------------|
| Z-statistic | -1.81 | -3.85 |
| P-value | 0.070 | 0.00012 |
| Difference between means | -0.0118 | -0.0255 |
| 95% Confidence Interval | [-0.0246, 0.0010] | [-0.0385, -0.0125] |

**1D evaluation tasks**: We have added a section characterizing the use of LieRE for 1D tasks. In short, LieRE is equivalent in representational capacity to RoPE with learnable frequencies for one-dimensional tasks. Please see the response to reviewer “udrb” for the proof of this fact that will be included in the paper. Our empirical experiment confirms this equivalence.
Higher-dimensional evaluation tasks: Thank you for the suggestion! To our knowledge, LieRE is the first approach to extend rotational position encodings to 3D data, and we are eager to explore scaling to 4D. Given the dataset size required to train transformers on high-dimensional data, such extensions may currently rely more on synthetic data or alternative ways of structuring dimensional information. We see this as an exciting next step and appreciate your insights on pushing this further.

**Related Work**: Thank you for pointing out the connection to Axial attention and Perceivers. We have added a section of related work focused on related works towards compute-efficient scaling beyond sequence data. 
Axial Attention [19] reduces computational complexity by applying attention along specific axes (e.g., rows and columns in images), enabling transformers to scale efficiently with high-dimensional data. Perceiver [20] utilizes latent tokens to compress high-dimensional inputs into a smaller set, improving scalability as input size and dimensionality grow. These methods address the inefficiencies of traditional transformers when applied to high-dimensional data. Additionally, techniques like Swin [21] and Vmamba [22] optimize compute for visual data. Swin Transformer introduces a hierarchical approach with shifted windows, limiting attention to local regions to reduce complexity while capturing global context.  Vmamba, on the other hand, proposes a visual state space model that represents images as a collection of spatial states, allowing attention to be applied efficiently across large-scale visual inputs by exploiting spatial locality and reducing redundant computation. 
It would be great to compound these methods with LieRE, as in this work we use a plain encoder transformer. 

Thank you for constructive feedback which we were able to use to further improve the paper. We are excited to share this work with the community. 


**=== LieRE vs. RoPE 1D proof  ===**

Though focused on higher dimensional inputs LieRE remains compatible with 1D tasks. It turns out that in the 1D setting, LieRE has identical representational capacity to RoPE. This is not the case for higher dimensional inputs for reasons that will be clearer later in the exposition. We include a cut-down version of the proof we propose to add to the paper below.

Recall that in the 1D setting positions are scalars. The LieRE rotation is $R=\\exp(tA)$ for some learnable skew-symmetric matrix A. Recall that skew-symmetric matrices can be written in the form $S^T \\Lambda S$ where $S$ and is orthogonal and $$
\\Lambda = \\begin{pmatrix}
0 & \\lambda_0 & & & \\\\
-\\lambda_0 & 0 & & & \\\\
& & 0 & \\lambda_1 & \\\\
& & -\\lambda_1 & 0 & \\\\
& & & & \\ddots
\\end{pmatrix}
$$

We can then use an identity of the matrix exponential to break down the LieRE rotation matrix. 
$R = exp(tS^T \\Lambda S) = S^T \\exp(t \\Lambda ) S$. For two tokens in positions $t, t’$ we denote the embeddings for a specific attention head as $x_t,x_{t’}$. If $K, Q$ denote the corresponding key and query linear transformation matrices we can write the attention inner product  with LieRE explicitly. 

**... continued at the end of next rebuttal ...**

#### Other Strengths And Weaknesses

Other weaknesses:
- The authors note that "RoPE-Mixed is sensitive to the initialization of weights," suggesting that LieRE might share this sensitivity, but they don't thoroughly explore how different initialization strategies might affect performance.

Other strenghts:
- The paper provides a solid mathematical foundation based on Lie group theory, extending positional encodings in beyond the current state-of-the-art methods.
- LieRE shows very good ability to generalize to image resolutions not seen during training, outperforming other methods especially at higher resolutions.
- The patch shuffling experiments offer valuable insights into how much the model utilizes positional information, with LieRE showing the most significant performance drop when positional information is disrupted.

#### Other Comments Or Suggestions

N/A

#### Essential References Not Discussed

N/A

#### Was Revised

`false`

#### Extra Scores

{}

---

### Review 4 — Reviewer_yH2U

- Review ID: `Rxw4EVd8Pl`

#### Summary

The authors introduce a type of positional embedding which extends the RoPE embeddings by introducing learnable rotation matrices.

## update after rebuttal
I thank the authors for their thorough response. In light of this, I will increase my score to weak accept.

#### Claims And Evidence

The authors present reasonable although somewhat limited experimental validation, by training ViT models on CIFAR-10, Imagenet-1k and UCF101.
The gap between their model and RoPE mixed (from which it is an incremental modification) is very small.

#### Methods And Evaluation Criteria

Yes

#### Theoretical Claims

N/A

#### Experimental Designs Or Analyses

N/A

#### Relation To Broader Scientific Literature

The litterature review is thorough, and I appreciate that the authors are honest in acknowledging similitudes with existing methods ("Note that the only difference between LieRE and RoPE-Mixed is that the latter constrains the rotations to be block-diagonal with block size two").

#### Questions For Authors

None

#### Rating

`3`

#### Rebuttal

Dear Reviewer yH2U,

Thank you for your thoughtful review and detailed feedback. We understand your concerns about the incremental nature and effectiveness of our work, and would like to address these directly:

**On novelty and the primary contribution of the work**: We split up our contributions broadly into (1) analysis and (2) method. 

**Analysis**: The closest work to ours is RoPE-Mixed. We build on their work with both a substantially larger performance delta and more extensive analysis. We aim to include coverage of technical details not present in the most similar prior work such as the sensitivity to weights initialization (see the table in the response to reviewer gAmX) and different methods of defining extrapolated token positions. In addition to an extended quantitative comparison, we aim to provide qualitative insights to how the position encodings affect inference and training dynamics with the attention maps and patch shuffling experiments. Finally, we extend experimental coverage to study the effect of the dimensionality of the input, a first even for existing position encodings. Concretely, to the extent of our knowledge, this is also the first work that benchmarks RoPE-Mixed for 3D data.

**Method**: We believe strongly that the presentation of a technical work should be as easy to understand. This motivates us to keep the connection to existing methods simple, including being direct about the settings where the methods are identical and the additional complexity of LieRE is worth the additional complexity.
Novel technical machinery is required in order to utilize high dimensional rotation matrices that are dense or have block size more than two. In particular, we introduce the method of encoding the positions in a basis of skew symmetric matrices which is then passed through the matrix exponential to obtain dense high-dimensional rotations. This use of Lie group theory is novel, and the key ingredient in enabling the use of dense high-dimensional rotations with allows LieRE to go beyond RoPE-Mixed with statistically significant performance improvements.
| Dataset | p-value (Rope-Mixed vs. LieRE_8) |
|---------|----------------------------------|
| CIFAR-100 | 1.3e-05 |
| ImageNet-1k | 6.3e-03 |
| UCF101 | 7.1e-04 |
| 384 x 384 (Resolution Invariance) | 4.0e-04 |

Your observation about block size unpredictability helped us recognize the need to better articulate our findings: LieRE_8 consistently demonstrates optimal performance across both 2D and 3D experiments (Appendix B.9. Basis parameters, Table 8). We will incorporate clear guidelines for practitioners based on our systematic analysis.

**Writing Improvements**: We will address the writing issues you identified: complete the unfinished sentence in section 5.2, fix the undefined reference and clarify Figures 1-2 with improved visual explanations (Figure 5: https://postimg.cc/D8w2tcKp).
We believe these revisions will better communicate both the theoretical contributions and practical benefits of our work.

We again thank Reviewer yH2U for their review of our paper. We hope that the above responses adequately address all concerns.


**=== LieRE vs. RoPE 1D proof continued ===**

$$
\\begin{align*}
x_t K R_t^T R_{t'} Q x_{t'} &= x_t K S^T \\exp(t \\Lambda) S S^T \\exp(t' \Lambda) S Q x_{t'} \\\\
&= x_t K S^T \\exp(t \\Lambda)^T \\exp(t' \\Lambda) S Q x_{t'} \\\\
&= x_t K S^T \\exp(t \\Lambda)^T \\exp(t' \\Lambda) S Q x_{t'}
\\end{align*}
$$

We let $K’=K S^T$ and $Q’= S Q $, since these matrices are all learnable we can fold the S matrix into parameters of the key and query linear layers for the given head, allowing us to simplify the above expression. 
$$ x_t K’ \\exp(t \\Lambda)^T \\exp(t’ \\Lambda) Q’ x_{t’} $$

Now we use the fact that each block is skew symmetric. In the case of two dimensions, 

$$\\exp\\left(\\begin{pmatrix} 0 & \\lambda \\\\ -\\lambda & 0 \\end{pmatrix}\\right) = \\begin{pmatrix} \\cos(\\lambda) & \\sin(\\lambda) \\\\ -\\sin(\\lambda) & \\cos(\\lambda) \\end{pmatrix} $$

If we let $R_\\lambda$ denote a block diagonal rotation matrices with 2D rotations of angles $\\lambda_0, \\ldots, \\lambda_n$, we can rewrite the above expression in a more familiar form. 

$$ x_t K’ R_{t \\Lambda} ^T R_{t’ \\Lambda} Q’ x_{t’} $$

This is exactly the formulation of the original RoPE position encoding. This also makes more clear how LieRE is different from RoPE-Mixed in the high dimensional setting. The above proof depends on the fact that we can decompose every rotation into a matrix of the form $S^T \\Lambda S$ with S not dependent on the position, allowing us to fold the orthogonal S matrices into the key and query matrices. This decomposition with constant S is guaranteed because the inputs to the matrix exponential differ by only a scalar factor. This is no longer true once we switch to more than a one dimensional basis of skew symmetric matrices.

#### Other Strengths And Weaknesses

Strengths:
- Paper is well-written and easy to follow
- Method seems to marginally outperform existing methods at little compute increase

Weaknesses:
- Novelty: as acknowledged by the authors, this method is an incremental modification of the existing RoPE-Mixed embeddings where instead of having block matrices of block size 2, the block size becomes a hyperparameter. 
- Effectiveness: I am not convinced of the benefits of this method. First, as shown in figure 5 and table 2, the increase in performance is rather marginal. Second, as shown in Figure 8, the effect of this hyperparameter on performance is rather unpredictable, which does not make this method particularly practical. Although the sections 5.2 and 5.6 of the paper are a bit more convincing, I remain lukewarm about the effectiveness of the method.
- Unpolished: the paper seems to have been rushed nearing the deadline and feels unpolished. Consider section 5.2: the second paragraph ends with an unfinished sentence ("The only exception to this is absolute position encodings, where we have variants trained on 800,000 and") and the third paragraph contains an undefined reference. Additionally, figures 1 and 2 are not very clear in my opinion.

#### Other Comments Or Suggestions

None

#### Essential References Not Discussed

N/A

#### Was Revised

`false`

#### Extra Scores

{'ethical_review_concerns': 'None'}

---

## Meta-Review / Decision Comment

The paper had initial mixed reviews, where the major concerns were about limited experimental validation, unconvincing results, and missing discussion of related work. On the other hand, reviewers agreed that the idea is elegant and interesting and the method is built on a strong mathematical foundation.

The negative reviewers increased their score after the rebuttal so the paper ended up with unanimous acceptance recommendation. I agree with the reviewers. Although submission would be stronger with more convincing experimental results, the idea of using skew-symmetric matrices to generalize RoPE is quite interesting and I'm glad to see it being implemented.


---

# Rethinking Addressing in Language Models via Contextualized Equivariant Positional Encoding — OpenReview 审稿全文归档

- Venue: **ICML 2025 poster**
- OpenReview forum: [https://openreview.net/forum?id=wgGC1N4rKy](https://openreview.net/forum?id=wgGC1N4rKy)
- Official paper page: [https://proceedings.mlr.press/v267/zhu25t.html](https://proceedings.mlr.press/v267/zhu25t.html)
- Reviewer handles are public OpenReview pseudonyms; no identity resolution is attempted.
- Source: ReviewArena public release (OpenReview-sourced). ICML 2025 uses the venue’s 1–5 Overall Recommendation scale; do not compare these numbers directly with ICLR’s 1–10 scale.

## Review Inventory (4 Official Reviews)

- Final decision: **Accept (poster)**

## Official Reviews

### Review 1 — Reviewer_qpJK

- Review ID: `u9A5LTtbuN`

#### Summary

This paper proposes a new method for learnable positional encodings, where they are allowed to depend on context/content. The positional encodings, termed TAPE (“conTextualized equivariAnt Position Encoding”), can be added to pre-trained transformers, with only the TAPE-relevant parameters fine-tuned. TAPE is permutation and orthogonal equivariant and uses higher order tensors, and performs well in experiments spanning arithmetic reasoning, long-context retrieval, and language modeling.

#### Claims And Evidence

L051, “This rigidity [the fixed distance dependence / locality bias] limits the ability of positional encodings to model long-range dependencies and makes it challenging to attend to distant query-key pairs.” Is there a citation or experiment to support the claim that it is the positional encodings, in particular, that make long-range dependencies hard for LLMs?

L686: “To the best of our knowledge, we are the first to introduce equivariance in language models, recognizing the symmetry in positional embeddings.” This is a strong claim. Is this not already done by RoPE, as noted in the paper?

#### Methods And Evaluation Criteria

Yes, although the highlighted benchmark datasets seem to be chosen as tasks where TAPE is expected to perform well (arithmetic and long context). However, other datasets are included in the appendix.

#### Theoretical Claims

n/a

#### Experimental Designs Or Analyses

The authors explain the success of TAPE in terms of permutation and orthogonal equivariance, as well as the tenderized representations, but only ablate orthogonal equivariance — this is the only area I see for improvement, as the other experiments seem very thorough.

#### Relation To Broader Scientific Literature

The paper presents a novel positional encoding, building on RoPE and others. It draws on insights from papers like Ebrahimi et al 2024 and Sinha et al 2022.

#### Questions For Authors

1. The arithmetic task is one where absolute positions are necessary (L316), but TAPE uses relative positions (Prop 3.1) — how then does TAPE do well? (Clarification of TAPE’s advantages and understanding of experimental results)
2. Is there an indexing problem with equation 7? Don’t the two sums cancel each other out? (Clarification for assessing paper’s accuracy)
3. Do the same hyperparameter choices (dimensions, etc) work across different tasks? (Affects my evaluation of advantages/disadvantages of the method)

#### Rating

`4`

#### Rebuttal

We greatly thank Reviewer qpJK for appreciating our contributions. We address the concerns as follows.

> W1: The motivation of the technique is a bit confusing. The authors claim that relative positional encodings are crucial for “stability and generalization to varying sequence lengths” (L223-224), and use relative positional encodings in their formulation (equation 6) and prove that the transformer is invariant to token index shift (Prop 3.1), but then highlight an arithmetic task where absolute positions are necessary. 

We apologize for the inaccuracy in our original statement (Line 316) regarding absolute positions that caused confusion. Please refer to our response to Q1 to see whether this clarification addresses the point.

---
>W2: Also, the use of tensors is not very well-motivated. Overall, although the experimental performance is seemingly very good, the design of TAPE seems a bit ad-hoc.

We acknowledge that certain implementation choices in TAPE, such as the tensorial embedding design, were empirically driven to optimize performance. As noted in Appendix D, our ablation studies validate the effectiveness of these architectural decisions.

---
>W3: The paper also cites geometric learning as inspiration — “This approach is inspired from the studies for geometric deep learning which processes graphs and point clouds by integrating token features with their geometric properties while preserving inherent physical symmetries” (L83) — but isn’t this property (positional encodings depending on relative distances) already satisfied by RoPE? Some additional ablations that could help clarify the design of TAPE include ablating the tensorial nature (reducing dimensions), ablating the dependence on context in the positional embeddings, etc.

Yes, RoPE also satisfies this property, which is precisely why we employ it as one of our instantiation methods. We do not claim this property as our novelty, but rather use it as a principle to motivate our design choices regarding contextualized position embeddings. As shown in Appendix D, we provide ablation studies that: (1) validate the design of equivariant and tensorial embeddings, (2) analyze the effects of Attention and MLP layers, and (3) investigate the impact of hyperparameter choices. Note that ablating the context-dependence in our positional embeddings (our core design) reduces to RoPE, which is consequently included as a baseline in our experiments.

---
> Q1: The arithmetic task is one where absolute positions are necessary (L316), but TAPE uses relative positions (Prop 3.1) — how then does TAPE do well? (Clarification of TAPE’s advantages and understanding of experimental results)

We appreciate this opportunity to clarify. There was an inaccuracy in the original L316 statement regarding absolute positions - we have now corrected this, as absolute positions are not necessarily required for arithmetic. The critical factor is learning the relative importance of different positions within the sequence. As detailed in L328 onwards, TAPE is able to learn these position-dependent importance relationships within the task context. A detailed explanation is also attached:

In arithmetic tasks, every digit has equal importance to the equation, regardless of its distance from the output. Traditional positional embeddings often assume a distance-decay effect, where words farther apart are less significant in the output. While this assumption is valid for most language tasks, it does not hold for arithmetic tasks. Positional contextualization enables dynamic reweighting of positional importance based on the task context, preserving effective distance decay for language tasks while addressing arithmetic contexts appropriately. This highlights TAPE’s potential advantages in arithmetic tasks.

---
> Q2: Is there an indexing problem with equation 7? Don’t the two sums cancel each other out? (Clarification for assessing paper’s accuracy)

No, the two sums in Equation 7 form a linear combination of vectors with weights summing to 1, making they cannot cancel out.

---
> Q3: Do the same hyperparameter choices (dimensions, etc) work across different tasks? (Affects my evaluation of advantages/disadvantages of the method)

Yes, we maintain consistent hyperparameters across all tasks (with L=R=2 in our main experiments). Additionally, Appendix D provides detailed ablation studies analyzing the impact of different hyperparameter choices.

---
*Response to Suggestions:*
1. Prop 3.1:  The orthogonality of R is indeed crucial for Proposition 3.1 to hold, as non-orthogonal transformations would violate the invariance properties demonstrated in Appendix B. While generalizing the assumptions is an interesting direction, we believe this extension merits dedicated future research.
2. Figure 2: The x- and y-axes represent the sequence lengths of the two operands respectively.
3. Typos: Thank you for your careful review. We have fixed all of them.

#### Other Strengths And Weaknesses

Strengths: 

The proposed positional encodings, TAPE, perform well compared to baselines on arithmetic and long-context tasks. They also admit an efficient implementation, and enable parameter-efficient fine-tuning that works better than LoRA and LongLoRA on passkey retrieval. Several additional experiments that I might have requested were already in the appendix, including ablations of orthogonal equivariance, and evaluation on other LLM tasks where long context is not necessarily the main challenge.

Weaknesses:

The motivation of the technique is a bit confusing. The authors claim that relative positional encodings are crucial for “stability and generalization to varying sequence lengths” (L223-224), and use relative positional encodings in their formulation (equation 6) and prove that the transformer is invariant to token index shift (Prop 3.1), but then highlight an arithmetic task where absolute positions are necessary. Also, the use of tensors is not very well-motivated. Overall, although the experimental performance is seemingly very good, the design of TAPE seems a bit ad-hoc. The paper also cites geometric learning as inspiration — “This approach is inspired from the studies for geometric deep learning which processes graphs and point clouds by integrating token features with their geometric properties while preserving inherent physical symmetries” (L83) — but isn’t this property (positional encodings depending on relative distances) already satisfied by RoPE? Some additional ablations that could help clarify the design of TAPE include ablating the tensorial nature (reducing dimensions), ablating the dependence on context in the positional embeddings, etc.

#### Other Comments Or Suggestions

I would recommend making Proposition 3.1 more general, by explicitly stating the necessary assumptions on E rather than immediately specializing to RoPE and random Fourier features. Also, would the result of Prop 3.1 not hold if f and g satisfied (4) and (5) for only R in the set of permutation matrices, rather than orthogonal matrices? This counterfactual would be good to include as part of the statement, if Prop 3.1 is supposed to motivate orthogonal invariance specifically. 

Figure 2: can the authors specify what the x and y-axis are in the figure caption? 

As a minor comment, there were a lot of typos. Here are some of them:

Typos:
L99, “adiditionaly” 
L77, “superioroty”
L150, “transformer” -> “transformers”
L162, “encoding” -> “encodings”
L237, “conTexturalized” 
L246, “O(r)-invariance” —> “O(r)-invariant”
L285, “test” —> “tested”
L297: “arthimetic” 
L663, “quardratic”
L672: “focus” -> “focuses”
L792, “Contextulization”

#### Essential References Not Discussed

There are other references that deal explicitly with general group equivariance in positional encodings for geometric applications, but I don’t think they are essential.

#### Was Revised

`false`

#### Extra Scores

{}

---

### Review 2 — Reviewer_G28F

- Review ID: `bwEjGaKcqR`

#### Summary

This paper introduces a new approach to processing language sequences using transformer blocks, where token features and positional embeddings are combined and contextualized. The authors extend traditional positional encoding by dividing it into multiple blocks, allowing for more flexible associations between tokens and their positions. They also ensure that their functions for token mixing and position contextualization are equivariant to permutations and orthogonal transformations, addressing limitations in existing models. The overall goal is to improve how transformer models process both token features and positional information in sequences.

## update after rebuttal

I think this is a good paper and keep my positive score.

#### Claims And Evidence

Yes.

#### Methods And Evaluation Criteria

Yes.

#### Theoretical Claims

Yes, I check the proof of Proposition 3.1.

#### Experimental Designs Or Analyses

I check all experiments.

#### Relation To Broader Scientific Literature

It provides a new way to handle positions in Transformer, which can be wildly applied to diverse domains.

#### Questions For Authors

None.

#### Rating

`4`

#### Rebuttal

We sincerely appreciate Reviewer G28F's positive assessment of our contributions and strong endorsement for acceptance. The reviewer provided one suggestion, to which we respond below:

> I do not see any major weaknesses. One suggestion is about the study of hyper-parameters. I think the authors choose B, L, R to align with RoPE for comparison. Since the proposed embeddings are learnable, it would be interesting to see how those hyper-parameters could affect the performance.

Thank you for your valuable suggestion. We have conducted hyperparameter exploration (detailed in the Appendix D), and our experiments demonstrate that the RoPE-like initialization yields the best performance among the configurations we tested.

#### Other Strengths And Weaknesses

Strengths
- Instead of a specialized architecture, this paper modifies attention and MLP layers of Transformer, allowing easy integration into existing transformers.
- Contextualized positional embeddings allow the model to dynamically adjust how positional information is interpreted depending on the surrounding tokens, improving its ability to capture more nuanced relationships between tokens.
- The authors conduct thorough evaluations on various tasks, including passkey retrieval, arithmetic learning, and training from scratch with different context lengths. Additionally, they provided clear and insightful visualizations of attention maps, which further enhance the understanding of their model's behavior.

Weaknesses
- I do not see any major weaknesses. One suggestion is about the study of hyper-parameters. I think the authors choose B, L, R to align with RoPE for comparison. Since the proposed embeddings are learnable, it would be interesting to see how those hyper-parameters could affect the performance.

#### Other Comments Or Suggestions

None.

#### Essential References Not Discussed

None.

#### Was Revised

`false`

#### Extra Scores

{}

---

### Review 3 — Reviewer_poAq

- Review ID: `nPZiGZH5CP`

#### Summary

This paper introduces TAPE ,a novel approach to enhancing position-based addressing in Transformers by dynamically adapting positional encodings across layers based on sequence context. TAPE ensures stability and robustness by enforcing permutation and orthogonal equivariance. Experimental results demonstrate that TAPE outperforms existing positional encoding techniques in language modeling, arithmetic reasoning, and long-context retrieval tasks.

#### Claims And Evidence

Yes, the claims are.

#### Methods And Evaluation Criteria

Yes, the proposed methods do.

#### Theoretical Claims

Yes, I checked the formulas in Section 3.

#### Experimental Designs Or Analyses

Yes, I checked the experimental results tables and their corresponding analysis.

#### Relation To Broader Scientific Literature

The Positional Encoding proposed in this paper aims to unleash the power of position-based addressing, as existing methods have gradually weakened this capability[1,2].

[1]Roformer: Enhanced transformer with rotary position embedding. Neurocomputing,2024

[2]Sun, Yutao, et al. "A length-extrapolatable transformer." 2022

#### Questions For Authors

- In Table 2 and Table 4, the authors present experimental results showing that the proposed method performs well on long-context tasks. I am curious whether this strong performance in long-context scenarios has theoretical support.

#### Rating

`3`

#### Rebuttal

We greatly thank Reviewer poAq for appreciating our contributions, providing valuable suggestions on improving the work, and supporting the acceptance of this work. We address the questions as follows.

>W1: The authors provide the running time of attention layers as experimental results in Table 4. However, since the proposed method updates positional features layer-wise through interactions and joint training with token representations in every Transformer layer, what is the experimental runtime during training and inference for large-scale language models?

Thank you for your suggestion. We have included the experimental runtime for full model inference in the updated table below.
| Method | TAPE | RoPE | FIRE | T5’s relative bias |
| --- | --- | --- | --- | --- |
| Samples Per Second | 58.6 | 71.8 | 33.4 | 46.8 |

>W2: Since the proposed method updates positional features layer-wise, how does this affect gradient in the model? Could the authors provide an analysis or empirical results on gradient changes to better understand the impact of this method on training process?

We appreciate this insightful question. During our experiments, we monitored gradient norms and did not observe significant differences compared to RoPE. However, as this aspect falls outside our primary focus (motivation, methodology, and significance), we did not conduct an in-depth analysis of gradient behavior.

>Q1: In Table 2 and Table 4, the authors present experimental results showing that the proposed method performs well on long-context tasks. I am curious whether this strong performance in long-context scenarios has theoretical support.

TAPE's superior performance in long-context scenarios can be attributed to two key factors: First, its learnable nature enables dynamic adaptation to varying context lengths. Second, as formally established in Proposition 3.1, our relative position encoding scheme possesses inherent generalization capabilities to unseen sequence lengths.
In contrast, conventional positional encoding methods exhibit fundamental limitations in long-context settings, as they either rely on predetermined distance-decay patterns, or lack this crucial relativity property.

#### Other Strengths And Weaknesses

Strengths：

- The proposed method is novel and effective, with corresponding theoretical support.

- The extensive experimental setup, especially in terms of performance on long-context tasks provides strong evidence of the effectiveness of the proposed method.

Weaknesses：

- The authors provide the running time of attention layers as experimental results in Table 4. However, since the proposed method updates positional features layer-wise through interactions and joint training with token representations in every Transformer layer, what is the experimental runtime during training and inference for large-scale language models?

- Since the proposed method updates positional features layer-wise, how does this affect gradient in the model? Could the authors provide an analysis or empirical results on gradient changes to better understand the impact of this method on training process?

#### Other Comments Or Suggestions

- Figure 3 could be represented using a different type of chart and would look more visually appealing if it occupies half a column.

#### Essential References Not Discussed

No, there aren’t.

#### Was Revised

`false`

#### Extra Scores

{}

---

### Review 4 — Reviewer_udZG

- Review ID: `u4v3Yd8gXc`

#### Summary

This paper proposes a new positional encoding method for LLMs, which could enhance the position-addressing ability of transformers. Permutation and orthogonal equivariance are also applied to enforce the positional encoding. This method demonstrates superior performance on various tasks, especially long-context tasks, such as passkey retrieval tasks.

#### Claims And Evidence

the claims are supported by experiments results and theoretical proof.

#### Methods And Evaluation Criteria

the authors propose context-aware positional encodings aiming to improve positional encoding and enhance the performance of LLMs during both pre-training and fine-tuning.

#### Theoretical Claims

I checked the proof of the proposition, and I think no issues with it.

#### Experimental Designs Or Analyses

I have checked the author’s experimental results and the baselines used for comparison. Regarding the need for more comparative methods on long-context tasks, I will elaborate on this specifically in the weakness.

#### Relation To Broader Scientific Literature

This paper contributes to the broader scientific literature by introducing a novel context-aware positional encoding method for LLMs, enhancing their position-addressing ability through permutation and orthogonal equivariance. It builds upon prior work on positional encoding techniques like RoPE[1], ALiBi[2].

[1] Su, Jianlin, et al. "Roformer: Enhanced transformer with rotary position embedding." Neurocomputing, 2024

[2]Press, Ofir, Noah A. Smith, and Mike Lewis. "Train short, test long: Attention with linear biases enables input length extrapolation.” 2021.

#### Questions For Authors

Computing a positional embedding at each layer increases the computational cost for larger-scale language models. Is the added time acceptable, and does the proposed method have scalability? Please provide a time complexity analysis.

#### Rating

`3`

#### Rebuttal

We greatly thank Reviewer udZG for appreciating our contributions, providing valuable suggestions on improving the work, and supporting the acceptance of this work. We address the questions as follows.

> W1: Existing positional encoding methods have introduced improvements to RoPE to better adapt it for long-sequence tasks, such as NTK-aware RoPE [4]. Could you provide comparative experimental results of this method on long-context tasks?

Thank you for your suggestion. We have implemented YaRN as an additional baseline in our SCROLLS benchmark experiments. The training is currently in progress, and we will update the results once they are available.

> Q1: Computing a positional embedding at each layer increases the computational cost for larger-scale language models. Is the added time acceptable, and does the proposed method have scalability? Please provide a time complexity analysis.

As shown in Table 3, TAPE introduces <1% additional parameters and approximately 12% increased computational cost (FLOPs). Since scaling typically involves stacking these layers, the overhead remains proportional. We believe the modest increase in memory usage and ~10% additional compute time is a reasonable trade-off for the benefits provided.

#### Other Strengths And Weaknesses

Strengths: The authors propose a dynamic, context-aware positional encoding, which can be applied during both the pre-training and fine-tuning stages. They provide extensive experimental results in the main paper and appendix, demonstrating the effectiveness of the method across various tasks. Additionally, the paper is well-structured, making it easy to understand.

Weakness: Existing positional encoding methods have introduced improvements to RoPE to better adapt it for long-sequence tasks, such as NTK-aware RoPE [4]. Could you provide comparative experimental results of this method on long-context tasks?

[4] Peng, Bowen, et al. "Yarn: Efficient context window extension of large language models." 2023.

#### Other Comments Or Suggestions

Several typos need to be corrected :
1. “superioroty” should be “superiority” in Line 76.
2. “fine-fining”should be “fine-tuning” in Line 399.

#### Essential References Not Discussed

N/A

#### Was Revised

`false`

#### Extra Scores

{}

---

## Meta-Review / Decision Comment

The paper introduces TAPE, a method designed to enhance positional embeddings by incorporating sequence content across layers. TAPE can be integrated into pre-trained transformers, with only the relevant parameters fine-tuned. The proposed method is novel, and it demonstrates strong performance compared to baselines on arithmetic and long-context tasks. Extensive experiments are conducted, and the technology quality is solid. Overall, this is a strong paper. For the final version, the authors should clarify the motivation further and provide additional results as per the reviewers' comments.


---

# Group Representational Position Encoding — OpenReview 审稿全文归档

- Venue: **ICLR 2026 Poster**
- OpenReview forum: [https://openreview.net/forum?id=itoNJ3gJl2](https://openreview.net/forum?id=itoNJ3gJl2)
- Official paper page: [https://proceedings.iclr.cc/paper_files/paper/2026/hash/5cb58625f49ddf70fe2d527e9e4bbae5-Abstract-Conference.html](https://proceedings.iclr.cc/paper_files/paper/2026/hash/5cb58625f49ddf70fe2d527e9e4bbae5-Abstract-Conference.html)
- Reviewer handles are the public OpenReview pseudonyms; no attempt is made to identify individuals.
- Source: public OpenReview review dump; fields are preserved as released, including review text, rebuttal comments, meta-review, and decision where available.

## Paper Abstract

We present GRAPE (Group RepresentAtional Position Encoding), a unified framework for positional encoding based on group actions. GRAPE brings together two families of mechanisms: (i) multiplicative rotations (Multiplicative GRAPE) in $\operatorname{SO}(d)$ and (ii) additive logit biases (Additive GRAPE) arising from unipotent actions in the general linear group $\mathrm{GL}$.
In Multiplicative GRAPE, a position $n\in\mathbb{Z}$ (or $t\in\mathbb{R}$) acts as $\mathbf{G}(n)=\exp(n\,\omega\,\mathbf{L})$ with a rank‑2 skew generator $\mathbf{L} \in \mathbb{R}^{d \times d}$, yielding a relative, compositional, norm‑preserving map with a closed‑form matrix exponential. RoPE is recovered exactly when the $d/2$ planes are the canonical coordinate pairs with log‑uniform spectrum. Learned commuting subspaces and compact non‑commuting mixtures strictly extend this geometry to capture cross-subspace feature coupling at $O(d)$ and $O(rd)$ cost per head, respectively.
In Additive GRAPE, additive logits arise as rank‑1 (or low‑rank) unipotent actions, recovering ALiBi and the Forgetting Transformer (FoX) as exact special cases while preserving an exact relative law and streaming cacheability. Altogether, GRAPE supplies a principled design space for positional geometry in long‑context models, subsuming RoPE and ALiBi as special cases.

## Review Inventory (4 Official Reviews, 11 Discussion/Comment Notes)

- Final decision: **Accept (Poster)**
- Official review ratings (review order): `4, 4, 2, 4`

## Official Reviews

### Official_Review — Reviewer_TbLJ

- Note ID: `QVRRVzJ60D`
- Discussion number: `1`
- Invitation: `ICLR.cc/2026/Conference/Submission20573/-/Official_Review`


#### Summary

This paper proposes GRAPE, a unified group-theoretic framework for positional encoding for 1D sequences that combines multiplicative rotations in SO(d) and additive unipotent actions in GL to recover and generalize methods like Rotary Position Embedding (RoPE), Attention with Linear Biases (ALiBi), and Forgetting Transformer (FoX).
Unlike prior work that focuses only on rotations, it formalizes both multiplicative and additive mechanisms under a single algebraic structure and introduces path-integral additive biases for contextual, streaming-friendly position encoding.
Experiments on FineWeb-Edu 100B with Llama show slightly improved stability and lower loss compared to RoPE, ALiBi, and FoX.

#### Soundness

`2`

#### Presentation

`2`

#### Contribution

`4`

#### Strengths

- The generator construction L is smart and novel. 
- The combination of both multiplicative and additive mechanisms under a single algebraic structure is a contributing perspective for this field. 
- The formal description of GRAPE is complete. 
- The method is efficient and is a direct extension of Rope for 1D sequences.

#### Weaknesses

The motivation of this work could be better described. Which practical problem does GRAPE solve? What can GRAPE encode what Rope and other variants cannot encode and why is that important in practice? 
- The comparison to prior works and the related work section is incomplete.
-- For the multiplicative GRAPE, there are several prior works like i.e. LieRE (Ostmeier et al.), STRING (Schenck et al.) that have conceptually predescribed and evaluated Lie Group structured positional encodings, where rank-2 exponentials in SO(d) and learned basis. How does GRAPE compare to just learning the 2x2 basis generators for the block diagonal rotation matrix?
-- How does GRAPE compare to YARN (Peng et. al)?
-- How does GRAPE compare to CoPE? 
- The paper aims to present a unified framework for positional encoding based on group actions for transformer in general, but only focuses on 1D sequence encoding and not higher dimensional inputs.
- Although it is valuable to present the learning curves, the experimental results could be better presented i.e. confidence intervals, at least two validation sets or a test set. 
- Ablations between multiplicative and additive would enhance the understanding of the practical contributions of each of them.

#### Questions

- Is the training fully converged? Would you mind running for 50 more epochs? 
- You emphasize the importance of exact relativity and orthogonality for translation invariance in positional encodings. Could you comment on what a strict structure is enables?
- What is the motivation for combining multiplicative and additive logit biases in your positional encoding design? Do they serve complementary roles (e.g., scaling vs shifting positional effects), and how does this impact learning stability or expressivity?

Minor:
- The abstract has undefined variables ($a$ and $b$ ... ) which are only later defined

#### Flag For Ethics Review

- No ethics review needed.

#### Rating

`4`

#### Confidence

`4`

#### Code Of Conduct

Yes

### Official_Review — Reviewer_MUmm

- Note ID: `WJod8GcscH`
- Discussion number: `2`
- Invitation: `ICLR.cc/2026/Conference/Submission20573/-/Official_Review`


#### Summary

This paper proposes GRAPE, a unified theoretical framework for positional encoding based on group theory. The authors categorize existing methods into two families: Multiplicative GRAPE (rotations, like RoPE) and Additive GRAPE (biases, like ALiBi). The paper claims that RoPE, ALiBi, and FoX are all exact special cases or instances of this framework. The authors then propose a new, endpoint-dependent variant called Path-Integral Additive GRAPE (PI-Add-GRAPE). In a minimal experiment, this new method is shown to achieve lower (or possibly comparable) loss than baselines on a language modeling task.

#### Soundness

`2`

#### Presentation

`2`

#### Contribution

`3`

#### Strengths

This paper provides a somehow novel viewpoint to design of positional encoding. The goal of unifying the two dominant (and seemingly different) positional encoding methods (rotations and biases) under a single mathematical framework is could be ambitious and interesting.

The proposed PI-Add-GRAPE mechanism, which introduces content-dependent biases, may be a novel concept. In theory, this dynamic approach could offer more expressive power than static position methods.

#### Weaknesses

Unclear Practical Benefit of the Theory: The paper spends significant effort on group theory formalism. However, the practical benefit of this complex formalization is unclear. It seems obvious that rotational embeddings like RoPE can be described by group theory (e.g., SO(2)). The paper does not clearly explain what new, practical advantages this complex theory provides over a simpler understanding. The claim of offering a "design space" is abstract and its benefit is not well-supported.

Insufficient Experimental Validation: The empirical evaluation is minimal and insufficient to support the paper's claims. It consists of a single set of training curves for one model configuration. The claim of a "persistent edge" also appears to be an overclaim; the validation loss for ALiBi looks very competitive with PI-Add-GRAPE.
Furthermore, the paper is missing empirical analyses to understand the proposed method. For example, there are no ablation studies, no length extrapolation tests (a key feature of ALiBi), and no analysis of attention distributions to show how the dynamic bias works. Section 7 feels aimless; it shows a result but provides no insight into why the method is good or what its specific advantages are.

Computational Cost: The PI-Add-GRAPE method (Section 6) is endpoint-dependent. This implies that during inference at step t, the bias for all t−1 previous keys must be recomputed relative to the current query, right? This likely introduces a significant O(t) computational overhead per step, which is a major drawback compared to the O(d) cost of RoPE or ALiBi. This trade-off is not benchmarked empirically.

Logical Gaps and Confusing Terminology: The logical connection in the introduction (from "These observations" to "motivate a unified formulation" ) is a significant jump and is not well-justified for me. In addition, there is some confusing expressions (e.g., the interchangeable use of "exact special case" and "exact instance" is confusing).

#### Questions

Can you clarify the computational overhead of PI-Add-GRAPE during training and inference?

Given the complexity of PI-Add-GRAPE and validation of RoPE's unstable result, do you plan to release a reference implementation? This would be crucial for reproducibility and adoption by the community.

#### Flag For Ethics Review

- No ethics review needed.

#### Rating

`4`

#### Confidence

`3`

#### Code Of Conduct

Yes

### Official_Review — Reviewer_dFym

- Note ID: `F0BCGcsCSA`
- Discussion number: `3`
- Invitation: `ICLR.cc/2026/Conference/Submission20573/-/Official_Review`


#### Summary

The paper proposes a framework for positional encoding based on group actions, dubbed as GRAPE (Group RepresentAtional Position Encoding). Specifically, the authors describe two families of positional encodings grounded in group actions: (1) Multiplicative
GRAPE based in multiplicative rotations; (2) Additive GRAPE from unipotent actions in the general linear group. The authors describe how existing positional encodings such as RoPE, ALiBi, and FoX can be recovered as special cases within this framework. The authors also provide some empirical evidence supporting the advantages of GRAPE over existing positional encodings.

#### Soundness

`2`

#### Presentation

`2`

#### Contribution

`2`

#### Strengths

The paper grounds the design space of positional encoding (PE) in group actions, arriving at a general framework which can possibly motivate more expressive and useful PEs.

#### Weaknesses

1. The message of the paper is confusing. For Multiplicative GRAPE, the authors stated the exact relative law in Section 2.2 which naturally leads to commuting Mul-GRAPE, but then describe non-commuting Mul-GRAPE in multiple places (e.g. abstract, related work, appendix) without any motivations.

2. The paper devotes section 3 and 4 for describing Multiplicative GRAPE, but does not use it in the empirical experiment. This casts doubts on the practical utility of Multiplicative GRAPE.

3. The empirical experiments are quite limited. The authors compare PI-Add-GRAPE with other baseline PEs only on their loss curves, without other metrics (e.g., perplexity) or downstream task performance, or ablations (e.g., context length, model size).

#### Questions

1. Can the authors compare their Mul-GRAPE with the recently proposed LieRE in [1], which parameterizes the rotation as a sum of skew-symmetric matrices (followed by matrix exponential)? LieRE seems to provide a more general parameterization, so I am curious to see if this results in any computational or performance differences.

2. In Prop 3.1: the equality of MS-GRAPE and ROPE only holds when the planes are the canonical coordinates pairs and the angles follow the log-uniform spectrum, right? If so, I suggest to make the statement more precise.

3. The authors introduces GRAPE as a way to provide a group-theoretic view of PEs. Does GRAPE provide additional insights of the existing PEs, such as which PE one should choose over another given certain tasks in mind (e.g., length extrapolation)? 

 References
 [1] Ostmeier et al., LieRE: Lie Rotational Positional Encodings, ICML 2025

#### Flag For Ethics Review

- No ethics review needed.

#### Rating

`2`

#### Confidence

`3`

#### Code Of Conduct

Yes

### Official_Review — Reviewer_fTQk

- Note ID: `mYK77N3gfx`
- Discussion number: `4`
- Invitation: `ICLR.cc/2026/Conference/Submission20573/-/Official_Review`


#### Summary

The paper proposes GRAPE, a unified group-theoretic framework for positional encoding in Transformers. It combines Multiplicative GRAPE (rotations in SO(d), generalizing RoPE) and Additive GRAPE (unipotent actions in GL, recovering ALiBi and FoX). GRAPE preserves exact relative relationships, supports streaming, and offers an extensible design space for long-context modeling.

#### Soundness

`4`

#### Presentation

`3`

#### Contribution

`3`

#### Strengths

The paper presents an elegant theoretical unification of multiplicative and additive positional mechanisms within a single group-theoretic framework. It offers closed-form and computationally efficient implementations, demonstrates strong compatibility with existing Transformer architectures, and provides extensibility toward contextual, learned-basis, and non-commuting variants for more expressive positional representations.

#### Weaknesses

This paper utilizes Lie algebras. While unifying existing work with Lie algebras is natural, its drawback is that it makes the paper's contribution seem more like the superiority of Lie algebras themselves rather than the authors' contribution. I believe the authors should emphasize more on how introducing Lie algebras facilitates combining the strengths of various existing methods, explaining why each strength is beneficial, and then supplementing with corresponding ablation experiments.

The experiments in this paper are somewhat limited, lacking extrapolation experiments and comparisons with more metrics. I suggest at least supplementing with the experiments in Tab. 4 of RoPE.

Due to the highly complex formulas, I cannot guarantee my complete understanding of this paper. The pseudocode in the appendix does not alleviate my concerns about reproducibility; I would appreciate to see the complete project code.

#### Questions

See Weakness.

#### Flag For Ethics Review

- No ethics review needed.

#### Rating

`4`

#### Confidence

`3`

#### Code Of Conduct

Yes

## Meta-Review

### Meta_Review — Area_Chair_ELwV

- Note ID: `BQQVON5PDk`
- Discussion number: `1`
- Invitation: `ICLR.cc/2026/Conference/Submission20573/-/Meta_Review`


#### Summary

All reviewers like the rigorous framework introduced to unify RoPE and Alibi with a principled mathematical foundation to what were previously heuristic distinctions in position encoding. Main concerns were about complex presentation of the approach, missing comparisons and limited experiments. Authors have addressed both these concerns in the response. The authors added extensive evaluations on standard benchmarks (ARC, HellaSwag, PIQA) for 355M and 770M models. These results demonstrated that the derived method (GRAPE-A) consistently outperforms strong baselines (RoPE, ALiBi), proving the theory translates to practical performance gains.

I suggest borderline acceptance.

#### Reviewer Concerns

Main concerns were about complex presentation of the approach, limited experiments. Authors have addressed both these concerns in the response. The authors added extensive evaluations on standard benchmarks (ARC, HellaSwag, PIQA) for 355M and 770M models. These results demonstrated that the derived method (GRAPE-A) consistently outperforms strong baselines (RoPE, ALiBi), proving the theory translates to practical performance gains

#### Reviewer Scores

fTQk 4-> 6
dFym 2 -> 6
MUmm 4-> 6
TbLJ 4 ->6

## Rebuttal and Discussion Comments

### Official_Comment — Authors

- Note ID: `jEEgxONxJD`
- Discussion number: `3`
- Invitation: `ICLR.cc/2026/Conference/Submission20573/-/Official_Comment`


#### Comment

We thank the reviewer for the insightful comments and for recognizing the value of our theoretical unification.

> Q1: "This paper utilizes Lie algebras... drawback is that it makes the paper's contribution seem more like the superiority of Lie algebras themselves... emphasize more how introducing Lie algebras facilitates combining the strengths..."

**A1:** We utilize the Lie algebra to formally define a general design space that unifies and composes the distinct advantages of existing positional encoding methods, rather than merely for theoretical formalism. By identifying RoPE as orthogonal rotations in $SO(d)$ (ensuring norm-preservation) and ALiBi/FoX as unipotent actions in $GL(d)$ (providing extrapolatable decay), GRAPE serves as the first theoretical framework unifying these methods. This unification empowers us to derive new positional encoding mechanisms; for instance, we introduce Path-Integral Additive GRAPE (PI-Add-GRAPE), which rigorously composes multiplicative orthogonality with additive path-dependent decay. During our rebuttal, we have provided more experiments on  Llama-type models of medium and large size. Our new experiments in Figures 1 and 2 and Tables 1 and 2 show that GRAPE-A (i.e., GRAPE-Add-PI) consistently yields lower losses than strong baselines such as RoPE, ALiBi, and FoX. This highlights the strength of GRAPE, which allows us to develop new positional encoding methods under this unified framework.

> Q2:  The experiments in this paper are somewhat limited, lacking extrapolation experiments and comparisons with more metrics. I suggest at least supplementing with the experiments in Tab. 4 of RoPE.

**A2:** We have expanded our experimental section in the revision (see General Response) by adding comprehensive evaluations on standard downstream benchmarks (including ARC, HellaSwag, and PIQA) in Tables 1 and 2, alongside detailed training stability analysis for both PI-Add-GRAPE (GRAPE-A) and Multiplicative GRAPE (GRAPE-M). These results confirm that GRAPE consistently outperforms the baselines, including RoPE, ALiBi, and Fox. Regarding your reference to "Table 4" in the original RoPE paper, we interpreted this as a request for robust downstream task performance to ensure alignment with standard baselines, which we have now fully incorporated to demonstrate the model's superior generalization capabilities.

> Q3: Due to the highly complex formulas... I would appreciate seeing the full project code.

**A3:**  We have uploaded the full supplementary codebase to facilitate verification.  Despite the theoretical depth, the actual implementation of GRAPE relies on standard linear algebra operations,  ensuring it is reproducible and easy to integrate.

### Official_Comment — Authors

- Note ID: `kfx0njVT1J`
- Discussion number: `4`
- Invitation: `ICLR.cc/2026/Conference/Submission20573/-/Official_Comment`


#### Comment

We thank the reviewer for the constructive feedback and insightful questions.

> W1: The message of the paper is confusing... describe non-commuting Mul-GRAPE... without any motivation.

**A1:** We appreciate you pointing this out. In the revision, we have explicitly clarified that both commuting and non-commuting variants strictly satisfy the exact relative property. This property stems from the group structure ($G(n+m)=G(n)G(m)$. The motivation for the non-commuting variant is to enable cross-subspace coupling: while commuting GRAPE (like RoPE) processes coordinate planes independently, non-commuting GRAPE allows rotational mixing across dimensions. This offers richer geometric expressiveness while preserving the same exact relative law. We have revised Section 2.2 and the abstract for clearer presentation.

> W2: The paper devotes sections 3 and 4 to describing Multiplicative GRAPE, but does not use it... casts doubts on practical utility...

**A2:** We have addressed this in the revision. We added new experiments evaluating Multiplicative GRAPE (GRAPE-M). The results confirm its practical utility, demonstrating that the theoretical formulation in Sections 3 and 4 translates into competitive empirical performance. Please refer to the updated Figures 1, 2, and Tables 1, 2 in the revised PDF."

> W3: The empirical experiments are quite limited... without other metrics... or downstream task performance, or ablations...

**A3:** We have enriched the empirical evaluation in the revision. As detailed in the General Response, we added: 1) Comprehensive evaluations on standard benchmarks (e.g., ARC, HellaSwag) via the LM Evaluation Harness, presented in Tables 1 and 2; 2) Comparative analysis across different model scales (Medium 355M and Large 770M) in Figures 1 and 2; 3) New results for Multiplicative GRAPE (GRAPE-M), identifying the specific performance gains derived from the additive path-integral component within the unified PI-Add-GRAPE framework.

> Q4: Can the authors compare their Mul-GRAPE with the recently proposed LieRE in [1]...?

**A4:** We thank the reviewer for pointing out this related work. This is a crucial distinction. While LieRE parameterizes rotations via a sum of skew-symmetric matrices, GRAPE offers decisive advantages in computational complexity, Contextual Capability, and scope of unification:
1.  Computational Efficiency ($O(d^3)$ vs. $O(d)$): LieRE relies on the numerical matrix exponential (e.g., `torch.matrix_exp`), which involves expensive matrix-matrix multiplications. In contrast, GRAPE decomposes the action into rank-2 subspaces using closed-form Rodrigues-type formulas (Section 2.3 in our paper). We only require **vector-vector multiplication**, avoiding the high cost of numerical matrix exponentials and achieving significant speedups.
2.  Contextual Capability: This efficiency unlocks contextual (data-dependent) GRAPE. LieRE cannot easily model data-dependent rotations because recomputing the matrix exponential for every unique token is computationally prohibitive. GRAPE's closed-form solution makes this computationally feasible.
3.  Broader Group Scope ($GL(d)$): LieRE is restricted to the rotation group. GRAPE generalizes to the General Linear Group ($GL(d)$), allowing us to strictly unify scaling, shearing, and decay effects (like ALiBi) within the same framework, capabilities that LieRE's rotation-focused approach does not cover.
We have added a detailed comparison with LieRE in the Appendix of the revised paper.

> Q5: In Prop 3.1, the equality of MS-GRAPE and ROPE only holds when the planes are the canonical coordinate pairs and the angles follow the log-uniform spectrum, right?

**A5:** You are right. We have updated Proposition 3.1 to explicitly state these preconditions. Crucially, GRAPE's advantage lies in extending beyond this special case, which allows for learned orthogonal bases and adaptive spectra to capture geometries that canonical RoPE cannot.

### Official_Comment — Authors — Official Comment by Authors II

- Note ID: `QEFzodjjV0`
- Discussion number: `5`
- Invitation: `ICLR.cc/2026/Conference/Submission20573/-/Official_Comment`


#### Comment

> Q6: The authors introduce GRAPE as a way to provide a group-theoretic view of PEs. Does GRAPE provide additional insights...?

**A6:** Yes, the group-theoretic perspective rigorously explains empirical trade-offs and suggests a unified design strategy:
1. Periodicity vs. Decay: The framework reveals that length extrapolation relies on monotonic decay to penalize distant interactions. Compact rotations ($SO(d)$/RoPE) are strictly norm-preserving and periodic, fundamentally lacking the mechanism to attenuate signals based on distance. In contrast, non-compact unipotent actions ($GL(d)$) naturally model such distance-dependent attenuation, theoretically explaining why additive methods (ALiBi) generally outperform pure rotations in length generalization.
2. Contextuality: For tasks requiring complex dependency modeling, our framework suggests Contextual GRAPE (data-dependent group actions). Unlike static PEs, contextual actions dynamically warp geometry based on input tokens, offering higher expressivity.
3. Composition: Consequently, a highly effective strategy is to compose these properties. GRAPE enables combining the stability of $SO(d)$ with the extrapolation of $GL(d)$ (as realized in GRAPE-A, GRAPE-AP), which can yield performance superior to utilizing either mechanism in isolation.

#### Title

Official Comment by Authors II

### Official_Comment — Authors

- Note ID: `rdg84K9r4J`
- Discussion number: `6`
- Invitation: `ICLR.cc/2026/Conference/Submission20573/-/Official_Comment`


#### Comment

We thank the reviewer for the insightful comments.

> W1: The comparison to prior works and the related work section are incomplete... i.e. LieRE (Ostmeier et al.), STRING (Schenck et al.)... How does GRAPE compare to YARN...?

**A1:** We thank the reviewer for pointing out these relevant works and have expanded the Related Work section to address them. Regarding LieRE and STRING, while they also explore Lie group structures, GRAPE distinguishes itself by generalizing beyond rotations ($SO(d)$) to the General Linear Group ($GL(d)$),  unifying additive decay mechanisms (ALiBi) within the same formalism. Furthermore, GRAPE utilizes a specific rank-2 factorization that yields efficient closed-form Rodrigues-type formulas, avoiding the computational cost of numerical matrix exponentials. Regarding learning 2x2 basis generators, this corresponds exactly to our "Learned Commuting" variant, yet GRAPE extends this by allowing non-commuting mixtures for cross-subspace coupling. Finally, YaRN is an orthogonal frequency interpolation strategy compatible with our framework, while CoPE relies on position counting, whereas GRAPE-Contextual achieves adaptivity via geometric path integration on the group manifold.

> W2:  The paper aims to present a unified framework for positional encoding based on group actions for the transformer in general, but it only focuses on 1D sequence encoding and not higher-dimensional inputs.

**A2:** While our primary focus is on 1D sequences such as language and text, the framework is theoretically dimension-agnostic. We have updated the Conclusion to explicitly clarify that GRAPE extends to 2D inputs via multi-parameter subgroups (direct sums of generators). This theoretical extension naturally subsumes mechanisms like 2D-RoPE, demonstrating the framework's generality while leaving specific image-domain experiments for future work.

> W3: Although it is valuable to present the learning curves, the experimental results could be better presented, i.e., confidence intervals, at least two validation sets, or a test set.

**A3:** We have expanded the evaluation in the revision. Specifically, we added comprehensive results on standard downstream benchmarks (including ARC, HellaSwag, and PIQA) in Tables 1 and 2. These serve as robust, standardized test sets beyond the initial validation loss. Furthermore, the consistent performance gains observed across both Medium (355M) and Large (770M) model scales (Figures 1 \& 2) demonstrate the method's stability and reproducibility.

> W4: Ablations between multiplicative and additive would enhance the understanding of the practical contributions of each of them.

**A4:** In the revision, we specifically added Methodological Ablations comparing the isolated Multiplicative GRAPE (GRAPE-M) against the unified PI-Add-GRAPE (GRAPE-A). These results (alongside the RoPE and ALiBi baselines) allow us to quantitatively isolate the contributions of each component, confirming that combining multiplicative stability with additive decay yields superior performance compared to using either mechanism in isolation.

> Q1: Is the training fully converged? Would you mind running for 50 more epochs?

**A1:** Training on 50B tokens aligns with standard practices for models of this scale. In LLM pretraining, models are typically trained for only 1 to 4 epochs (often just one pass) to maximize data diversity. Running for "50 more epochs" is never done in this context, as it leads to severe overfitting without meaningful generalization gains. Our stopping criteria thus follow established community norms.

> Q2:  You emphasize the importance of exact relativity and orthogonality for translation invariance in positional encodings. Could you comment on what a strict structure enables?

**A2:**  A strict group structure guarantees that the interaction operator depends strictly on the relative offset $(j-i)$, guaranteeably eliminating any leakage of absolute position information; this is the theoretical prerequisite for length generalization. Simultaneously, orthogonality ensures the transformation is an isometry, preventing signal explosion or rank collapse as positions increase, which is critical for maintaining signal fidelity in long-context modeling.

### Official_Comment — Authors — Official Comments by Authors II

- Note ID: `jjMmNYfYMy`
- Discussion number: `7`
- Invitation: `ICLR.cc/2026/Conference/Submission20573/-/Official_Comment`


#### Comment

> Q3: What is the motivation for combining multiplicative and additive logit biases in your positional encoding design? Do they serve complementary roles (e.g., scaling vs shifting positional effects), and how does this impact learning stability or expressivity?

**A3:** To clarify, while our framework theoretically unifies them (Eq. 6.3), in our experiments, we intentionally evaluated them as distinct mechanisms to rigorously isolate their respective contributions. The motivation for the unified design space is indeed their complementarity: Multiplicative ($SO(d)$) offers norm-preservation for stability, while Additive ($GL(d)$) offers monotonic decay for extrapolation. By testing them separately, we demonstrate that the additive path-integral component alone is sufficient to outperform baselines, validating the strength of the derived mechanism without relying on the rotational prior.

#### Title

Official Comments by Authors II

### Official_Comment — Authors

- Note ID: `RSEAqjaxSg`
- Discussion number: `8`
- Invitation: `ICLR.cc/2026/Conference/Submission20573/-/Official_Comment`


#### Comment

We thank the reviewer for the detailed and valuable feedback.

---
> W1: "Unclear Practical Benefit of the Theory... The paper does not clearly explain what new, practical advantages this complex theory provides..."

**A1:** The practical benefit is not merely descriptive. The framework transforms positional encoding design from heuristics into a rigorous Lie group-based derivation process. Specifically, it enabled us to derive PI-Add-GRAPE (GRAPE-M in our experiments), which mathematically unifies the stability of $SO(d)$ (RoPE) with the extrapolation of $GL(d)$ (ALiBi). This derived mechanism outperforms baselines in our experiments, proving that the theory can lead to superior, concrete designs. Furthermore, it provides mathematical guarantees for creating valid contextual (data-dependent) positional encodings.

---
> W2: Insufficient Experimental Validation: The empirical evaluation is minimal... The claim of a "persistent edge" also appears to be an overclaim; the validation loss for ALiBi looks very competitive... Section 7 feels aimless...

**A2:** We have significantly enriched the evaluation in the revision (see General Response). We expanded beyond single loss curves to more standard downstream benchmarks (Tables 1 and 2) and added specific ablations on Model Scale and Methodology (GRAPE-M vs. GRAPE-A (PI-Add-GRAPE)). These new results confirm that GRAPE consistently outperforms other position encoding methods, including RoPE, ALiBi, and FoX, supporting our claim. Furthermore, we have rewritten Section 7 to provide the requested analytical depth, explicitly linking these empirical gains to the underlying group-theoretic properties.

---
> W3: "Computational Cost: The PI-Add-GRAPE method (Section 6) is endpoint-dependent... This likely introduces a significant O(t) computational overhead per step..."

**A3:** We agree that Path-Integral Additive GRAPE introduces an endpoint-dependent bias $b_h(t,j)$. However, this does not change the asymptotic complexity of self-attention.
Concretely, at decoding step $t$ and for each head, we compute $\{\psi_h(t,\ell)\}_{\ell \le t}$ as below:

$$
\ell \mapsto \langle \mathbf{p}_{t,h}, \mathbf{R}\_\ell \mathbf{p}\_{\ell,h} \rangle
$$

using cached probes $\mathbf{R}\_\ell \mathbf{p}\_{\ell,h}$ (Section 6). This costs $O(t d)$ operations, followed by an $O(t)$ prefix sum to obtain the full row $\{b_h(t,j)\}\_{j \le t}$. The baseline attention scores at step $t$ already require $O(t d)$ vector dot products to compute $\{q_t^\top k_j\}\_{j \le t}$. Thus, GRAPE-AP adds only a constant-factor overhead to the existing $O(t d)$ per-step attention cost, and the overall asymptotic complexity of the model remains unchanged. In practice, these additional probe dot products are inexpensive relative to the main QK and AV matrix multiplications and do not introduce a new computational bottleneck.

> W4: Logical Gaps and Confusing Terminology: The logical connection in the introduction... is a significant jump... confusing expressions...

**A4:** We have revised the manuscript to address these points, specifically refining the introduction to bridge the logical gap and standardizing terminology throughout the paper to ensure consistency.

---
> Q5: Can you clarify the computational overhead of PI-Add-GRAPE during training and inference?

**A5:** Please refer to our response to W3 above. In summary, the overhead is negligible relative to the attention matrix multiplication, and we utilize caching mechanisms to ensure efficiency during both training and inference.

---
> Q6: Given the complexity of PI-Add-GRAPE... do you plan to release a reference implementation?

**A6:** Yes, we have included the full supplementary codebase in the revision to facilitate verification and community adoption. As demonstrated in the code, the actual implementation relies on standard linear algebra operations, ensuring it is reproducible and straightforward to integrate.

### Official_Comment — Reviewer_MUmm

- Note ID: `gyfF7l3Btb`
- Discussion number: `9`
- Invitation: `ICLR.cc/2026/Conference/Submission20573/-/Official_Comment`


#### Comment

Thank you for your response. I may not be able to find time for a detailed review until later, so I am posting a quick reply for now.

First, regarding the "General Response" mentioned in your comment, could you please check if it was submitted correctly? I currently only see the replies to individual reviewers and cannot find the general note.

I also appreciate the clarification regarding the computational cost and your decision to release the implementation code.

Regarding the additional experiments, while a quick glance suggests they are very informative and useful, I must keep in mind the ICLR guideline stating that:

>Area chairs and reviewers reserve the right to ignore changes that are significantly different from the original paper.

Therefore, if the additional material is significant, I intend to base my comprehensive judgment primarily on the standard of the original submission rather than significant addition of experiments. However, I will certainly take into account any materials that help clarify misunderstandings. Of course, I don't mind if other reviewers evaluate the additional materials.

### Official_Comment — Authors

- Note ID: `REBpBlPk9A`
- Discussion number: `10`
- Invitation: `ICLR.cc/2026/Conference/Submission20573/-/Official_Comment`


#### Comment

Dear Reviewer, we are currently adding our general response. Thanks for your quick response!

Best,

The authors.

### Official_Comment — Authors — General Response to Reviewers and Area Chairs

- Note ID: `tH7TCOTj1o`
- Discussion number: `11`
- Invitation: `ICLR.cc/2026/Conference/Submission20573/-/Official_Comment`


#### Title

General Response to Reviewers and Area Chairs

#### Comment

Dear Reviewers and Area Chairs, we sincerely thank you for your careful reading and constructive comments.  

In the revised version, we expand and clarify both the theory and the experiments. We now give a short overview of the main updates that respond to these concerns.

---

### 1. Expanded Experiments and New Results

We now report results on standard downstream benchmarks from the LM Evaluation Harness. We consider a medium model with 355M parameters and a large model with 770M parameters. Both models train on 50B tokens from FineWeb-Edu 100B. The training pipeline is identical across positional encodings. Only the position mechanism changes.

Our main additive variant, GRAPE-A (Path Integral Additive GRAPE), matches or beats strong baselines RoPE, ALiBi, and FoX with or without KV shift on average zero-shot accuracy.

**Table 1: Medium model (355M), zero-shot accuracy on LM Eval Harness**

| Method              | ARC-E | ARC-C | BoolQ | HellaSwag | OBQA | PIQA | WinoGrande | SciQ  | Avg.  |
|---------------------|:-----:|:-----:|:-----:|:---------:|:----:|:----:|:----------:|:-----:|:-----:|
| RoPE                | 59.34 | 30.89 | **61.22** | 45.46 | 34.00 | 69.42 | 52.49 | 74.70 | 53.44 |
| ALiBi               | 57.07 | 30.80 | 61.16 | **46.98** | 34.60 | 69.48 | 52.96 | 79.70 | 54.09 |
| FoX                 | 56.78 | 29.01 | 59.11 | 43.07 | 32.80 | 67.74 | 51.07 | 76.10 | 51.96 |
| FoX (w/ KV-shift)   | 57.11 | 30.55 | 60.34 | 44.32 | 33.80 | 69.31 | 52.17 | 78.40 | 53.25 |
| **GRAPE-A**         | **59.68** | **31.91** | 60.06 | 46.27 | **35.00** | **69.64** | **53.83** | **79.90** | **54.54** |
| GRAPE-M (Ctx)       | 56.02 | 29.35 | 58.81 | 44.88 | **35.00** | 68.61 | 52.09 | 76.50 | 52.66 |
| GRAPE-M (non-Ctx)   | 56.31 | 30.55 | 61.77 | 44.82 | 34.40 | 68.44 | 53.67 | 75.20 | 53.15 |

Key observations for the 355M model: GRAPE-A reaches the highest average score of 54.54. This improves over ALiBi with 54.09 and RoPE with 53.44, and it stays above FoX and FoX with KV shift on most tasks.

**Table 2: Large model (770M), zero-shot accuracy on LM Eval Harness**

| Method              | ARC-E | ARC-C | BoolQ | HellaSwag | OBQA | PIQA | WinoGrande | SciQ  | Avg.  |
|---------------------|:-----:|:-----:|:-----:|:---------:|:----:|:----:|:----------:|:-----:|:-----:|
| RoPE                | 62.25 | 33.02 | 58.23 | 50.92 | 37.60 | 70.89 | 55.88 | 80.50 | 56.16 |
| ALiBi               | **63.43** | **34.81** | 59.69 | 52.88 | 36.80 | 71.33 | 56.20 | 82.40 | 57.19 |
| FoX                 | 59.22 | 32.00 | 59.69 | 49.78 | **38.00** | 71.00 | 54.62 | 79.20 | 55.44 |
| FoX (w/ KV-shift)   | 60.77 | 32.85 | **62.51** | 49.38 | **38.00** | 70.62 | 54.78 | 81.40 | 56.29 |
| **GRAPE-A**         | 62.79 | 33.19 | 59.11 | **53.18** | 36.00 | **71.98** | **57.62** | **84.10** | **57.25** |

Key observations for the 770M model: GRAPE-A again gives the best average score, 57.25. This slightly improves over ALiBi with 57.19 and stays clearly above RoPE with 56.16 and FoX variants.

### Official_Comment — Authors — General Response II

- Note ID: `44AABkyZbt`
- Discussion number: `12`
- Invitation: `ICLR.cc/2026/Conference/Submission20573/-/Official_Comment`


#### Comment

### 2. Clarifications on Theory and Exact Relative Law

Several reviewers asked for clearer motivation and a cleaner story for the group-theoretic construction. We address these concerns in the revision as follows.

1. We now stress that the Lie algebra formalism defines a concrete design space of positional operators that recovers RoPE, ALiBi, and FoX as exact instances, extends them to learned bases, non-commuting mixtures, and contextual variants, and guarantees exact relative laws and streaming cacheability whenever we stay inside the group construction.

   New experiments in Tables 1 and 2 show that one instance from this design, GRAPE-A / GRAPE-AP, outperforms RoPE, ALiBi, and FoX in our benchmarks, so the formal unification leads to a real gain and not only a rephrasing.

2. Section 2.2 now explicitly states that

   - the exact relative law  
     $\mathbf{G}(n+m)=\mathbf{G}(n)\mathbf{G}(m)$ and $\mathbf{G}(t-s)=\mathbf{G}(s)^{-1}\mathbf{G}(t)$  
     follows from the one-parameter subgroup property,
   - This statement does not require that generators commute across subspaces, so non-commuting GRAPE-M remains exactly relative and at the same time introduces cross-subspace coupling that RoPE cannot express.

3. RoPE fixes coordinate planes and uses a log uniform frequency spectrum. GRAPE-M allows learned orthogonal bases, so planes no longer tie to coordinate axes, and compact non-commuting mixtures that rotate and mix information across planes in a controlled way.
   In addition, GRAPE-A and GRAPE-AP view ALiBi and FoX as unipotent general linear actions with exact relative laws and streaming behavior. The path integral construction in GRAPE-AP then allows contextual, content-dependent decay, and the group structure still keeps the model mathematically well behaved.

4. For GRAPE-AP, the per-step cost stays $O(td)$, the same order as the baseline attention score $q_t^\top k_j$. The extra additive term comes from one extra similarity sweep plus a prefix sum, so the asymptotic complexity of the layer does not change.

   Section 6 and the Appendix now contain a more detailed complexity discussion. We describe the implementation of endpoint-dependent path integrals through cached probe vectors, and show that this mechanism does not create a new computational bottleneck.

---
### 3. Comparison with LieRE, STRING, YaRN, and CoPE

- LieRE uses skew-symmetric generators in $\mathrm{SO}(d)$ and dense matrix exponentials (for example `torch.matrix_exp`). Each head needs time $\mathcal{O}(d^3)$ and $\mathcal{O}(d^2)$ parameters. GRAPE-M uses rank-2 generators with closed-form Rodrigues type exponentials. It applies only vector-vector operations. Time and parameters per head scales as $\mathcal{O}(d)$. 

- STRING matches special separable and translation invariant choices of planes and spectra inside the GRAPE-M family.

- YaRN is an orthogonal frequency interpolation scheme for RoPE. It acts on the spectrum $\{\theta_i\}$ of a commuting GRAPE-M instance. The GRAPE-M view stays compatible with YaRN style rescaling and recentering of frequencies. 

- CoPE defines contextual position counting as a function of the input sequence. GRAPE expresses contextuality through data dependent group actions on phases, generators, or path integrals. These actions preserve exact relative laws. They keep a clear geometric meaning.

---

### 4. 2D and 3D GRAPE for Multimodal Position Encoding

- We add new text in the conclusion and appendix to explain the 2D and 3D GRAPE, which are direct sums of generators and multi-parameter subgroups for **2D and 3D** inputs such as images, video, and multimodal data.
- In this view, 2D-RoPE appears as one concrete special case inside the general group design.

---

### 5. Reproducibility and Code Release

- We have uploaded the codebase as supplementary files.
- It includes GRAPE-M and GRAPE-A implementations and scripts for reproduction of the LLaMA type training runs and the language model evaluation harness results.
- The code uses only standard linear algebra such as matrix vector multiplies, dot products, and prefix sums. Integration in existing codebases stays simple.

#### Title

General Response II

### Official_Comment — Reviewer_dFym

- Note ID: `hA5yyI8LjM`
- Discussion number: `13`
- Invitation: `ICLR.cc/2026/Conference/Submission20573/-/Official_Comment`


#### Comment

I appreciate the authors for the detailed responses, and the paper revisions along with additional experiments. My follow-up questions based on the authors' answers:

**Q1: Motivation and justification for GRAPE-M**

> A1: ... non-commuting GRAPE allows rotational mixing across dimensions. This offers richer geometric expressiveness...

The authors motivate GRAPE-M as more expressive than RoPE,  but I don't find convincing empirical results or conceptual discussions on where GRAPE-M can outperform RoPE. Do I miss anything here?

> A6: ...Periodicity vs. Decay: The framework reveals that length extrapolation relies on monotonic decay to penalize distant interactions....
 
* This claim that ALiBi and its generalizations (e.g. GRAPE-A) prefers RoPE and its generalizations (e.g., GRAPE-M) for length generalization tasks is supported by the newly added empirical results where ALiBi/GRAPE-A outperforms RoPE/GRAPE-M for most eval tasks. 
* However, I am still confused by reading this together with A1 (and the whole paper). If we know A6 already, why bother proposing GRAPE-M? Perhaps there are certain tasks where GRAPE-M shines, but it remains unclear the utility of GRAPE-M.

**Q2: Computational costs for different positional encoding variants**

I thank the authors for providing the extra lm-eval experiments. Can you also provide compute time/memory comparison across these methods? While the authors added the discussion on computational complexity, I am curious to see if the newly-proposed variants require extra wall-clock time or memory consumption in practice.

## Decision

### Decision — Program_Chairs — Paper Decision

- Note ID: `FpI4TCADHA`
- Discussion number: `1`
- Invitation: `ICLR.cc/2026/Conference/Submission20573/-/Decision`


#### Title

Paper Decision

#### Decision

Accept (Poster)


---

# Selective Rotary Position Embedding — OpenReview 审稿全文归档

- Venue: **ICLR 2026 Poster**
- OpenReview forum: [https://openreview.net/forum?id=AQo1SEElNb](https://openreview.net/forum?id=AQo1SEElNb)
- Official paper page: [https://proceedings.iclr.cc/paper_files/paper/2026/hash/113e6f1d94af5df4f306fbcb3f82339f-Abstract-Conference.html](https://proceedings.iclr.cc/paper_files/paper/2026/hash/113e6f1d94af5df4f306fbcb3f82339f-Abstract-Conference.html)
- Reviewer handles are the public OpenReview pseudonyms; no attempt is made to identify individuals.
- Source: public OpenReview review dump; fields are preserved as released, including review text, rebuttal comments, meta-review, and decision where available.

## Paper Abstract

Positional information is essential for language modeling. Softmax Transformers with Rotary Position Embeddings (RoPE) encode it with fixed-angle rotations, while linear Transformers rely on input-dependent gates that only decay past key-value norms. We provide a theoretical argument for the necessity of a rotation and decay component in well-performing sequence models, and observe that the missing ingredient in linear models is precisely the rotation that softmax attention performs implicitly. We introduce Selective Rotary Position Embedding (*Selective RoPE*), an input-dependent, learnable rotary embedding that generalizes RoPE to arbitrary angles and composes seamlessly with decay gates. Equipping gated linear attention with *Selective RoPE* yields a complex-valued recurrent layer that can be implemented efficiently with the “RoPE trick”. On synthetic benchmarks (MQAR, copying, state tracking) and 370M-parameter language-model pre-training, the method improves recall, downstream accuracy, and expressivity while adding minimal architectural overhead. We open-source our implementation [here](https://github.com/timurcarstensen/selective-rope).

## Review Inventory (4 Official Reviews, 5 Discussion/Comment Notes)

- Final decision: **Accept (Poster)**
- Official review ratings (review order): `4, 6, 4, 4`

## Official Reviews

### Official_Review — Reviewer_Rgdu

- Note ID: `ZDa85mPL54`
- Discussion number: `1`
- Invitation: `ICLR.cc/2026/Conference/Submission21436/-/Official_Review`


#### Summary

The authors generalise RoPE to a mechanism allowing it to choose angles in a way that is input dependent. The authors perform analysis mainly on linear-attention models and show that selective rope seems to improve performance over baselines such as RoPE or NoPE.

#### Soundness

`2`

#### Presentation

`2`

#### Contribution

`2`

#### Strengths

The connection between SSMs, linear attention, and RoPE is interesting. I particularly liked the presentation in Table 1.

#### Weaknesses

My main area of research is in Transformers and not linear attention although I have some experience with linear attention and RFF.s 

I do not think I quite understand the point of "Softmax attention implicitly applies a selective rotation, to encode relative positional information between tokens." Are you arguing that the rotations come from the relationship between the softmax kernel and RFF? So you can view the softmax kernel as applying RoPE but where the angles are sampled IID from a Gaussian. This however would really only be true if your angle samples tend to infinity of course. Is this how you are connecting RoPE with a "NoPE" softmax?

I found the notation slightly hard to follow especially as someone not coming from SMMs. I mainly found confusing that RoPE is really a method used in Transformers, but the paper seems to only implemented the selective mechanism for linear attention and was not implemented for normal quadratic Transformers? Is there something stopping you from implementing for a normal Transformer? 

Minor
Typo in abstract: rotation in all angels -> rotation in all angles

#### Questions

Please see questions in the weaknesses

#### Flag For Ethics Review

- No ethics review needed.

#### Rating

`4`

#### Confidence

`3`

#### Code Of Conduct

Yes

### Official_Review — Reviewer_uVEP

- Note ID: `6H5GN7zbKm`
- Discussion number: `2`
- Invitation: `ICLR.cc/2026/Conference/Submission21436/-/Official_Review`


#### Summary

This paper presents a version of RoPE with learned, input-dependent arbitrary rotations. Theoretical analysis provided indicates that softmax attention implicitly performs selective rotations, motiving the proposed architecture. Selective RoPE uses a learned linear projection and cumulative sum to produce input-dependent rotations. An analysis of diagonal SSMs is provided which shows distinct roles for the real and imaginary parts of the state matrix, motiving the incorporation of Selective RoPE with GLA to provide better memory. Experiments on language modeling and show improvements with Selective RoPE compared to RoPE and softmax attention in GLA models.

#### Soundness

`3`

#### Presentation

`3`

#### Contribution

`3`

#### Strengths

The paper gives a detailed theoretical justification for the architecture design. The analyses of softmax attention as implicit rotation and spectral leakage in SSMs may be useful to future work. Experimental evidence is provided to support claims.

#### Weaknesses

The conclusion that softmax attention applies implicit selective rotation is based on the RFF approximation and additional normalization assumptions. The paper does not prove that the resultant normalized approximation converges in the limit to softmax attention, and so this analysis may be overstating the connection. 


The real-data language modeling results in table 3 omit the RoPE condition.

#### Questions

Did you compute RoPE setting for Table 3?

#### Flag For Ethics Review

- No ethics review needed.

#### Rating

`6`

#### Confidence

`3`

#### Code Of Conduct

Yes

### Official_Review — Reviewer_zi2k

- Note ID: `QLlAawBCMK`
- Discussion number: `3`
- Invitation: `ICLR.cc/2026/Conference/Submission21436/-/Official_Review`


#### Summary

The paper proposes a selective Rotary Position Embedding (RoPE) that uses an input-dependent rotation to enhance the performance of models with Gated Linear Attention. The paper provides a theoretical analysis of how softmax attention performs a hidden form of rotation and further proposes to determine the rotation angle via a linear projection of the query. Experiments are conducted on GLA showing that selective RoPE achieves better performance than NoPE and RoPE.

#### Soundness

`2`

#### Presentation

`2`

#### Contribution

`2`

#### Strengths

* The paper is clear with summarized insights and clear figures.

* The analyses on the implicit rotation of softmax attention are interesting.

* The paper provides an in-depth analysis from the RFF perspective.

#### Weaknesses

* While the paper takes a lot of effort in the derivation of implicit selective rotation in softmax attention, the proposed method is applied to gated linear attention. Given that the derivation heavily relies on the Random Fourier Features (RFF) approximation, the practical impact of the proposed method has not been validated.

* Limited experiments. The experiments are conducted on small-scale models, which raises my concern about the stability and scalability of the proposed method. The leaned rotation angle may lead to unstable training.

#### Questions

* It would be appreciated if the authors could provide additional experimental results of applying selective RoPE to softmax attention with more details on the experiments.

* Since rotations are composable. Can the proposed method be equivalently viewed as applying a rotation to the queries and keys before the RoPE operation? What is the significance of doing so?

#### Flag For Ethics Review

- No ethics review needed.

#### Rating

`4`

#### Confidence

`3`

#### Code Of Conduct

Yes

### Official_Review — Reviewer_NYgW

- Note ID: `h6PRb8jVaZ`
- Discussion number: `4`
- Invitation: `ICLR.cc/2026/Conference/Submission21436/-/Official_Review`


#### Summary

The paper introduces Selective Rotary Position Embedding (Selective RoPE), an input-dependent mechanism designed to generalize standard Rotary Position Embeddings (RoPE) by performing rotations at arbitrary, selective frequencies.

#### Soundness

`2`

#### Presentation

`3`

#### Contribution

`2`

#### Strengths

Interesting theoretical insights, such as:
- Softmax attention implicitly applies a selective rotation, to encode relative positional information between tokens
- Linear Transformers can be enhanced by using both forgetting via real decay and rotation via imaginary gate.

#### Weaknesses

Very limited evaluation for language modeling.  Would be good to have at least RoPE as baseline (Table 3) and evaluate Selective RoPE in different settings (i.e context length)

#### Questions

Do you expect GLA  and Softmax Transformers to benefit equally from Selective RoPE?

#### Flag For Ethics Review

- No ethics review needed.

#### Rating

`4`

#### Confidence

`2`

#### Code Of Conduct

Yes

## Meta-Review

### Meta_Review — Area_Chair_f3Df

- Note ID: `WxvAn5YB8H`
- Discussion number: `1`
- Invitation: `ICLR.cc/2026/Conference/Submission21436/-/Meta_Review`


#### Summary

This paper proposes Selective Rotary Position Embedding (Selective RoPE), an input-dependent generalization of RoPE motivated by a theoretical analysis that interprets softmax attention as implicitly performing selective rotations under an RFF approximation. The method is primarily instantiated in linear attention and SSM-style models (e.g., GLA, Gated DeltaNet), and extended to a softmax Transformer with decay. The paper aims to bridge positional encoding mechanisms across softmax attention, linear attention, and SSMs through a unified rotational perspective.

All reviewers acknowledge that the theoretical analysis is interesting and non-trivial, particularly the connections drawn between softmax attention, RFFs, implicit rotations, and the real/imaginary components of state transitions. Several reviewers note that these insights could be valuable beyond the specific method proposed.

The rebuttal substantially strengthens the empirical evaluation. In response to concerns about limited experiments and missing baselines, the authors add comprehensive language-modeling results across three architectures (GLA, Gated DeltaNet, and Transformer with decay), consistently comparing NoPE, RoPE, and Selective RoPE. These results show consistent but modest improvements in perplexity and downstream accuracy, addressing a major concern raised by multiple reviewers. The extension of Selective RoPE to a softmax-based Transformer also resolves ambiguity about whether the method is restricted to linear attention models.

However, one central concern remains insufficiently resolved. Multiple reviewers explicitly raised training stability and scalability issues, noting that learned, input-dependent rotations could introduce optimization instabilities, especially at higher learning rates or larger scales. In the rebuttal, the authors acknowledge these issues and state that stability problems exist without additional mechanisms (phase gate, normalization, QK-norm), and that these additions “remedy” the observed instabilities. While the authors provide qualitative explanations and reference prior work on spectral bias, the paper does not present a clear ablation or quantitative analysis that isolates stability effects, such as learning-rate sensitivity curves, failure cases, or comparisons showing when and why instability arises and how each component addresses it. The promised large-scale (1.3B) results and stability analyses are not yet available, leaving this concern partially speculative.

Overall, the work presents a conceptually interesting and reasonably well-executed idea, and the rebuttal meaningfully improves the experimental coverage. At the same time, the lack of a clean, explicit ablation focused on optimization stability and the reliance on assumptions in the softmax–RFF connection prevent the paper from fully substantiating its claims at scale. I would love to give a borderline accept with conditioned that the authors should clearly elaborate the stability ablation in the paper.

#### Reviewer Concerns

The rebuttal and revised manuscript addressed most of the major concerns raised by the reviewers. The primary issue, shared by several reviewers, was the limited empirical evaluation and the absence of RoPE baselines. This was directly resolved through substantially expanded experiments that include RoPE and No Position Embedding across Gated Linear Attention, Gated DeltaNet, and softmax transformers with decay. These results consistently show that Selective RoPE improves performance, addressing concerns about empirical validity and generality.

Questions about whether Selective RoPE applies beyond linear attention and state-space models were also addressed. The authors implemented and evaluated the method on softmax transformers with decay, demonstrating clear gains over both RoPE and NoPE. This resolves earlier ambiguity about the scope of the method.

Concerns regarding the evaluation of training stability of large models from learned, input-dependent rotations still remained, while the authors provided a reasonable theoretical motivation and empirical evidence that these additions stabilize training, though very large-scale results are still in progress.

Some concerns remain partially outstanding. The interpretation that softmax attention implicitly performs selective rotations still relies on Random Fourier Feature approximations and normalization assumptions rather than a formal equivalence, and large-scale validation is not yet complete. These issues are now clearly scoped and do not substantially weaken the overall contribution.

#### Reviewer Scores

Reviewer NYgW (initial: 4):
Likely to increase to 6, given that the main concern about limited language modeling evaluation and missing RoPE baselines was comprehensively addressed.

Reviewer zi2k (initial: 4):
Likely stay at 4, as stability concerns and some reservations about theory and scale may remain.

Reviewer uVEP (initial: 6):
Likely to remain at 6, as their primary concern about missing RoPE baselines and clarification of assumptions was resolved, even if some theoretical caveats persist.

Reviewer Rgdu (initial: 4):
Likely to increase to 6, given improved clarity, explicit softmax experiments, and better positioning of the RFF-based interpretation.

## Rebuttal and Discussion Comments

### Official_Comment — Authors

- Note ID: `NjQLeS7HvH`
- Discussion number: `1`
- Invitation: `ICLR.cc/2026/Conference/Submission21436/-/Official_Comment`


#### Comment

We thank the reviewer for their feedback and for considering our theoritical results **insightful and interesting**. Below we reply to the reviewers concern about limited language modeling experiments:


We conducted extensive evaluations of Selective RoPE on both the Softmax Transformer (w/ Decay) [1] and Gated DeltaNet [2], a state-of-the-art linear transformer. For all models, we also evaluated RoPE and NoPE as baselines. Across every architecture Selective RoPE consistently improves perplexity (PPL) and downstream language-modeling benchmark performance.

Results for the 370M-parameter Transformer, Gated DeltaNet, and GLA models, each evaluated with RoPE, Selective RoPE, and no positional embedding (NoPE), are summarized in the **table included in the general response** to all reviewers, which clearly shows Selective RoPE's superiority against other methods in all models.

We are in the process of evaluating Selective RoPE on 1.3B parameter versions of the considered models. We are also running evaluations at different context lengths and will include the results for the latter together with the 1.3B results. We ask the reviewers for some time since these results are very compute intensive and difficult to realize in an academic setting. We expect the results to be ready within the next week and will update our responses accordingly.


------
### References

[1] Forgetting Transformer: Softmax Attention with a Forget Gate. Lin, Z., Nikishin, E., He, X. O., & Courville, A. (2025). ICLR 2025. 

[2]  Gated Delta Networks: Improving Mamba2 with Delta Rule. Yang, S., Kautz, J., & Hatamizadeh, A. (2025) ICLR 2025.

### Official_Comment — Authors

- Note ID: `d5KFYXCFSD`
- Discussion number: `2`
- Invitation: `ICLR.cc/2026/Conference/Submission21436/-/Official_Comment`


#### Comment

We thank the reviewer for their thoughtful feedback and are pleased that our insights, analysis of softmax attention, and in-depth theoretical contributions resonated with them. Below, we address the concerns raised in the review:

**1) Training stability of Selective RoPE:** We agree with the reviewer regarding their concerns about the training stability of Selective RoPE. This is in line with our observations that a naive application of Selective RoPE can lead to instabilities at higher learning rates where they are otherwise not observed for models using NoPE or RoPE at the same learning rate. This can indeed be traced back to the difficulties of learning functions with high-frequency components using gradient descent and has been studied in the literature [3, 4]. To remedy these issues we introduced a gating term on the rotary component (*phase gate* in the manuscript) and a weight normalization on the input projection for Selective RoPE. We also found that QK-norm [5] applied *after* Selective RoPE helps training stability significantly. Adding these normalization components has remedied all of our observed stability issues. We are currently running larger scale experiments (1.3B models) and will update our response with the stability results at this scale once the training runs finish. We expect these results to become available within the next week. 

**2) Applying Selective RoPE to softmax Transformer:**  We thank the reviewer for their thoughtful suggestion to improve the breadth of our empirical results. We applied Selective RoPE to the Softmax Transformer (with Decay) [1]. Our results show that Selective RoPE significantly improves the performance of the Softmax Transformer (w/ Decay) compared to both RoPE and No Position Embedding (NoPE). Additionally, we incorporated GLA with RoPE and Gated DeltaNet [2] as new state-of-the-art baselines, and we find that Selective RoPE provides performance gains for these models as well. Results are provided in **Table as our general response to all reviewers.**


**3) Experiment on Larger scale models:** We have added two new baselines—Softmax Transformer w/ Decay and Gated DeltaNet—as state-of-the-art representations of sequence models. For all models, we evaluate RoPE, Selective RoPE, and No Position Embedding (NoPE) as consistent baselines, across which Selective RoPE shows superior performance. As mentioned in the previous point, we are currently training models at the *1.3B scale* and expect to report results within the coming week.

---
### References

[1] Forgetting Transformer: Softmax Attention with a Forget Gate. Lin, Z., Nikishin, E., He, X. O., & Courville, A. (2025). ICLR 2025. 


[2]  Gated Delta Networks: Improving Mamba2 with Delta Rule. Yang, S., Kautz, J., & Hatamizadeh, A. (2025) ICLR 2025.

[3] On the spectral bias of neural networks. N. Rahaman, A. Baratin, D. Arpit, F. Draxler, M. Lin, F. Hamprecht, Y. Bengio, and A. Courville. (2019) ICML 2019

[4] Towards a Mathematical Theory of Super-resolution. E. Candès, C. Fernandez-Granda. (2014) Communications on Pure and Applied Mathematics
 
[5] Query-Key Normalization for Transformers. A. Henry, P. Dachapally, S. Pawar, Y. Chen. (2020) Findings of EMNLP 2020

### Official_Comment — Authors

- Note ID: `VR3NtcrpKS`
- Discussion number: `3`
- Invitation: `ICLR.cc/2026/Conference/Submission21436/-/Official_Comment`


#### Comment

We thank the reviewer for their thoughtful feedback and are glad that they found our theoretical insights valuable. Below, we address the main concern raised by the reviewer, along with their specific questions:


**Concern: Clarifying the Assumptions Behind Implicit Selective Rotation in Softmax Attention**

Thank you for your attention to detail. Referring to the derivation of the linear attention with rotation at Appendix A.2, we note that the equations (16), (17), and (18) result in a recurrence with the scalar term $\exp\left(\frac{\Vert \mathbf{q}\_t\Vert\_2^2 - \Vert \mathbf{q}\_{t-1}\Vert\_2^2}{2}\right)$ in the transition matrix along with the rotation matrix. Consequently, our assumption for a fixed norm for the query would allow for this term to vanish, resulting in the rotation matrix being the only component in the transition matrix. We have multiple reasons for why this assumption makes sense: 

1. The normalization of queries and keys is actually a common practice in softmax attention. Namely, OLMO and Kimi-K2 both incorporate the QK-Norm idea introduced by [1] in their model for better stability. This idea was first introduced in [2].

2. There is evidence in the literature that the norm of the query has an extremely sharp distribution over sequence, especially when compared to the norm of the key [3]. In order to further test this claim, we have performed our own experiments, the results of which you can observe [here](https://drive.google.com/drive/folders/1Sm0KnlHpds_bUrKu3MSasTLpY3hN12yP?usp=sharing). In this experiment, we trained a 24 layer transformer with 24 heads (340M parameters) on 15B tokens, and plotted the norm of the query over a sequence of size 1024. As we observe, the norm of the query remains consistently in a small neighborhood, effectively looking constant. 
Regarding the norm of the key, from (18) we observe that we can absorb the norm of the key into the value vector $\mathbf{v}_t$. Consequently, the assumption over the norm of the key was solely made for the purpose of keeping the mathematical derivation simple and easy to follow.


**Question: Adding RoPE to Table 3**
We thank the reviewer for their suggestion. We have included RoPE as a baseline in our language modeling experiments and show that **Selective RoPE outperforms** standard RoPE for GLA. 

|Model|LMB.(ppl↓)|LMB.(acc↑)|PIQA(acc↑)|Hella.(acc_n↑)|Wino.(acc↑)|ARC-e(acc↑)|ARC-c(acc_n↑)|Avg.|
|-|-|-|-|-|-|-|-|-|
|**GLA**|||||||||
|NoPE|**19.21**|**39.4**|*69.7*|**48.0**|53.1|50.9|24.6|*47.6*|
|RoPE|23.96|36.1|*69.7*|47.7|**54.0**|50.9|*25.1*|47.2|
|Selective RoPE|*21.16*|*37.4*|**70.6**|*47.9*|*53.9*|**52.0**|**26.2**|**48.0**|

Moreover, we have extended our experiments to include Gated DeltaNet and Transformer (with decay), and demonstrate that Selective RoPE consistently outperforms RoPE across all these models as shown in **general response to all reviewers**, highlighting its broader applicability and potential.


--------
### References
[1] Scaling Vision Transformers to 22 Billion Parameters. M. Dehghani, J. Djolonga, B. Mustafa, P. Padlewski, J. Heek, J. Gilmer, A. Steiner, M. Caron, R. Geirhos, I. Alabdulmohsin. (2023) ICML 2023.

[2] Query-Key Normalization for Transformers. A. Henry, P. R. Dachapally, S. S. Pawar, and Y. Chen. (2020) Findings of EMNLP 2020.

[3] TRAMS: Training-free Memory Selection for Long-range Language Modeling. H. Yu, C. Wang, Y. Zhang, and W. Bi. (2023) Findings of EMNLP 2023.

### Official_Comment — Authors

- Note ID: `d62ExdjIRy`
- Discussion number: `4`
- Invitation: `ICLR.cc/2026/Conference/Submission21436/-/Official_Comment`


#### Comment

We thank the reviewer for their positive feedback and for finding our theorem and presentation interesting. Below, we clarify the question regarding the connection between softmax transformers and random rotations.

**Question on softmax attention implicitly applies a selective rotation**

The reviewer’s understanding is correct. Our theorem relies on approximating softmax using Random Fourier Feature (RFF) kernels and shows that softmax attention can be interpreted as applying random rotations, where the angles are sampled i.i.d. from a Gaussian distribution. In the limit as the number of random features goes to infinity, this approximation converges exactly to the softmax kernel. This interpretation motivates our design of **Selective RoPE** for linear transformers and SSMs, with the goal of boosting their performance and reducing the performance gap with standard softmax transformers. We also provide empirical evidence for this formulation to help improve the performance of softmax attention, which you can observe in the new version of the paper.

**Applying Selective RoPE to softmax Transformer**

We thank the reviewer for the thoughtful suggestion. We have applied Selective RoPE to the softmax transformer (w. Decay) and the results are shown below:

|Model|LMB.(ppl↓)|LMB.(acc↑)|PIQA(acc↑)|Hella.(acc_n↑)|Wino.(acc↑)|ARC-e(acc↑)|ARC-c(acc_n↑)|Avg.|
|-|-|-|-|-|-|-|-|-|
|**Transformer** (w. Decay)||||||||
|NoPE|26.04|37.4|*69.6*|47.0|**55.2**|50.7|*25.8*|47.6|
|RoPE|*23.16*|*37.7*|69.5|*47.6*|*55.0*|**52.7**|25.3|*48.0*|
|SelectiveRoPE| **21.89** | **38.2** | **70.2** | **47.8** | 54.1 | *52.4* | **26.1** | **48.1**|

These results show that **Selective RoPE outperforms both standard RoPE and having no positional embedding (NoPE)**.

Moreover, although RoPE is primarily used in softmax transformers, its origin lies in linear attention with an imaginary gating mechanism, as discussed in Section 2 of our paper and in the original RoPE paper [1]. Our goal is to show that Selective RoPE is rooted in softmax attention via the RFF connection, while also naturally extending to linear transformers through an input-dependent imaginary forget gate.

**Minor Typo in abstract**
Thank you for pointing out the type, we have fixed it. Generally, we have improved the clarity and readability of the manuscript as we have detailed in the general response to all reviewers. The reviewer will also find that we have unified the notation in the main body of the paper and rely on Transformer notation with $q,k,v$ instead of SSM notation.

-----

### References


[1] RoFormer: Enhanced Transformer with Rotary Position Embedding. Jianlin Su, Yu Lu, Shengfeng Pan, Ahmed Murtadha, Bo Wen, Yunfeng Liu. 2024 Neurocomputing

### Official_Comment — Authors — General Response to the Reviewers

- Note ID: `e6zc4CnQsz`
- Discussion number: `5`
- Invitation: `ICLR.cc/2026/Conference/Submission21436/-/Official_Comment`


#### Title

General Response to the Reviewers

#### Comment

## General Response

We thank all the reviewers for their constructive feedback on our manuscript. As requested by several reviewers, we have extensively expanded our language-modeling experiments by including Gated DeltaNet (GDN) and the Softmax Transformer (w/ Decay). We compare Selective RoPE against RoPE and No Position Embedding across all models—GLA, GDN, and the Transformer w/ Decay (FoX)—and show that Selective RoPE consistently and significantly boosts performance for all models. 

|Model|LMB.(ppl↓)|LMB.(acc↑)|PIQA(acc↑)|Hella.(acc_n↑)|Wino.(acc↑)|ARC-e(acc↑)|ARC-c(acc_n↑)|Avg.|
|-|-|-|-|-|-|-|-|-|
|**GLA**|||||||||
|NoPE|*23.15*|**39.4**|*69.7*|**48.0**|53.1|*50.9*|24.6|*47.6*|
|RoPE|23.96|36.1|*69.7*|47.7|**54.0**|*50.9*|*25.1*|47.2|
|Selective RoPE|**21.16**|*37.4*|**70.6**|*47.9*|*53.9*|**52.0**|**26.2**|**48.0**|
|**Gated DeltaNet**|||||||||
|NoPE|22.50|37.2|**70.9**|*47.6*|53.2|*52.0*|**25.9**|47.8|
|RoPE|*20.84*|*38.9*|*70.7*|**48.2**|*53.4*|51.3|25.1|*48.0*|
|Selective RoPE|**19.28**|**39.4**|70.1|*47.6*|**54.9**|**52.4**|*25.4*|**48.3**|
|**Transformer** (w/ Decay)||||||||
|NoPE|26.04|37.4|*69.6*|47.0|**55.2**|50.7|*25.8*|47.6|
|RoPE|*23.16*|*37.7*|69.5|*47.6*|*55.0*|**52.7**|25.3|*48.0*|
|Selective RoPE|**21.89**|**38.2**|**70.2**|**47.8**|54.1|*52.4*|**26.1**|**48.1**|

Moreover, we have included several new experimental results: 

- An **ablation of the additional architectural components introduced in Selective RoPE** (phase gate and bias) on the MAD benchmark, showing that the core Selective RoPE mechanism already achieves  gains over NoPE/RoPE and that the variant with both phase gate and bias attains the best overall MAD average while preserving improvements across all MAD tasks.
- A corresponding ablation of these components in the language-modeling setup, where we systematically vary the presence of the phase gate and bias for GLA, GDN, and the Transformer w/ Decay (FoX in the manuscript); this confirms that Selective RoPE is robust across architectures, that the phase gate mainly helps optimization stability and downstream accuracy, and that adding only a bias does not yield consistent additional gains.
- An **efficient Triton implementation of Selective RoPE**, demonstrating that our fused kernel is almost as fast as RoPE/NoPE in prefill throughput and achieves up to a 3.4× speedup over the PyTorch-compile implementation at long sequence lengths, thereby addressing concerns about the runtime overhead of Selective RoPE. We will publish our implementation after the closure of the review phase.


We would also like to note that we have considerably improved the writing and general presentation of the manuscript. The content of the paper has remained unchanged but the readability has improved significantly. The primary changes are: 

1. Consolidating the motivation for Selective RoPE and the description of the method in Section 3 (before: Sections 3 through 5). 
2. Moving implementation details into their own subsection in the prelude to the experiments in Section 4.
3. Addition of a related work section (Section 5)

## Decision

### Decision — Program_Chairs — Paper Decision

- Note ID: `d7ScXBogCW`
- Discussion number: `1`
- Invitation: `ICLR.cc/2026/Conference/Submission21436/-/Decision`


#### Title

Paper Decision

#### Decision

Accept (Poster)
