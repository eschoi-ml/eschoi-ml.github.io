[<-PREV](toxiccomment.md)[NEXT->](toxiccomment-part1.md)

# Introduction

## 1. Original data overview: Wikipedia corpus

Wikipedia corpus dataset was used in this project. It comprises 63M comments from discussions relating to user pages and articles dating from 2004-2015. 
Each commet was tagged by human raters for toxicity in the following six categories

- toxic
- severe_toxic
- obscene
- threat
- insult
- identity_hate

The tagging was done via crowdsourcing which means that the dataset was rated by different people and the tagging might not be 100% accurate too.

## 2. Create new data by reorganizing
In this project, I reorganized the dataset as binary classes with clean vs. Toxic comments by combining the six categories.
Because the original data was tagged subjectively by multiple people and the majority toxic comments were multi-tagged, the dataset resulted in high imbalance between clean and Toxic comments or among Toxic comments. Thus, instead of following the Kaggle challenge, I thought it was more reasonable to reorganize the data as a binary classification problem. Note that the clean data was also sampled so that the size of clean and toxic data was matched. 
