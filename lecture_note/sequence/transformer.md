[<-PREV](sequence.md)

# Transformer 

## 1. Intuition
![image](images/image1.png)

## 2. Attention mechanism and Transformer's architecture

![image](images/equation1.png)
![image](images/image2_2.png)

### 3. Pre-training Transformer models
![image](images/image3.png)
- Many more options at [https://huggingface.co/transformers/pretrained_models.html](https://huggingface.co/transformers/pretrained_models.html)


## 4. BERT (Bidirectional Encoder Representations from Transformers)
### 4.1 Text preprocessing - Tokenization
![image](images/image4.png)

![image](images/image5.png)

### 4.2 Pretraining BERT
![image](images/image6.png)
1. Pre-training: Semi-supervised training on larget amounts of text
    - Masked Language Model (MLM)
    - Next Sentence Prediction (NSP) 
2. Fine-tuning: Supervised training on a specific task with a labeled dataset 

### 4.3 How to use BERT for classification
![image](images/image7.png)
1. Use a [CLS] token (Context vector) only
2. Use last_hidden_states and Pooling
3. Use last_hidden_states as input for RNN model


[<-PREV](sequence.md)
