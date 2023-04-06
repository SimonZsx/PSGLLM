# Flops Calculation


## Matrix Multiplication 


A m x k × X 𝑘 x n  matrix multiplication requires 2 m x k × n FLOPs
(factor of 2 needed to account for multiplies and adds)

```
Batch size B
Sequence length s
Hidden dimension h 
Vocabulary V
```
## Embedding 

```bash
6V/lh
```
## Attention 


Key, Query, Value transformation (6Bsh^2 operations), attention matrix computation (2Bs^2h operations), attention over values (2Bs^2h operations), and post-attention
linear projection (2Bsh^2 operations). 

## Feed-forward

The feed-forward network increases the hidden size to 4h and then reduces it back to h; this requires 16Bsh^2 FLOPs. 

## Sum

forward: 24Bsh^2 + 4Bs^2h 
backward: 2 times of forward as it computes partial gradients w.r.t. weights and inputs. 
recmpute: 1 forward 

96Bsh^2 + 16Bs^2h

# Memory calculation 


```bash
P = 12 * lh^2 (1 + 13/(12h) + (V+s)/12lh)
```


