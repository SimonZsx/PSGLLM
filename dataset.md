#Dataset collections 


### Wikipedia training. 


step1. 

```bash 
wget https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2
```

step2.

Use the https://github.com/attardi/wikiextractor

We recommend using the --json argument when using WikiExtractor, which will dump the Wikipedia data into loose json format (one json per line), making it more manageable on the file system and also readily consumable by our codebase. We recommend further preprocessing this json dataset by nltk punctuation standardization. For BERT training, use the --split-sentences flag to preprocess_data.py as described above to include sentence breaks in the produced index. If you'd like to use Wikipedia data for GPT training you should still clean it with nltk/spacy/ftfy, but do not use the --split-sentences flag.

Refer to https://github.com/microsoft/Megatron-DeepSpeed#gpt-3-example


### UltraChat - Large-scale, Informative, and Diverse Multi-round Dialogue Data' THUNLP 

GitHub: github.com/thunlp/UltraChat

### awesome-instruction-tuning(ChatGPT|LLaMA)-dataset - A collection of open-source dataset to train instruction-following LLMs (ChatGPT,LLaMA,Alpaca) dongdong 

GitHub: github.com/yaodongC/awesome-instruction-dataset

### 'Awesome-instruction-tuning - A curated list of awesome instruction tuning datasets, models, papers and repositories.' zhilizju 

GitHub: github.com/zhilizju/Awesome-instruction-tuning
