# TIGER implementation

## 📚Dataset

The data used in this project comes from the paper [NineRec](https://arxiv.org/pdf/2309.07705).Data can be downloaded
from this [link](https://drive.google.com/file/d/15RlthgPczrFbP4U7l6QflSImK5wSGP5K/view). \
Specifically, __Bili_Cartoon__ from __Downstream Dataset__ is used in this project. After downloading, put it into the
folder `./data/Bili_Cartoon`

## 🪄Large Language Model

T5-base is chosen for this project, it can be downloaded
from [here](https://huggingface.co/google-t5/t5-base/tree/main).

After downloading, create a folder __LLM__ at the same level as the project, and create a subfolder named __t5-base__
under the folder. Put the downloaded large language model into the subfolder.

## 💡What improved in this project

This project implements four kinds of ID:<br />

-**SID from text**: Using SentenceT5 convert item title and description(concatenate) to get semantic embedding. Then training RQ-VAE to get SID<br />
-**SID from pretrained model**: As same as **SID from text**, but replacing semantic embedding with embedding table of pretrained sequential model(SASRec from RecBole)<br />
-**SID from text without conflict**: In **SID from text** doesn't solve hash conflict problem, this variant adds additional token to the end of SID(the same as TIGER)<br /> 
-**CID(chunked ID)**: Using k-base to get CID, which described in [MBGen](https://arxiv.org/pdf/2405.16871)


## 🚀Quick Start

___Step 1: generating text embedding___.

```bash
python ./data/generate_text_embed.py
```

___Step 2: processing interaction data___

```bash
python ./data/preprocess.py
```

___Step 3: item tokenization___

```bash
python ./tokenizer/main.py
```

___Step 4: supervised fine-tuning LLM___

```bash
python sh train.sh
```

___Step 5: evaluation___

```bash
python sh evaluation.sh
```

## 🙇Acknowledgement

Most of the code in this project references from [LC-Rec](https://arxiv.org/pdf/2311.09049). And I make some
improvements and add some comments.