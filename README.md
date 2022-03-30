# gov_ner_project

使用 [Hugging Face Hub](https://huggingface.co/models) 上的語言模型來進行 NER 模型的訓練，本專案使用ALBERT (albert-base-chinese, albert-tiny-chinese)進行模型訓練。

本專案使用open data政府組織列表資料作為NER標籤的訓練基礎，希望建立政府組織(GORG)這個新的標籤。

- [程式建置](#程式建置)
- [程式說明](#程式說明)
    - [專案流程](#專案流程)
    - [資料探索](#資料探索)
    - [資料格式轉換](#資料格式轉換)
    - [模型訓練](#模型訓練)
    - [模型驗證與結果](#模型驗證與結果)
- [Tips](#tips)
- [相關資料](#相關資料)


## 程式建置

1. Vscode
2. 確認 python 與 pip 的路徑是 Virtual environment
3. 安裝程式的相關套件

## 程式說明

### 專案流程

1. 下載政府open data的政府組織列表資料，並將政府組織名稱存入.txt檔。
2. 將學校、醫院、銀行、圖書館等同時有公私立的組織去除，以免機器訓練過程造成混淆。
3. 透過Longest Common Subsequence (LCS)演算法，擴增列表。
4. 將部分組織改為正則表達式來代表，使得地區較不會被誤判為政府組織。
5. 將原本的資料重新標籤，加入GORG這個新的標籤。
6. 使用transformer來進行模型訓練。
7. 最後進行模型驗證（confusion matrix）。

### 資料探索
 - 原始資料的tsv檔包含（id, tokens, ner_tags)
 - ner label 包含（Ｏ, B-PER, I-PER, B-ORG, I-ORG, B-LOC, I-LOC, <font color=#FF6600>B-GORG(新增), I-GORG(新增)</font>）

 - 資料概況

    標籤 | PER | LOC | ORG | GORG
    ------|:-----:|------:|------:|------:
    All | 46824 | 38356 | 71008 | 22915
    Train | 37336 | 30669 | 56937 | 18336
    Dev | 4805 | 3835 | 7009 | 2310
    Test | 4683 | 3852 | 7062 | 2269

### 資料格式轉換
 - 有別於英文版本以word為單位的切割，中文版本使用character來切割，因此在訓練模型前就已經將句子分割完成。(由於tokenizer會將空白自動移除，因此可以透過`[MASK]`來取代空白)
 - 將原本的tsv檔案轉存為json檔案，以便後續進行機器學習。
 - 檔案分為 train, validation, test 資料量為 8:1:1

### 模型訓練
 - 使用主機訓練模型，建立設定檔 config.json 包含以下參數
    - "model_name_or_path": "ckiplab/albert-base-chinese" ,
    - "tokenizer_name" : "bert-base-chinese",
    - "train_file": "訓練集",
    - "validation_file" : "驗證集",
    - "test_file": "測試集",
    - "label_file": "./label.json",
    - "task_name" : "ner",
    - "do_train" : true,
    - "do_eval" : true,
    - "do_predict" : true,
    - "metric_for_best_model": "eval_f1",
    - "load_best_model_at_end" : true,
    - "overwrite_output_dir" :true,
    - "output_dir": "./model/best_model_0329_1",
    - "learning_rate": 5e-4,
    - "eval_steps": 500,
    - "save_steps": 500,
    - "logging_steps": 500,
    - "evaluation_strategy": "steps",
    - "num_train_epochs": 3.0,
    - "per_device_train_batch_size": 8,
    - "per_device_eval_batch_size": 8,
    - "save_total_limit": 10,
    - "warmup_ratio": 0.1,
    - "preprocessing_num_workers":2,
    - "logging_first_step": true,
    - "ignore_data_skip": true,
    - "return_entity_level_metrics": true

- Wandb設定的補充說明如下 :
    - wandb_api_key : 新增 .env 檔案, 並新增 WANDB_API_KEY 變數值(在wandb的帳號中索取)
    - wandb_project_name : 專案名稱, 包含每次的訓練結果
    - wandb_tags : 透過標籤，紀錄該次訓練時的一些變數, 標籤可用來進行不同訓練結果的分組

- 使用[colab](https://colab.research.google.com/github/huggingface/notebooks/blob/master/examples/token_classification.ipynb)進行模型訓練 :
    - transformers 有將部分常用模型寫成colab可以執行的版本，如此一來就可以使用免費的GPU來進行訓練。
    - 依照colab上指示進行套件安裝，如果要使用albert-base, bert-base等模型需要多輸入`!pip install -U transformers`這個指令，才能訓練成功。
- 如果使用模型內建的seqeval matrics，要注意在假設沒有 B 標籤時, 程式碼會把第一個 I 標籤換成 B 標籤，會與另外自己將 B 和 I 的標籤合併時的precision和recall有落差。 

### 模型驗證與結果

驗證過程可以使用transformers所提供的pipeline套件或是自行撰寫符合pytorch or tensorflow的程式碼來預測。

1. ckiplab/albert-tiny-chinese 訓練結果

    標籤 | PER | LOC | ORG | GORG
    ------|:-----:|------:|------:|------:
    precision | 0.8888 | 0.7195 | 0.7552 | 0.8041
    recall | 0.8973 | 0.7292 | 0.7933 | 0.8682
    f1 | 0.8930 | 0.7243 | 0.7738 | 0.8349

    overall_accuracy:0.9581

2. ckiplab/albert-base-chinese 訓練結果

    標籤 | PER | LOC | ORG | GORG
    ------|:-----:|------:|------:|------:
    precision |0.9237  |0.7590  |0.7918  | 0.8501
    recall | 0.9441 |0.7947  | 0.8508 | 0.8722
    f1 | 0.9338 |0.7764  |0.8202  | 0.8610

    overall_accuracy:0.9667

#### 補充資料
- 所有資料集中重複10次以上的標籤佔比

標籤 | PER | LOC | ORG | GORG
------|:-----:|------:|------:|------:
all_unique | 22767 | 8529 | 29624 | 6049
all_occur | 44355 | 35804 | 70591 | 22915
\>10_unique | 416 | 327 | 843 | 396
\>10_occur | 8917 | 23379 | 18823 | 11125
\>10_occur/all_occur | 0.2010 | 0.6530 | 0.2666 | 0.4855

- 可以看到地點(LOC)和政府組織(GORG)中出現的詞彙大多都由屬於少部分的詞彙，因此應該要將重複10次以上的標籤去除，來看precision和recall會比較準確。

## Tips

- 訓練注意事項 :
    1. 訓練資料內的空白字元不會被訓練，因此資料預處理時會忽略空白字元的 Token
        - 預測時可以透過`[MASK]`來取代空白字元的 Token
    2. 當調高 Epoch數時, 可以使得 Evaluation/F1 上升, 但 Evaluation/Loss 也會因此上升，需要留意。
    3. Batch size 建議使用3進行訓練, （也可以嘗試4 or 5），但當提升batch size時Evaluation/Loss 也會因此上升。
    4. 使用pipeline預測時，會造成英文字母無法切分成characters的情況，像是berlin 會分成 "be", "##rlin", 目前pipeline尚未提供解方，因此雖然pipeline比較方便使用，但並不適合在同時含有中英文的文章時使用。
    5. 使用模型的 Tokenizer 需要帶上 `is_split_into_words=True`, 且將訓練資料的每一個字都視為一個 Token，來進行預測和訓練。

- colab 使用需要注意的地方
    1. 如果要使用albert-base, bert-base等模型需要多輸入`!pip install -U transformers`這個指令，才能訓練成功。
    2. 參數的部分不需要有一個檔案，直接在Notebook上更改就行。
    3. 登入Hugging Face可以直接將結果和參數git到Hugging Face，且可以使用Hugging Face的API，將模型視覺化。
- NER 準確度存在迷思 :
  - 準確度高可能是被少數詞彙撐高的, 因為這些詞彙的標記數量多, 於是準確度高
  - 因此可以將那些常出現的詞彙先踢除，來看precision和recall會必較準確

## 相關資料

- [ckiplab/ckip-transformers](https://github.com/ckiplab/ckip-transformers)
- [Fine-tuning a model on a token classification task](https://github.com/huggingface/notebooks/blob/master/examples/token_classification.ipynb)
- [Hugging face Pipelines](https://huggingface.co/docs/transformers/main_classes/pipelines)
- [Hugging face Tokenizer](https://huggingface.co/docs/transformers/main_classes/tokenizer)