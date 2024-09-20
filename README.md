# dataset_tools

自製全自動打標all in one  

需求python 3.10 3.11 windows

整合 florence-2 自然語言、WD14 tagger、clip score (long clip)、aesthetic predictor v2.5美學模型

+自定義打標流程


直接抓caption.py下來 (不要用git clone)

python caption.py "資料集位置" 

就可以跑了


py後面可以放args (中間要空格)

--folder_name 前置角色名 "aaa appearance," OR "two preson include aaa appearance and bbb appearance" ... ___

--not_char 如果前置是概念不是角色 "aaa in the image, " ... ___

--clothtag 前置衣服標(當單人時) ", with black tank top, grey shorts" ... ___

--peoplotag 前置多人動作標(當雙人時) ", back-to-back, after_kiss" ... ___

--drop_colortag刪除WD14顏色標 del "black hair" "brown eyes"...  (前置與florence-2打的會保留)

=====

--upgrade 升級腳本，有需要才用

--drop_chartag 自動刪除角色特徵標 如果用wildcard應該是不需要


打出來格式是三行wildcard，___ 之前是前置

前置, 全部標排序___

accurate, 前置, ___少的標

inaccurate, 前置, ___更少的標


前置是 "排除標" "概念名?" "人數" "角色名?" "衣服?" "多人動作?" "nsfw標" "美學標"


在kohya-ss 中Additional parameters填上

--enable_wildcard 使用多行wildcard，不然只會用第一行

--keep_tokens_separator="__" 固定 __之前的標


可以加上

--network_train_unet_only 只訓練UNET

打勾Shuffle caption 洗牌標籤



推薦

角色設置 python caption.py "資料集位置" --folder_name --clothtag --peoplotag --drop_colortag

概念設置 python caption.py "資料集位置" --not_char

kohya-ss Additional Parameters

--enable_wildcard --keep_tokens_separator="__" --network_train_unet_only 

洗牌 


如果有訓練效果反饋或有什麼架構的微調建議再跟我說


