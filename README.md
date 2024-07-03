# dataset_tools

自製全自動打標all in one  

整合 florence-2 自然語言、WD14 tagger、clip score (long clip)、aesthetic predictor v2.5美學模型

+自定義打標流程


直接抓caption.py下來 python caption.py "資料集位置" 

就可以跑了


py後面可以放args (中間要空格)

--folder_name 前置角色名 "aaa," OR "two preson include aaa and bbb" ... ___

--not_char 如果前置是概念不是角色 "aaa in the image, " ... ___

--clothtag 前置衣服標(當單人時) ", with black tank top, grey shorts" ... ___

--peoplotag 前置多人動作標(當雙人時) ", back-to-back, after_kiss" ... ___

--drop_colortag刪除WD14顏色標 del "black hair" "brown eyes"...  (前置與florence-2打的會保留)



其他比較不會用到的

--continue_caption="天數整數" 從n天內打的標繼續

--rawdata 只打一行 也許正則圖這樣做比較好 不確定沒測試

--debiased 設置clip score上限減少florence偏差，如果發現florence很多作品名、角色幻覺可以用 不然不需要 會刪掉一些正常的標 

--custom_keeptag="字串類似is doing" 自定義前置，實驗性很慢效果差、不要用

--upgrade 升級腳本，有需要才用


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


