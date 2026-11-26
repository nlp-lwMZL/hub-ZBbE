# 多模型并排 case 表

> 每个候选取一条，展示 gold 与 6 臂在同一句上的预测 span。

## [all_correct] cdtier_clean_ner_test_00056

**文本**：通过360安全大脑我们还原捕获到了Kimsuky组织的部分攻击样本和完整攻击流程

**gold**：Attacker[17:24]=Kimsuky

| 方法 | 预测 span |
|---|---|
| bilstm_linear | Attacker[17:24]=Kimsuky |
| bilstm_crf | Attacker[17:24]=Kimsuky |
| roberta_linear | Attacker[17:24]=Kimsuky |
| roberta_crf | Attacker[17:24]=Kimsuky |
| qwen_lora_sft | Attacker[17:24]=Kimsuky |
| qwen_full_sft | Attacker[17:24]=Kimsuky |

## [all_correct] cdtier_clean_ner_test_00082

**文本**：主要针对领域有政府机构、军工企业、核能行业等

**gold**：Industry[7:11]=政府机构, Industry[12:16]=军工企业, Industry[17:21]=核能行业

| 方法 | 预测 span |
|---|---|
| bilstm_linear | Industry[7:11]=政府机构, Industry[12:16]=军工企业, Industry[17:21]=核能行业 |
| bilstm_crf | Industry[7:11]=政府机构, Industry[12:16]=军工企业, Industry[17:21]=核能行业 |
| roberta_linear | Industry[7:11]=政府机构, Industry[12:16]=军工企业, Industry[17:21]=核能行业 |
| roberta_crf | Industry[7:11]=政府机构, Industry[12:16]=军工企业, Industry[17:21]=核能行业 |
| qwen_lora_sft | Industry[7:11]=政府机构, Industry[12:16]=军工企业, Industry[17:21]=核能行业 |
| qwen_full_sft | Industry[7:11]=政府机构, Industry[12:16]=军工企业, Industry[17:21]=核能行业 |

## [long_text] cdtier_clean_ner_test_00017

**文本**：附录IOCIPDomainil.**..plugin-verifica**.attachment-download.services.enc2globaattachment-download.**-stest.enc2globaauthentication-services.email-smtp-protocol-update-notification.saMD58ad47895f3af1f06d

**gold**：Tools[799:801]=双杀, Tools[802:804]=双星, Tools[805:809]=噩梦公式

| 方法 | 预测 span |
|---|---|
| bilstm_linear | （空） |
| bilstm_crf | （空） |
| roberta_linear | （空） |
| roberta_crf | （空） |
| qwen_lora_sft | （空） |
| qwen_full_sft | （空） |

## [long_text] cdtier_clean_ner_test_00051

**文本**：DomainIOCURLhttp:///ln.corehttp:///ln/a.asphttp:///dota3.tar.gzhttp:///a.jsphttp:///aa.jsphttp:///report.jsphttp:///ipc.jsphttp:///ipco.jsphttp:///ms.jsphttp:///mso.jsphttp:///rdp.jsphttp:///rdpo.jsph

**gold**：

| 方法 | 预测 span |
|---|---|
| bilstm_linear | （空） |
| bilstm_crf | （空） |
| roberta_linear | （空） |
| roberta_crf | （空） |
| qwen_lora_sft | （空） |
| qwen_full_sft | （空） |

## [campaign] cdtier_clean_ner_test_00033

**文本**：利用社会工程学方案攻击是Lazarus的显著特征之一，在近两年的时间内，该组织进行了多起类似方案的攻击活动，譬如“DreamJob”、“针对安全研究人员的攻击”等

**gold**：Attacker[12:19]=Lazarus, Campaign[57:65]=DreamJob, Campaign[68:79]=针对安全研究人员的攻击

| 方法 | 预测 span |
|---|---|
| bilstm_linear | Attacker[12:19]=Lazarus, Attacker[58:65]=reamJob |
| bilstm_crf | Attacker[12:19]=Lazarus, Attacker[58:65]=reamJob |
| roberta_linear | Attacker[12:19]=Lazarus, Attacker[56:64]=“DreamJo, Industry[70:71]=安 |
| roberta_crf | Attacker[12:19]=Lazarus, Attacker[56:64]=“DreamJo, Industry[70:71]=安 |
| qwen_lora_sft | Attacker[12:19]=Lazarus |
| qwen_full_sft | Attacker[12:19]=Lazarus, Campaign[57:65]=DreamJob, Campaign[68:76]=针对安全研究人员 |

## [campaign] cdtier_clean_ner_test_00145

**文本**：但在血茜草活动中，该诱骗活动与钓鱼网站活动同时进行的情况，在业界并不多见

**gold**：Campaign[2:5]=血茜草

| 方法 | 预测 span |
|---|---|
| bilstm_linear | （空） |
| bilstm_crf | （空） |
| roberta_linear | Attacker[2:5]=血茜草 |
| roberta_crf | Attacker[2:5]=血茜草 |
| qwen_lora_sft | Campaign[2:5]=血茜草 |
| qwen_full_sft | Campaign[2:5]=血茜草 |

## [qwen_only] cdtier_clean_ner_test_00141

**文本**：同时奇安信威胁情报中心ALPHA平台(ti.qianxin.com)对此次攻击活动相关IOC已有APT-C-23标签

**gold**：Attacker[48:56]=APT-C-23

| 方法 | 预测 span |
|---|---|
| bilstm_linear | （空） |
| bilstm_crf | （空） |
| roberta_linear | （空） |
| roberta_crf | （空） |
| qwen_lora_sft | Attacker[48:56]=APT-C-23 |
| qwen_full_sft | Attacker[48:56]=APT-C-23 |

## [qwen_only] cdtier_clean_ner_test_00163

**文本**：图17钓鱼邮件3.2恶意宏代码执行图18恶意的邮件附件恶意代码被加密保存在诱饵文档excel表格的D500单元格的批注中（见图19）

**gold**：Tools[41:46]=excel

| 方法 | 预测 span |
|---|---|
| bilstm_linear | （空） |
| bilstm_crf | （空） |
| roberta_linear | （空） |
| roberta_crf | （空） |
| qwen_lora_sft | （空） |
| qwen_full_sft | Tools[41:46]=excel |
