# NER 错误分析报告

> 从 6 臂逐条预测离线统计；span=(type,start,end)，匹配看字符重叠 IoU。
> 目的：回答『模型是分不清类型，还是定不准边界』——这是选改进方向的关键。

## 1. 错误分解（gold span 的去向）

| 方法 | TP | 类型错(边界对) | 边界错(类型对) | 都错 | 漏召 | 多预测 | TP率 |
|---|---:|---:|---:|---:|---:|---:|---:|
| bilstm_linear | 67 | 1 | 139 | 17 | 208 | 101 | 0.1551 |
| bilstm_crf | 70 | 1 | 113 | 14 | 234 | 48 | 0.162 |
| roberta_linear | 215 | 6 | 98 | 19 | 94 | 144 | 0.4977 |
| roberta_crf | 232 | 8 | 101 | 12 | 79 | 163 | 0.537 |
| qwen_lora_sft | 231 | 7 | 18 | 1 | 175 | 129 | 0.5347 |
| qwen_full_sft | 267 | 6 | 21 | 2 | 136 | 135 | 0.6181 |

读法：
- 类型错高 → 标签体系/相似类型混淆（如 Attacker vs Tools）；
- 边界错高 → span 边界定义或分词对齐问题（老师『只差一个股』就是这类）；
- 漏召高 → 召回不足（长尾类/截断）；多预测高 → 精确率问题。

## 2. 类型混淆 Top（gold → pred）

### bilstm_linear

| gold | pred | 次数 |
|---|---|---:|
| Tools | Attacker | 7 |
| Attacker | Tools | 4 |
| Region | Industry | 2 |
| Industry | Region | 2 |
| Industry | Attacker | 2 |
| Campaign | Attacker | 1 |

### bilstm_crf

| gold | pred | 次数 |
|---|---|---:|
| Tools | Attacker | 6 |
| Attacker | Tools | 4 |
| Region | Industry | 2 |
| Campaign | Attacker | 1 |
| Industry | Attacker | 1 |
| Industry | Region | 1 |

### roberta_linear

| gold | pred | 次数 |
|---|---|---:|
| Tools | Attacker | 6 |
| Attacker | Tools | 6 |
| Campaign | Attacker | 5 |
| Region | Industry | 3 |
| Campaign | Industry | 1 |
| Industry | Tools | 1 |
| Industry | Region | 1 |
| Tools | Industry | 1 |
| Attacker | Region | 1 |

### roberta_crf

| gold | pred | 次数 |
|---|---|---:|
| Campaign | Attacker | 5 |
| Tools | Attacker | 4 |
| Region | Industry | 3 |
| Industry | Region | 2 |
| Attacker | Tools | 2 |
| Campaign | Industry | 1 |
| Industry | Tools | 1 |
| Tools | Industry | 1 |
| Attacker | Region | 1 |

### qwen_lora_sft

| gold | pred | 次数 |
|---|---|---:|
| Tools | Attacker | 2 |
| Attacker | Campaign | 2 |
| Region | Attacker | 1 |
| Attacker | Tools | 1 |
| Industry | Region | 1 |
| Tools | Region | 1 |

### qwen_full_sft

| gold | pred | 次数 |
|---|---|---:|
| Region | Industry | 2 |
| Tools | Region | 2 |
| Industry | Tools | 1 |
| Region | Attacker | 1 |
| Industry | Region | 1 |
| Attacker | Tools | 1 |

## 3. 边界错的具体样子（类型对但 span 不同）

### roberta_crf

- gold `Tools[64:75]=GoogleDrive` vs pred `Tools[62:73]=x或GoogleDri`
- gold `Tools[56:63]=Dropbox` vs pred `Tools[54:61]=指向Dropb`
- gold `Attacker[9:11]=灵猫` vs pred `Attacker[8:10]=“灵`
- gold `Tools[41:50]=MoleStage` vs pred `Tools[43:46]=leS`
- gold `Industry[14:16]=军工` vs pred `Industry[14:17]=军工企`
- gold `Attacker[34:42]=KonniAPT` vs pred `Attacker[34:40]=KonniA`
- gold `Tools[6:10]=驱动人生` vs pred `Tools[5:9]=“驱动人`
- gold `Industry[134:136]=能源` vs pred `Industry[133:137]=、能源和`

### qwen_full_sft

- gold `Industry[14:16]=军工` vs pred `Industry[14:18]=军工企业`
- gold `Campaign[68:79]=针对安全研究人员的攻击` vs pred `Campaign[68:76]=针对安全研究人员`
- gold `Industry[62:64]=能源` vs pred `Industry[62:66]=能源企业`
- gold `Region[25:29]=半岛地区` vs pred `Region[25:27]=半岛`
- gold `Tools[34:43]=HallajPro` vs pred `Tools[34:46]=HallajProRAT`
- gold `Tools[84:91]=Houdini` vs pred `Tools[84:94]=HoudiniRAT`
- gold `Tools[53:59]=Xtreme` vs pred `Tools[53:62]=XtremeRAT`
- gold `Tools[72:79]=Revenge` vs pred `Tools[72:83]=Revenge-RAT`

## 4. 多模型并排 case（引子）

从下列候选 id 里挑 5~8 条，手工查 `outputs/preds/<method>_test.jsonl` 做并排表：

### all_correct
- `cdtier_clean_ner_test_00056`：通过360安全大脑我们还原捕获到了Kimsuky组织的部分攻击样本和完整攻击流程…
- `cdtier_clean_ner_test_00082`：主要针对领域有政府机构、军工企业、核能行业等…
- `cdtier_clean_ner_test_00091`：Lazarus组织在Windows和MacOS平台上对加密货币方面的用户进行针对性的攻击，而且更加隐蔽…

### long_text
- `cdtier_clean_ner_test_00017`：附录IOCIPDomainil.**..plugin-verifica**.attachment-download.services.enc2globaattachment-download.**-stest.enc2globaauthen…
- `cdtier_clean_ner_test_00051`：DomainIOCURLhttp:///ln.corehttp:///ln/a.asphttp:///dota3.tar.gzhttp:///a.jsphttp:///aa.jsphttp:///report.jsphttp:///ipc.…
- `cdtier_clean_ner_test_00053`：异或解密过程中使用的密钥为：D762A7DE3BB6A4BADB0C605D4FEC69218C05CFAA93C717CA5F17E7D14A3B8B6F8E31E10F71C442CDF767608C6D00202A0CA82AB72B…

### campaign
- `cdtier_clean_ner_test_00033`：利用社会工程学方案攻击是Lazarus的显著特征之一，在近两年的时间内，该组织进行了多起类似方案的攻击活动，譬如“DreamJob”、“针对安全研究人员的攻击”等…
- `cdtier_clean_ner_test_00145`：但在血茜草活动中，该诱骗活动与钓鱼网站活动同时进行的情况，在业界并不多见…
- `cdtier_clean_ner_test_00146`：（一）约稿型（一）约稿型通过著名期刊进行约稿一直都是境外情报部门的常用伎俩，但不同的是，血茜草活动主要目的是，让受害者误以为对方是合法机构的让受害者误以为对方是合法机构的前提下前提下，发送内部信息到攻击者处…

### qwen_only
- `cdtier_clean_ner_test_00141`：同时奇安信威胁情报中心ALPHA平台(ti.qianxin.com)对此次攻击活动相关IOC已有APT-C-23标签…
- `cdtier_clean_ner_test_00163`：图17钓鱼邮件3.2恶意宏代码执行图18恶意的邮件附件恶意代码被加密保存在诱饵文档excel表格的D500单元格的批注中（见图19）…
- `cdtier_clean_ner_test_00194`：该漏洞是存在于AndroidBinder组件中的UAF漏洞，可被利用来提升权限到root级别…

并排表格式：`句子 | gold | BiLSTM-CRF | RoBERTa-CRF | Qwen-LoRA | Qwen-full`