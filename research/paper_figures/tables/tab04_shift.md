**Accuracy and prediction flips under distribution shift (mean ± s.d. over seeds)**

| Model | Variant | Test set | n | Accuracy | Macro-F1 | Flip rate vs FP32 | McNemar p<0.05 (runs) |
|---|---|---|---|---|---|---|---|
| MobileNetV2 | FP32 | PlantVillage test | 8143 | 0.9971 ± 0.0002 | 0.9962 ± 0.0005 | – | – |
| MobileNetV2 | FP32 | Background removed | 7964 | 0.8578 ± 0.0054 | 0.8268 ± 0.0078 | – | – |
| MobileNetV2 | FP32 | PlantDoc (all), strict | 2911 | 0.2692 ± 0.0096 | 0.2349 ± 0.0051 | – | – |
| MobileNetV2 | FP32 | PlantDoc (all), restricted | 2911 | 0.3007 ± 0.0023 | 0.2476 ± 0.0013 | – | – |
| MobileNetV2 | FP16 (autocast) | PlantVillage test | 8143 | 0.9971 ± 0.0002 | 0.9962 ± 0.0005 | 0.0000 ± 0.0000 | 0/3 |
| MobileNetV2 | FP16 (autocast) | Background removed | 7964 | 0.8578 ± 0.0051 | 0.8270 ± 0.0074 | 0.0008 ± 0.0003 | 0/3 |
| MobileNetV2 | FP16 (autocast) | PlantDoc (all), strict | 2911 | 0.2692 ± 0.0095 | 0.2349 ± 0.0053 | 0.0035 ± 0.0005 | 0/3 |
| MobileNetV2 | FP16 (autocast) | PlantDoc (all), restricted | 2911 | 0.3007 ± 0.0017 | 0.2478 ± 0.0009 | 0.0032 ± 0.0004 | 0/3 |
| MobileNetV2 | FP16 (converted) | PlantVillage test | 8143 | 0.9971 ± 0.0002 | 0.9963 ± 0.0004 | 0.0000 ± 0.0001 | 0/3 |
| MobileNetV2 | FP16 (converted) | Background removed | 7964 | 0.8576 ± 0.0050 | 0.8267 ± 0.0077 | 0.0012 ± 0.0005 | 0/3 |
| MobileNetV2 | FP16 (converted) | PlantDoc (all), strict | 2911 | 0.2690 ± 0.0088 | 0.2347 ± 0.0047 | 0.0044 ± 0.0004 | 0/3 |
| MobileNetV2 | FP16 (converted) | PlantDoc (all), restricted | 2911 | 0.3005 ± 0.0017 | 0.2475 ± 0.0014 | 0.0033 ± 0.0002 | 0/3 |
| MobileNetV2 | INT8 dynamic (Khan et al.) | PlantVillage test | 8143 | 0.9971 ± 0.0002 | 0.9961 ± 0.0005 | 0.0001 ± 0.0001 | 0/3 |
| MobileNetV2 | INT8 dynamic (Khan et al.) | Background removed | 7964 | 0.8577 ± 0.0050 | 0.8269 ± 0.0075 | 0.0013 ± 0.0002 | 0/3 |
| MobileNetV2 | INT8 dynamic (Khan et al.) | PlantDoc (all), strict | 2911 | 0.2694 ± 0.0103 | 0.2345 ± 0.0054 | 0.0061 ± 0.0016 | 0/3 |
| MobileNetV2 | INT8 dynamic (Khan et al.) | PlantDoc (all), restricted | 2911 | 0.3001 ± 0.0019 | 0.2466 ± 0.0003 | 0.0055 ± 0.0009 | 0/3 |
| MobileNetV2 | INT8 PTQ | PlantVillage test | 8143 | 0.9962 ± 0.0010 | 0.9947 ± 0.0015 | 0.0029 ± 0.0010 | 1/3 |
| MobileNetV2 | INT8 PTQ | Background removed | 7964 | 0.8542 ± 0.0076 | 0.8230 ± 0.0067 | 0.0549 ± 0.0164 | 3/3 |
| MobileNetV2 | INT8 PTQ | PlantDoc (all), strict | 2911 | 0.2667 ± 0.0062 | 0.2338 ± 0.0007 | 0.1775 ± 0.0201 | 0/3 |
| MobileNetV2 | INT8 PTQ | PlantDoc (all), restricted | 2911 | 0.2971 ± 0.0043 | 0.2421 ± 0.0008 | 0.1742 ± 0.0264 | 0/3 |
| MobileNetV2 | INT8 QAT | PlantVillage test | 8143 | 0.9952 ± 0.0006 | 0.9929 ± 0.0006 | 0.0037 ± 0.0008 | 3/3 |
| MobileNetV2 | INT8 QAT | Background removed | 7964 | 0.8920 ± 0.0182 | 0.8565 ± 0.0163 | 0.0785 ± 0.0050 | 3/3 |
| MobileNetV2 | INT8 QAT | PlantDoc (all), strict | 2911 | 0.2768 ± 0.0058 | 0.2471 ± 0.0007 | 0.2471 ± 0.0195 | 1/3 |
| MobileNetV2 | INT8 QAT | PlantDoc (all), restricted | 2911 | 0.3072 ± 0.0036 | 0.2549 ± 0.0037 | 0.2465 ± 0.0264 | 0/3 |
| MobileNetV2 | INT8 PTQ outlier-aware (ours) | PlantVillage test | 8143 | 0.9961 ± 0.0011 | 0.9946 ± 0.0016 | 0.0025 ± 0.0008 | 1/3 |
| MobileNetV2 | INT8 PTQ outlier-aware (ours) | Background removed | 7964 | 0.8585 ± 0.0088 | 0.8276 ± 0.0079 | 0.0526 ± 0.0191 | 2/3 |
| MobileNetV2 | INT8 PTQ outlier-aware (ours) | PlantDoc (all), strict | 2911 | 0.2681 ± 0.0058 | 0.2350 ± 0.0007 | 0.1512 ± 0.0086 | 0/3 |
| MobileNetV2 | INT8 PTQ outlier-aware (ours) | PlantDoc (all), restricted | 2911 | 0.2996 ± 0.0052 | 0.2468 ± 0.0032 | 0.1454 ± 0.0213 | 0/3 |
| MobileNetV2 | INT8 QAT outlier-aware (ours) | PlantVillage test | 8143 | 0.9944 ± 0.0008 | 0.9918 ± 0.0019 | 0.0041 ± 0.0011 | 3/3 |
| MobileNetV2 | INT8 QAT outlier-aware (ours) | Background removed | 7964 | 0.8706 ± 0.0164 | 0.8375 ± 0.0165 | 0.0784 ± 0.0043 | 2/3 |
| MobileNetV2 | INT8 QAT outlier-aware (ours) | PlantDoc (all), strict | 2911 | 0.2627 ± 0.0016 | 0.2385 ± 0.0058 | 0.2534 ± 0.0185 | 1/3 |
| MobileNetV2 | INT8 QAT outlier-aware (ours) | PlantDoc (all), restricted | 2911 | 0.2997 ± 0.0028 | 0.2525 ± 0.0049 | 0.2483 ± 0.0112 | 0/3 |
| MobileNetV3-Small | FP32 | PlantVillage test | 8143 | 0.9952 ± 0.0006 | 0.9921 ± 0.0014 | – | – |
| MobileNetV3-Small | FP32 | Background removed | 7964 | 0.6953 ± 0.0353 | 0.6548 ± 0.0274 | – | – |
| MobileNetV3-Small | FP32 | PlantDoc (all), strict | 2911 | 0.1892 ± 0.0053 | 0.1759 ± 0.0061 | – | – |
| MobileNetV3-Small | FP32 | PlantDoc (all), restricted | 2911 | 0.2199 ± 0.0070 | 0.1843 ± 0.0059 | – | – |
| MobileNetV3-Small | FP16 (autocast) | PlantVillage test | 8143 | 0.9952 ± 0.0007 | 0.9921 ± 0.0016 | 0.0002 ± 0.0002 | 0/3 |
| MobileNetV3-Small | FP16 (autocast) | Background removed | 7964 | 0.6961 ± 0.0358 | 0.6554 ± 0.0283 | 0.0067 ± 0.0019 | 1/3 |
| MobileNetV3-Small | FP16 (autocast) | PlantDoc (all), strict | 2911 | 0.1888 ± 0.0038 | 0.1750 ± 0.0041 | 0.0111 ± 0.0015 | 0/3 |
| MobileNetV3-Small | FP16 (autocast) | PlantDoc (all), restricted | 2911 | 0.2199 ± 0.0061 | 0.1839 ± 0.0047 | 0.0100 ± 0.0024 | 0/3 |
| MobileNetV3-Small | FP16 (converted) | PlantVillage test | 8143 | 0.9952 ± 0.0007 | 0.9921 ± 0.0015 | 0.0002 ± 0.0003 | 0/3 |
| MobileNetV3-Small | FP16 (converted) | Background removed | 7964 | 0.6956 ± 0.0350 | 0.6549 ± 0.0274 | 0.0073 ± 0.0009 | 3/3 |
| MobileNetV3-Small | FP16 (converted) | PlantDoc (all), strict | 2911 | 0.1889 ± 0.0052 | 0.1752 ± 0.0051 | 0.0123 ± 0.0017 | 0/3 |
| MobileNetV3-Small | FP16 (converted) | PlantDoc (all), restricted | 2911 | 0.2196 ± 0.0069 | 0.1832 ± 0.0051 | 0.0118 ± 0.0032 | 0/3 |
| MobileNetV3-Small | INT8 dynamic (Khan et al.) | PlantVillage test | 8143 | 0.9950 ± 0.0006 | 0.9918 ± 0.0015 | 0.0003 ± 0.0001 | 0/3 |
| MobileNetV3-Small | INT8 dynamic (Khan et al.) | Background removed | 7964 | 0.6959 ± 0.0360 | 0.6549 ± 0.0268 | 0.0086 ± 0.0001 | 0/3 |
| MobileNetV3-Small | INT8 dynamic (Khan et al.) | PlantDoc (all), strict | 2911 | 0.1891 ± 0.0055 | 0.1765 ± 0.0061 | 0.0247 ± 0.0027 | 0/3 |
| MobileNetV3-Small | INT8 dynamic (Khan et al.) | PlantDoc (all), restricted | 2911 | 0.2195 ± 0.0088 | 0.1845 ± 0.0079 | 0.0223 ± 0.0019 | 0/3 |
| MobileNetV3-Small | INT8 PTQ | PlantVillage test | 8143 | 0.0458 ± 0.0246 | 0.0169 ± 0.0095 | 0.9541 ± 0.0246 | 3/3 |
| MobileNetV3-Small | INT8 PTQ | Background removed | 7964 | 0.0424 ± 0.0224 | 0.0159 ± 0.0059 | 0.9300 ± 0.0742 | 3/3 |
| MobileNetV3-Small | INT8 PTQ | PlantDoc (all), strict | 2911 | 0.0077 ± 0.0042 | 0.0067 ± 0.0008 | 0.9020 ± 0.0007 | 3/3 |
| MobileNetV3-Small | INT8 PTQ | PlantDoc (all), restricted | 2911 | 0.0432 ± 0.0058 | 0.0236 ± 0.0064 | 0.8618 ± 0.0248 | 3/3 |
| MobileNetV3-Small | INT8 QAT | PlantVillage test | 8143 | 0.6457 ± 0.0140 | 0.5378 ± 0.0156 | 0.3540 ± 0.0136 | 3/3 |
| MobileNetV3-Small | INT8 QAT | Background removed | 7964 | 0.4594 ± 0.0207 | 0.3214 ± 0.0195 | 0.5470 ± 0.0253 | 3/3 |
| MobileNetV3-Small | INT8 QAT | PlantDoc (all), strict | 2911 | 0.0891 ± 0.0038 | 0.0623 ± 0.0061 | 0.8192 ± 0.0092 | 3/3 |
| MobileNetV3-Small | INT8 QAT | PlantDoc (all), restricted | 2911 | 0.1154 ± 0.0084 | 0.0736 ± 0.0070 | 0.8053 ± 0.0097 | 3/3 |
| MobileNetV3-Small | INT8 PTQ outlier-aware (ours) | PlantVillage test | 8143 | 0.9889 ± 0.0010 | 0.9800 ± 0.0032 | 0.0094 ± 0.0015 | 3/3 |
| MobileNetV3-Small | INT8 PTQ outlier-aware (ours) | Background removed | 7964 | 0.5645 ± 0.0502 | 0.5203 ± 0.0440 | 0.2042 ± 0.0340 | 3/3 |
| MobileNetV3-Small | INT8 PTQ outlier-aware (ours) | PlantDoc (all), strict | 2911 | 0.1727 ± 0.0104 | 0.1538 ± 0.0096 | 0.3305 ± 0.0176 | 3/3 |
| MobileNetV3-Small | INT8 PTQ outlier-aware (ours) | PlantDoc (all), restricted | 2911 | 0.2057 ± 0.0108 | 0.1640 ± 0.0104 | 0.3088 ± 0.0199 | 2/3 |
| MobileNetV3-Small | INT8 QAT outlier-aware (ours) | PlantVillage test | 8143 | 0.9908 ± 0.0012 | 0.9847 ± 0.0017 | 0.0074 ± 0.0012 | 3/3 |
| MobileNetV3-Small | INT8 QAT outlier-aware (ours) | Background removed | 7964 | 0.6706 ± 0.0484 | 0.6220 ± 0.0435 | 0.1529 ± 0.0161 | 3/3 |
| MobileNetV3-Small | INT8 QAT outlier-aware (ours) | PlantDoc (all), strict | 2911 | 0.1768 ± 0.0026 | 0.1621 ± 0.0038 | 0.3205 ± 0.0197 | 2/3 |
| MobileNetV3-Small | INT8 QAT outlier-aware (ours) | PlantDoc (all), restricted | 2911 | 0.2084 ± 0.0010 | 0.1691 ± 0.0026 | 0.3166 ± 0.0186 | 2/3 |
