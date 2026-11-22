**What each INT8 workflow actually quantizes (first seed; identical across seeds)**

| Model | Variant | parameters() | Stored weights | INT8 layers | FP32 layers | Size (MB) | Operator census |
|---|---|---|---|---|---|---|---|
| MobileNetV2 | FP32 | 2272550 | 2272550 | 0 | 53 | 9.34 | n/a |
| MobileNetV2 | INT8 dynamic (Khan et al.) | 2223872 | 2272550 | 1 | 52 | 9.20 | PARTIAL INT8: quantized conv 0, linear 1; still FLOAT conv 52, linear 0, add 10, mul 0 |
| MobileNetV2 | INT8 PTQ | 0 | 2255494 | 53 | 0 | 2.70 | FULL INT8: 52 conv + 1 linear + 10 add quantized |
| MobileNetV2 | INT8 QAT | 0 | 2255494 | 53 | 0 | 2.70 | FULL INT8: 52 conv + 1 linear + 10 add quantized |
| MobileNetV2 | INT8 PTQ outlier-aware (ours) | 471680 | 2255494 | 50 | 3 | 4.67 | PARTIAL INT8: quantized conv 49, linear 1; still FLOAT conv 3, linear 0, add 0, mul 0 | float blocks: ['features.17'] |
| MobileNetV2 | INT8 QAT outlier-aware (ours) | 473920 | 2257734 | 50 | 3 | 4.73 | PARTIAL INT8: quantized conv 49, linear 1; still FLOAT conv 3, linear 0, add 0, mul 0 | float blocks: ['features.17'] |
| MobileNetV3-Small | FP32 | 1556806 | 1556806 | 0 | 54 | 6.37 | n/a |
| MobileNetV3-Small | INT8 dynamic (Khan et al.) | 927008 | 1556806 | 2 | 52 | 4.49 | PARTIAL INT8: quantized conv 0, linear 2; still FLOAT conv 52, linear 0, add 0, mul 0 |
| MobileNetV3-Small | INT8 PTQ | 0 | 1550750 | 54 | 0 | 1.88 | FULL INT8: 52 conv + 2 linear + 6 add + 9 mul quantized |
| MobileNetV3-Small | INT8 QAT | 0 | 1550750 | 54 | 0 | 1.88 | FULL INT8: 52 conv + 2 linear + 6 add + 9 mul quantized |
| MobileNetV3-Small | INT8 PTQ outlier-aware (ours) | 1160 | 1550750 | 49 | 5 | 1.88 | PARTIAL INT8: quantized conv 47, linear 2; still FLOAT conv 5, linear 0, add 0, mul 1 | float blocks: ['features.0', 'features.1'] |
| MobileNetV3-Small | INT8 QAT outlier-aware (ours) | 1208 | 1550798 | 49 | 5 | 1.88 | PARTIAL INT8: quantized conv 47, linear 2; still FLOAT conv 5, linear 0, add 0, mul 1 | float blocks: ['features.0', 'features.1'] |
