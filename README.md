# BHDSI-Dataset
Estimating building heights using optical and SAR remote sensing imagery is crucial for understanding urban morphology and optimizing urban stock space. However, existing datasets have several limitations: the small sample size makes it difficult to meet the demands of deep learning-based remote sensing information extraction, and the limited area coverage does not provide sufficient geographic diversity and spatial feature representation. Moreover, the lack of open-source datasets restricts their broader application and validation in research. To address these issues, this paper constructs a large-scale dataset focused on building height regression, covering the central urban areas of 62 cities in China, with a total of 5606 samples, including Sentinel-1 and Sentinel-2 remote sensing imagery and ground truth building heights. Compared with other datasets, this dataset features a larger sample size, broader coverage, and reasonable building height distribution, better meeting the training needs of deep learning models. Based on this, the paper evaluates the BHDSI dataset and other similar datasets using the same deep learning model and compares the performance of multiple models in building height regression tasks when using the BHDSI dataset. The study shows that the BHDSI dataset outperforms others in building height regression tasks. Further analysis reveals that regions with lower building heights have relatively higher estimation accuracy when using the BHDSI dataset. Additionally, the design of neural network decoders plays a more significant role in the regression task. This dataset and the experimental results provide important references and support for future research in building height estimation.
 
The download link is here.https://drive.google.com/drive/folders/1551nlQeoTB8cBbvT362jnfz70GOhIfU5?usp=sharing
<center>
![sample](https://github.com/tuuuuuuring/BHDSI-Dataset/blob/main/%E5%B8%83%E5%B1%801gai.png)
 </center>
<center>

### 城市样本数量分布
| 城市       | 样本数量 | 城市       | 样本数量 | 城市       | 样本数量 | 城市       | 样本数量 |
|------------|----------|------------|----------|------------|----------|------------|----------|
| 澳门       | 3        | 惠州       | 22       | 厦门       | 62       | 西安       | 104      |
| 保定       | 23       | 济南       | 79       | 汕头       | 20       | 西宁       | 16       |
| 北京       | 190      | 嘉兴       | 36       | 上海       | 342      | 香港       | 31       |
| 常州       | 72       | 金华       | 20       | 绍兴       | 33       | 徐州       | 33       |
| 成都       | 225      | 昆明       | 103      | 深圳       | 113      | 烟台       | 65       |
| 大连       | 165      | 拉萨       | 7        | 沈阳       | 165      | 扬州       | 13       |
| 东莞       | 98       | 兰州       | 27       | 石家庄     | 76       | 银川       | 12       |
| 鄂尔多斯   | 11       | 洛阳       | 27       | 苏州       | 103      | 长春       | 103      |
| 佛山广州   | 277      | 南昌       | 35       | 台州       | 26       | 长沙       | 98       |
| 福州       | 37       | 南京       | 128      | 太原       | 40       | 郑州       | 147      |
| 贵阳       | 54       | 南宁       | 62       | 唐山       | 35       | 中山       | 19       |
| 哈尔滨     | 76       | 南通       | 49       | 天津       | 96       | 重庆       | 199      |
| 海口       | 20       | 宁波       | 74       | 温州       | 51       | 珠海       | 20       |
| 杭州       | 124      | 青岛       | 150      | 无锡       | 79       |            |          |
| 合肥       | 91       | 泉州       | 38       | 芜湖       | 29       |            |          |
| 呼和浩特   | 14       | 三亚       | 29       | 武汉       | 176      |            |          |
</center>
