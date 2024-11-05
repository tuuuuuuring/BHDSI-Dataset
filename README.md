## Menu

- [Home](#BHDSI-Dataset)
- [Sample Example](#Sample-Example)
- [City Sample Size Distribution](#Distribution-of-City-Sample-Sizes)
- [Download Dataset](https://drive.google.com/drive/folders/1551nlQeoTB8cBbvT362jnfz70GOhIfU5?usp=sharing)
- [Model Comparison](#Model-Accuracy-Comparison)
- [Citation](#Article-Citation)

# BHDSI-Dataset

Estimating building heights using optical and SAR remote sensing imagery is crucial for understanding urban morphology and optimizing urban stock space. However, existing datasets have several limitations: the small sample size makes it difficult to meet the demands of deep learning-based remote sensing information extraction, and the limited area coverage does not provide sufficient geographic diversity and spatial feature representation. Moreover, the lack of open-source datasets restricts their broader application and validation in research.

To address these issues, this paper constructs a large-scale dataset focused on building height regression, covering the central urban areas of 62 cities in China, with a total of 5606 samples, including Sentinel-1 and Sentinel-2 remote sensing imagery and ground truth building heights. Compared with other datasets, this dataset features a larger sample size, broader coverage, and reasonable building height distribution, better meeting the training needs of deep learning models. Based on this, the paper evaluates the BHDSI dataset and other similar datasets using the same deep learning model and compares the performance of multiple models in building height regression tasks when using the BHDSI dataset. The study shows that the BHDSI dataset outperforms others in building height regression tasks. Further analysis reveals that regions with lower building heights have relatively higher estimation accuracy when using the BHDSI dataset. Additionally, the design of neural network decoders plays a more significant role in the regression task. This dataset and the experimental results provide important references and support for future research in building height estimation.

The download link is here:  
[Download BHDSI-Dataset](https://drive.google.com/drive/folders/1551nlQeoTB8cBbvT362jnfz70GOhIfU5?usp=sharing)

# Sample Example
![sample](https://github.com/tuuuuuuring/BHDSI-Dataset/blob/main/%E5%B8%83%E5%B1%801gai.png)

<center>

# Distribution of City Sample Sizes

| City        | Sample Size | City        | Sample Size | City        | Sample Size | City        | Sample Size |
|-------------|-------------|-------------|-------------|-------------|-------------|-------------|-------------|
| Macau       | 3           | Huizhou     | 22          | Xiamen      | 62          | Xi'an       | 104         |
| Baoding     | 23          | Jinan       | 79          | Shantou     | 20          | Xining      | 16          |
| Beijing     | 190         | Jiaxing     | 36          | Shanghai    | 342         | Hong Kong   | 31          |
| Changzhou   | 72          | Jinhua      | 20          | Shaoxing    | 33          | Xuzhou      | 33          |
| Chengdu     | 225         | Kunming     | 103         | Shenzhen    | 113         | Yantai      | 65          |
| Dalian      | 165         | Lhasa       | 7           | Shenyang    | 165         | Yangzhou    | 13          |
| Dongguan    | 98          | Lanzhou     | 27          | Shijiazhuang| 76          | Yinchuan    | 12          |
| Ordos       | 11          | Luoyang     | 27          | Suzhou      | 103         | Changchun   | 103         |
| Foshan Guangzhou | 277    | Nanchang    | 35          | Taizhou     | 26          | Changsha    | 98          |
| Fuzhou      | 37          | Nanjing     | 128         | Taiyuan     | 40          | Zhengzhou   | 147         |
| Guiyang     | 54          | Nanning     | 62          | Tangshan    | 35          | Zhongshan   | 19          |
| Harbin      | 76          | Nantong     | 49          | Tianjin     | 96          | Chongqing   | 199         |
| Haikou      | 20          | Ningbo      | 74          | Wenzhou     | 51          | Zhuhai      | 20          |
| Hangzhou    | 124         | Qingdao     | 150         | Wuxi        | 79          |             |             |
| Hefei       | 91          | Quanzhou    | 38          | Wuhu        | 29          |             |             |
| Hohhot      | 14          | Sanya       | 29          | Wuhan       | 176         |             |             |

</center>

# Model Accuracy Comparison
Based on this dataset and the project's code, the performance of some benchmark models is as follows
| Network Type      | Backbone       | Decoder        | RMSE (m) | MAE (m) | Accuracy (%) | IoU (%) | F1-Score (%) |
|-------------------|----------------|----------------|----------|---------|--------------|---------|--------------|
| Res-Unet          | ResNet50       | Unet           | 5.988    | 2.443   | 0.143        | 0.293   | 0.443        |
| VGG-Unet          | VGG16          | Unet           | 5.873    | 2.173   | 0.165        | 0.322   | 0.461        |
| Efficient-Unet    | EfficientNetb3 | Unet           | 5.948    | 2.146   | 0.158        | 0.295   | 0.425        |
| UperNet           | ResNet50       | FPN            | 6.224    | 2.440   | 0.125        | 0.249   | 0.383        |
| DeepLabV3         | ResNet50       | ASPP           | 6.720    | 2.604   | 0.095        | 0.190   | 0.301        |

# Article Citation

The article related to this dataset can be found here:
xxxx. [DOI:xxxxxxxx/xxx.xxxxxxxx]
