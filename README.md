# CSSNet

**UPDATE: We have open source the real ziyuan hyperspectral datasets used in this paper to contribute the hyperspectral image fusion field.**

Include the hyperspectral image/pan image and hyperspectral image/multi-spectral image, which could be freely downloaded from the website:

hyperspectral image/pan image:https://aistudio.baidu.com/aistudio/datasetdetail/95831

hyperspectral image/multi-spectral image:https://aistudio.baidu.com/aistudio/datasetdetail/120965

Note that these two real datasets are acquired by the Ziyuan 1-02D satellite. and they have been spatially registered, respectively.

You could load these two image datasets by the code in the "main_hmfusion.py" and "main_hpfusion.py" as:

main_hmfusion.py:

```
from save_image_ziyuan_reduce import generate_data, crop_data
name = 'zy'
ms, pan, label = generate_data(ratio=3) 
ms_crop, pan_crop, label_crop = crop_data(ms, pan, label, training_size=16)
```

main_hpfusion.py

```
from save_image_ziyuan_hp import generate_data, crop_data
name = 'zy'
ms, pan, label = generate_data(ratio=12) 
ms_crop, pan_crop, label_crop = crop_data(ms, pan, label, training_size=16)
```

And remember to adjust the train and test patch number for the different "training_size".

The code of the paper ‘Hyperspectral image super-resolution network based on cross-scale non-local attention’.(TGRS)

The fusion between the hyperspectral and multi-spectral images could be conducted by running the ‘main_hmfusion.py’.

The fusion between the hyperspectral and PAN images could be conducted by running the ‘main_hpfusion.py’.

Note that the main network difference between these two fusion taks is the training strategy, including optimizer, training epochs.

If you have any questions, please feel free to contact me.

Please consider cite the paper if you find it helpful.

Email:cug_lsl@cug.edu.cn
