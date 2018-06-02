## 声明

* 由于本项目源于合作项目，故无法提供LoadData.py部分，敬请谅解。
* 本项目主要为针对DPCNN（**Deep Pyramid Convolutional Neural Networks for Text Categorization** ）的论文复现以及基于知乎看山杯Inception的修改和复现，后者效果略优。
* 本项目基于Pytorch框架实现，但内部使用 *tensorboardX* 进行tensorboard支持。
* 项目所需非基础requirement列于requirements.txt中，可使用`pip install -r requirements.txt`命令一键安装。
* 模型置于models文件夹下。
* 感谢Zekang Li的部分代码合作。

## 模型简介

* BasicModule：基类模型，提供基础的保存读取功能
* TextCNNDeep：基于原始论文的dpcnn模型，前置特征提取为标准TextCNN
* TextCNNInc：基于知乎看山杯的TextCNN模型
* TextCNNIncDeep：改进后的dpcnn模型，基于知乎看山杯的TextCNN模型

## 参数部分

### 公共参数设置
- self.USE_CUDA           = torch.cuda.is_available()  # GPU是否可用
- self.RUNNING_ON_SERVER  = False                      # 代码运行在本地还是服务器
- self.SUMMARY_PATH       = "summary/TextCNN_char"     # 设定tensorboard保存路径
- self.NET_SAVE_PATH      = "./source/trained_net/"    # 训练好的网络的储存位置
- self.TRAIN_DATASET_PATH = "../test_train/xx.txt"     # 训练集位置
- self.TEST_DATASET_PATH  = "../test_train/xx.txt"     # 测试集位置
- self.NUM_EPOCHS         = 1000                       # 本次BATCH数目
- self.BATCH_SIZE         = 32                         # 每个BATCH数据大小
- self.TOP_NUM            = 4                          # 测试时需求前几的Acc
- self.NUM_WORKERS        = 4                          # pytorch用几个线程工作读数据
- self.IS_TRAINING        = True                       # 选择模式“训练”或“测试”
- self.ENSEMBLE_TEST      = False                      # 测试模式下是否需要模型融合测试
- self.LEARNING_RATE      = 0.001                      # 学习率
- self.RE_TRAIN           = False                      # 本次训练是否要加载之前训练好的模型
- self.TEST_POSITION      = 'xxx Server'               # 本次训练运行在哪里

### 模型共享参数设置
- self.OPTIMIZER          = 'Adam'                     # 优化器选择
- self.USE_CHAR           = True                       # 使用char还是词
- self.USE_WORD2VEC       = True                       # 使用词语时是否使用词向量
- self.NUM_CLASSES        = 1890                       # 本次训练的分类数
- self.EMBEDDING_DIM      = 512                        # 词嵌入的维度
- self.VOCAB_SIZE         = 20029                      # 生成的词库大小
- self.CHAR_SIZE          = 3403                       # 生成的字库大小

### TextCNN模型设置
- self.TITLE_DIM          = 200                        # 中间层维度
- self.SENT_LEN           = 20                         # 句子截断长度
- self.LINER_HID_SIZE     = 2000                       # fc中间层维度
- self.KERNEL_SIZE        = [1,2,3,4,5]                # 卷积核大小

###  TextCNNInc模型设置
self.SIN_KER_SIZE = [1, 3]                             # 单层卷积卷积核大小
self.DOU_KER_SIZE = [(1, 3), (3, 5)]                   # 双层卷积卷积核大小。元组内第一项为第一层
                                                       # conv的核尺寸，第二项为第二层conv的核尺寸，

 