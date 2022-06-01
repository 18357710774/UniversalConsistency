# coding:utf8
import warnings
import torch as t

class DefaultConfig(object):
    model = 'FCNet'  # 使用的模型，名字必须与models/__init__.py中的名字一致
    load_model_path = None  # 加载预训练的模型的路径，为None代表不加载

    batch_size = 50  # batch size
    use_gpu = True  # user GPU or not
    num_workers = 0  # how many workers for loading data
    print_freq = 50  # print info every N batch
    max_epoch = 10000

    weight_decay_flag = False
    weight_decay = 0.001

    lr_decay = 0.5  # when val_loss increase, lr = lr*lr_decay
    lr_decay_step = 1  # Period of learning rate decay (每隔lr_decay_step个epoch学习率缩小lr_decay倍)

    criterion = t.nn.MSELoss()

    if t.cuda.is_available() & use_gpu:
        device = t.device('cuda')
    else:
        device = t.device('cpu')

    def _parse(self, kwargs):
        """
        根据字典kwargs 更新 config参数
        """
        for k, v in kwargs.items():
            if not hasattr(self, k):
                warnings.warn("Warning: opt has not attribut %s" % k)
            setattr(self, k, v)

        # print('user config:')
        # for k, v in self.__class__.__dict__.items():
        #     if not k.startswith('_'):
        #         print(k, getattr(self, k))


opt = DefaultConfig()
