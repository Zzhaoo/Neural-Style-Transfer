from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import matplotlib as mpl

mpl.rcParams['figure.figsize'] = (12,12)
mpl.rcParams['axes.grid'] = False
# mpl是用来进行图像绘制的一个包，rcParams是为了重制默认参数，比如上面两个是为了改动图像大小和消除网格
import matplotlib.pyplot as plt
import numpy as np
# 矩阵操作
import time
# 计时操作
import functools
# 进行复杂的函数处理
import IPython.display as display
'''下面两行不是macos的话就去掉'''
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'


#加载图片源
def load_img(img_path):
    #这个方法是为了加载图片，因为图片的较长边要被限制在512像素
    max_dim = 512
    image = tf.io.read_file(img_path)
    image = tf.image.decode_image(image,channels = 3)

    # decode_image将PNG编码的图像解码为uint8或uint16张量
    #     channels表示解码图像的期望数量的颜色通道.
    #     接受的值是：
    #     0：使用PNG编码图像中的通道数量.
    #     1：输出灰度图像.
    #     3：输出RGB图像.
    #     4：输出RGBA图像
    image = tf.image.convert_image_dtype(image, tf.float32)
    # 图片归一化，将image的每个像素的3个intRGB转化为32位浮点数
    # 图片最终应该是384*512*3，然后每一项都是32位浮点数
    shape = tf.cast(tf.shape(image)[:-1], tf.float32)
    # 获取图片尺寸的张量，先将图片的最后一维像素值去掉（变为384*512），再将图片尺寸（本来为int型整数）转化为浮点数（384*512变为384.0*512.0）
    long_dim = max(shape)
    new_shape = tf.cast(shape * max_dim / long_dim, tf.int32)
    image = tf.image.resize(image, new_shape)
    # 这一步将图片进行适当的缩小或放大，使得较长边变为512像素
    image = image[tf.newaxis, :]
    return image

#显示加载好的图片
def showImage(image, title=None):
    if len(image.shape) > 3:
        image = tf.squeeze(image,axis=0)
        # 该方法是去除掉张量中为1的维度，因为图片基本上都是长*宽*3，所以多出的维度一般都是1维的
    plt.imshow(image)
    plt.show()
    if title:
        plt.title(title)

'''因为不是所有的VGG层都要用到，所以设置了这个方法，输入你要用的层名，返回包含这些层的模型'''
def vgg_layers(layer_names):
    # 加载模型。 加载已经在 imagenet 数据上预训练的 VGG
    vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
    vgg.trainable = False

    outputs = [vgg.get_layer(name).output for name in layer_names]
    # 将input和output组合成为模型
    model = tf.keras.Model([vgg.input], outputs)
    return model


'''
下面方法用于计算Gram矩阵
'''
def get_gram_matrix(input_tensor):
    # 先算输入矩阵的外积，再求平均
    result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor) # result[b,c,d] = sum_j input_tensor[b,i,j,c] * input_tensor[b,i,j,d]
    input_shape = tf.shape(input_tensor)
    num_locations = tf.cast(input_shape[1]*input_shape[2], tf.float32)
    return result/(num_locations)


'''
该类调用了vgg_layers方法和gram_matrix方法
在图像上调用此模型，可以返回 style_layers 的 gram 矩阵和 content_layers 的内容，用于后面的梯度下降'''
class StyleContentModel(tf.keras.models.Model):
    def __init__(self, style_layers, content_layers):
        super(StyleContentModel, self).__init__()
        self.vgg = vgg_layers(style_layers + content_layers)

        self.style_layers = style_layers
        self.content_layers = content_layers

        self.num_style_layers = len(style_layers)
        self.vgg.trainable = False

    def __call__(self, inputs):
        "Expects float input in [0,1]"
        preprocessed_input = tf.keras.applications.vgg19.preprocess_input(inputs*255.0)
        outputs = self.vgg(preprocessed_input)
        # 获取vgg输出的特征层和内容层的输出
        style_outputs, content_outputs = (outputs[:self.num_style_layers],
                                          outputs[self.num_style_layers:])
        # 对风格特征输出求gram矩阵
        style_outputs = [get_gram_matrix(style_output)
                         for style_output in style_outputs]

        content_dict = {content_name: value
                        for content_name, value
                        in zip(self.content_layers, content_outputs)}


        style_dict = {style_name: value
                      for style_name, value
                      in zip(self.style_layers, style_outputs)}

        return {'content': content_dict, 'style': style_dict}

'''为了使图像像素值在0-1之间'''
def clip_0_1(image):
  return tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)


'''返回联合loss'''
def style_content_loss(outputs):
    style_outputs = outputs['style']
    content_outputs = outputs['content']

    # 使用最简单的均方误差和作为损失函数
    style_loss = tf.add_n([tf.reduce_mean((style_outputs[name]-style_targets[name])**2)
                           for name in style_outputs.keys()])

    content_loss = tf.add_n([tf.reduce_mean((content_outputs[name]-content_targets[name])**2)
                             for name in content_outputs.keys()])

    loss = style_weight/num_style_layers*style_loss + content_weight/num_content_layers*content_loss
    return loss

'''优化部分，将高频分量损失也算到损失函数中'''
def high_pass_x_y(image):
  x_var = image[:,:,1:,:] - image[:,:,:-1,:]
  y_var = image[:,1:,:,:] - image[:,:-1,:,:]
  return x_var, y_var

'''优化部分，将高频分量损失也算到损失函数中'''
def total_variation_loss(image):
  x_deltas, y_deltas = high_pass_x_y(image)
  return tf.reduce_mean(x_deltas**2) + tf.reduce_mean(y_deltas**2)

'''训练'''
@tf.function()
def train_step(image):
  with tf.GradientTape() as tape:
    outputs = extractor(image)
    loss = style_content_loss(outputs)# 获得损失值
    loss += total_variation_weight * total_variation_loss(image)

  grad = tape.gradient(loss, image)
  opt.apply_gradients([(grad, image)])
  image.assign(clip_0_1(image))# 使图片像素值在0-1内

if __name__ == "__main__":
    content_path = "city.jpg"
    style_path = "starnight.jpg"
    # 这一步是下载了两张图片，第一张是一浦东内容图片，第二张是梵高风格图片
    content_image = load_img(content_path)
    style_image = load_img(style_path)

    plt.subplot(1, 2, 1)
    # 将当前画图区域分为1行2列，当前为位置1，这是内容图像
    showImage(content_image, 'Content_image')

    plt.subplot(1, 2, 2)
    # 将当前画图区域分为1行2列，当前为位置2，这是风格图像
    showImage(style_image, 'Style_image')

    '''
    下面进行训练
    使用的是VGG19网络，这是一个已经预训练好的用于分类的神经网络
    '''
    '''下面加载没有分类部分的VGG19，并列出各层名称
    前几层是边缘、纹理等低级特征
    后几层是高级特征，如轮子、眼睛、嘴巴等'''

    #从github上下载vgg19网络，并列出各层的名字
    vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')

    print()
    for layer in vgg.layers:
        print(layer.name)

    # 选择用于表示内容的层，因为底层注重细节，高层注重整体，所以表示内容的话选择中间层最好
    content_layers = ['block5_conv2']

    # 选择用于表示风格的层
    style_layers = ['block1_conv2',
                    'block2_conv2',
                    'block3_conv2',
                    'block4_conv2',
                    'block5_conv2']

    num_content_layers = len(content_layers)
    num_style_layers = len(style_layers)


    '''构建提取器'''
    extractor = StyleContentModel(style_layers, content_layers)
    '''获取风格图片的风格层输出和内容图片的内容层输出'''
    style_targets = extractor(style_image)['style']
    content_targets = extractor(content_image)['content']
    # 这两个直接提取出了风格和内容的目标值

    image = tf.Variable(content_image)
    # 这个是目标图像，首先让他和内容图像形状一样（Variable是用于初始化的函数）
    opt = tf.optimizers.Adam(learning_rate=0.02, beta_1=0.99, epsilon=1e-1)
    # 优化函数，这在train_step方法中被使用到

    style_weight = 1e-2
    content_weight = 1e4
    # 使用两个损失的加权组合来获得总损失，这在style_content_loss方法中被使用到
    total_variation_weight = 1e8
    # 上面是高频分量的损
    showImage(image.read_value())

    '''下面进行一段很长很长的优化'''
    start = time.time()

    epochs = 10
    steps_per_epoch = 50

    step = 0
    for n in range(epochs):
        for m in range(steps_per_epoch):
            step += 1
            train_step(image)
            print(".", end='')
        display.clear_output(wait=True)
        showImage(image.read_value())
        plt.title("Train step: {}".format(step))
        plt.show()
    end = time.time()
    print("Total time: {:.1f}".format(end - start))
