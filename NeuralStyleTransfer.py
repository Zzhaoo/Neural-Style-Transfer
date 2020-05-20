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
    # 将图片变为张量，先将图片的最后一维去掉（变为384*512），再将图片尺寸（本来为int型整数）转化为浮点数（384*512变为384.0*512.0）
    long_dim = max(shape)
    scale = max_dim / long_dim
    new_shape = tf.cast(shape * scale, tf.int32)
    image = tf.image.resize(image, new_shape)
    # 这一步将图片进行适当的缩小或放大，使得较长边变为512像素
    image = image[tf.newaxis, :]
    return image

def showImage(image, title=None):
    if len(image.shape) > 3:
        image = tf.squeeze(image,axis=0)
        # 该方法是去除掉张量中为1的维度，因为图片基本上都是长*宽*3，所以多出的维度一般都是1维的
    plt.imshow(image)
    plt.show()
    if title:
        plt.title(title)


def vgg_layers(layer_names):
    # 加载模型。 加载已经在 imagenet 数据上预训练的 VGG
    vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
    vgg.trainable = False

    outputs = [vgg.get_layer(name).output for name in layer_names]

    model = tf.keras.Model([vgg.input], outputs)
    return model


'''
下面方法用于风格计算
事实证明，图像的风格可以通过不同 feature maps (特征图)上的平均值和相关性来描述。 
通过在每个位置计算 feature (特征)向量的外积，并在所有位置对该外积进行平均,可以计算出包含此信息的 Gram 矩阵
'''
def gram_matrix(input_tensor):
  result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
  input_shape = tf.shape(input_tensor)
  num_locations = tf.cast(input_shape[1]*input_shape[2], tf.float32)
  return result/(num_locations)


'''这个类用于返回风格和内容张量的模型
该类调用了vgg_layers方法和gram_matrix方法
在图像上调用此模型，可以返回 style_layers 的 gram 矩阵（风格）和 content_layers 的内容'''
class StyleContentModel(tf.keras.models.Model):
    def __init__(self, style_layers, content_layers):
        super(StyleContentModel, self).__init__()
        self.vgg = vgg_layers(style_layers + content_layers)
        self.style_layers = style_layers
        self.content_layers = content_layers
        self.num_style_layers = len(style_layers)
        self.vgg.trainable = False

    def call(self, inputs):
        "Expects float input in [0,1]"
        inputs = inputs * 255.0
        preprocessed_input = tf.keras.applications.vgg19.preprocess_input(inputs)
        outputs = self.vgg(preprocessed_input)
        style_outputs, content_outputs = (outputs[:self.num_style_layers],
                                          outputs[self.num_style_layers:])

        style_outputs = [gram_matrix(style_output)
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


'''返回loss'''
def style_content_loss(outputs):
    style_outputs = outputs['style']
    content_outputs = outputs['content']
    style_loss = tf.add_n([tf.reduce_mean((style_outputs[name]-style_targets[name])**2)
                           for name in style_outputs.keys()])
    style_loss *= style_weight / num_style_layers

    content_loss = tf.add_n([tf.reduce_mean((content_outputs[name]-content_targets[name])**2)
                             for name in content_outputs.keys()])
    content_loss *= content_weight / num_content_layers
    loss = style_loss + content_loss
    return loss

'''用于更新图像'''
@tf.function()
def train_step(image):
  with tf.GradientTape() as tape:
    outputs = extractor(image)
    loss = style_content_loss(outputs)# 获得损失值

  grad = tape.gradient(loss, image)
  opt.apply_gradients([(grad, image)])
  image.assign(clip_0_1(image))# 使图片像素值在0-1内

if __name__ == "__main__":
    content_path = tf.keras.utils.get_file('turtle.jpg',
                                           'https://storage.googleapis.com/download.tensorflow.org/example_images/Green_Sea_Turtle_grazing_seagrass.jpg')
    style_path = tf.keras.utils.get_file('kandinsky.jpg',
                                         'https://storage.googleapis.com/download.tensorflow.org/example_images/Vassily_Kandinsky%2C_1913_-_Composition_7.jpg')
    # 这一步是下载了两张图片，第一张是一张海龟图片，第二张是风格图片
    content_image = load_img(content_path)
    style_image = load_img(style_path)

    plt.subplot(1, 2, 1)
    # 将当前画图区域分为1行2列，当前为位置1
    showImage(content_image, 'Content_image')

    plt.subplot(1, 2, 2)
    # 将当前画图区域分为1行2列，当前为位置2
    showImage(style_image, 'Style_image')

    '''
    下面进行训练
    使用的是VGG19网络，这是一个已经预训练好的用于分类的神经网络
    '''
    x = tf.keras.applications.vgg19.preprocess_input(content_image * 255)
    x = tf.image.resize(x, (224, 224))
    vgg = tf.keras.applications.VGG19(include_top=True, weights='imagenet')
    prediction_probabilities = vgg(x)
    prediction_probabilities.shape

    predicted_top_5 = tf.keras.applications.vgg19.decode_predictions(prediction_probabilities.numpy())[0]
    [(class_name, prob) for (number, class_name, prob) in predicted_top_5]
    '''下面加载没有分类部分的VGG19，并列出各层名称
    前几层是边缘、纹理等低级特征
    后几层是高级特征，如轮子、眼睛、嘴巴等'''
    vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
    print()
    for layer in vgg.layers:
        print(layer.name)

    # 内容层将提取出我们的 feature maps （特征图）
    content_layers = ['block5_conv2']

    # 我们感兴趣的风格层
    style_layers = ['block1_conv1',
                    'block2_conv1',
                    'block3_conv1',
                    'block4_conv1',
                    'block5_conv1']

    num_content_layers = len(content_layers)
    num_style_layers = len(style_layers)
    '''
    那么,为什么我们预训练的图像分类网络中的这些中间层的输出允许我们定义风格和内容的表示？
    从高层理解，为了使网络能够实现图像分类（该网络已被训练过），它必须理解图像。 
    这需要将原始图像作为输入像素并构建内部表示，这个内部表示将原始图像像素转换为对图像中存在的 feature (特征)的复杂理解。
    这也是卷积神经网络能够很好地推广的一个原因：它们能够捕获不变性并定义类别（例如猫与狗）之间的 feature (特征)，这些 feature (特征)与背景噪声和其他干扰无关。 
    因此，将原始图像传递到模型输入和分类标签输出之间的某处的这一过程，可以视作复杂的 feature (特征)提取器。
    通过这些模型的中间层，我们就可以描述输入图像的内容和风格
    '''
    style_extractor = vgg_layers(style_layers)
    style_outputs = style_extractor(style_image * 255)

    # 查看每层输出的统计信息
    for name, output in zip(style_layers, style_outputs):
        print(name)
        print("  shape: ", output.numpy().shape)
        print("  min: ", output.numpy().min())
        print("  max: ", output.numpy().max())
        print("  mean: ", output.numpy().mean())
        print()




    '''在图像上调用此模型'''
    extractor = StyleContentModel(style_layers, content_layers)
    results = extractor(tf.constant(content_image))
    style_results = results['style']
    print('Styles:')
    for name, output in sorted(results['style'].items()):
        print("  ", name)
        print("    shape: ", output.numpy().shape)
        print("    min: ", output.numpy().min())
        print("    max: ", output.numpy().max())
        print("    mean: ", output.numpy().mean())
        print()

    print("Contents:")
    for name, output in sorted(results['content'].items()):
        print("  ", name)
        print("    shape: ", output.numpy().shape)
        print("    min: ", output.numpy().min())
        print("    max: ", output.numpy().max())
        print("    mean: ", output.numpy().mean())



    '''梯度下降'''
    '''计算每个图像的输出和目标的均方误差，然后取这些损失值的加权和。'''
    style_targets = extractor(style_image)['style']
    content_targets = extractor(content_image)['content']
    # 这两个直接提取出了风格和内容的目标值

    image = tf.Variable(content_image)
    # 这个是目标图像，首先让他和内容图像形状一样

    opt = tf.optimizers.Adam(learning_rate=0.02, beta_1=0.99, epsilon=1e-1)
    # 优化函数，这在train_step方法中被使用到

    style_weight = 1e-2
    content_weight = 1e4
    # 使用两个损失的加权组合来获得总损失，这在style_content_loss方法中被使用到


    '''下面进行一段很长很长的优化'''
    start = time.time()

    epochs = 5
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