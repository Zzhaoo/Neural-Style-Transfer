基本思想是分别从内容和风格图像中提取内容和风格特征，并将这两个特征重新组合成为目标图像，之后在线迭代地重建目标图像，依据是生成图像与内容和风格图像之间的loss



内容损失函数使用的是两者通过VGG网络提取的特征之间的均方误差和

均方误差和公式：

![image-20200528210732786](/Users/a1466055840/Library/Application Support/typora-user-images/image-20200528210732786.png)

我们的代码里用的也是这一公式：

```python
 content_loss = tf.add_n([*tf.reduce_mean((content_outputs[name]-content_targets[name])**2) for name in content_outputs.keys()])
```

风格损失函数使用的是两者通过VGG网络提取的特征之间的格拉姆矩阵的均方误差和

均方误差和和上面一样，格拉姆矩阵：

![image-20200528204845192](/Users/a1466055840/Library/Application Support/typora-user-images/image-20200528204845192.png)



```python
def gram_matrix(input_tensor):
  result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
  input_shape = tf.shape(input_tensor)
  num_locations = tf.cast(input_shape[1]*input_shape[2], tf.float32)
  return result/(num_locations)
```

然后还是均方误差和：

```python
    style_loss = tf.add_n([tf.reduce_mean((style_outputs[name]-style_targets[name])**2) for name in style_outputs.keys()])
```

需要注意的是，迭代优化的不是VGG模型的参数，而是我们用内容图片加上噪声后的输入图像x，通过内容损失和风格损失来优化x的像素，可以理解为是直接在输出图像上进行迭代优化