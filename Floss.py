im1 = gt_image
im2 = out_image

img1_s1 = tf.dtypes.cast(im1, tf.complex64)
img2_s1 = tf.dtypes.cast(im2, tf.complex64)


FFT1_s1 = tf.signal.fft3d(img1_s1)
FFT2_s1 = tf.signal.fft3d(img2_s1)
first_stage = tf.math.reduce_mean(tf.abs(FFT1_s1-FFT2_s1))


im1_s2 = tf.image.resize_bicubic(im1, (tf.constant(256, tf.int32), tf.constant(256, tf.int32)))
im2_s2 = tf.image.resize_bicubic(im2, (tf.constant(256, tf.int32), tf.constant(256, tf.int32)))

img1_s2 = tf.dtypes.cast(im1_s2, tf.complex64)
img2_s2 = tf.dtypes.cast(im2_s2, tf.complex64)

FFT1_s2 = tf.signal.fft3d(img1_s2)
FFT2_s2 = tf.signal.fft3d(img2_s2)
second_stage = tf.math.reduce_mean(tf.abs(FFT1_s2-FFT2_s2))


im1_s3 = tf.image.resize_bicubic(im1, (tf.constant(128, tf.int32), tf.constant(128, tf.int32)))
im2_s3 = tf.image.resize_bicubic(im2, (tf.constant(128, tf.int32), tf.constant(128, tf.int32)))

img1_s3 = tf.dtypes.cast(im1_s3, tf.complex64)
img2_s3 = tf.dtypes.cast(im2_s3, tf.complex64)

FFT1_s3 = tf.signal.fft3d(img1_s3)
FFT2_s3 = tf.signal.fft3d(img2_s3)
third_stage = tf.math.reduce_mean(tf.abs(FFT1_s3-FFT2_s3))  

#is actually FFT-loss (multistage)
dct_loss = (first_stage + (second_stage) + (third_stage))/1200
