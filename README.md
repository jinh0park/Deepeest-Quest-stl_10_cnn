# Deepest Quest  1

박진호 (jinh0park@naver.com)
---

## (a) 기본 문제

요구사항 충족 여부:

1. \# of paramenters <= 5M    --> O
2. Top 1 accuracy >= 65%    --> O
3. Use Learning Rate Scheduling    --> O
4. Use as least two data augmentation    --> O
5. Use seperate training set, validation set, and test set_random_set    --> O

---

## 0. 모델 설명

1. 단순한 CNN 모델
2. Layers:
    1. Input Layer: $[96, 96, 1]$ (rgb -> grey scale 변환 거침)
    2. Conv Layer-1: $[96, 96, 64]$
    3. Max-Pool Layer-1: $[32, 32, 64]$
    4. Conv Layer-2: $[32, 32, 64]$
    5. Max-Pool Layer-2: $[16, 16, 64]$
    6. Conv Layer-3: $[16, 16, 128]$
    7. Max-Pool Layer-3: $[8, 8, 128]$
    8. 2. Conv Layer-4: $[8, 8, 256]$
    9. Max-Pool Layer-4: $[4, 4, 256]$
    10. FC Layer: $625$ units

---

### 1. \# of paramenters <= 5M    --> O

https://stackoverflow.com/questions/38160940/how-to-count-total-number-of-trainable-parameters-in-a-tensorflow-model

code:

    # parameter 개수 측정
    print("----------parameter count start-----------")
    total_parameters = 0
    for variable in tf.trainable_variables():
    # shape is an array of tf.Dimension
    shape = variable.get_shape()
    print(shape)
    print(len(shape))
    variable_parameters = 1
    for dim in shape:
        print(dim)
        variable_parameters * = dim.value
    print(variable_parameters)
    total_parameters += variable_parameters
    print(total_parameters)
    print("----------parameter count end-----------")

output:

    ----------parameter count start-----------

    (3, 3, 1, 64)

    ...

    10

    2973477

    ----------parameter count end-----------

파라미터 수 = 2973477 < 3M!

---

### 2. Top 1 accuracy >= 65%

Test set accuracy Output:

    step 1 Accuracy: 0.688
    step 2 Accuracy: 0.695
    step 3 Accuracy: 0.699
    step 4 Accuracy: 0.725
    Test Accuracy: 0.7018


---

### 3. Use Learning Rate Scheduling    --> O

Exponential Decay를 사용하였습니다.

    starter_learning_rate = 0.001
    learning_rate = tf.train.exponential_decay(starter_learning_rate, step,
                                                    20000, 0.9, staircase=True)

---

### 4. Use as least two data augmentation    --> O

https://medium.com/ymedialabs-innovation/data-augmentation-techniques-in-cnn-using-tensorflow-371ae43d5be9

Flip and Central Scale 두 가지 방법을 이용하였습니다.
Augmentation 이후에는 Train set을 shuffle하여 Augmentation에 의한 편향성을 없앴습니다.

    def flip_images(X_imgs):
        X_flip = []
        tf.reset_default_graph()
        X = tf.placeholder(tf.float32, shape=(96, 96, 1))
        tf_img1 = tf.image.flip_left_right(X)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for img in X_imgs:
                flipped_imgs = sess.run([tf_img1], feed_dict={X: img})
                X_flip.extend(flipped_imgs)
        X_flip = np.array(X_flip, dtype=np.float32)
        return X_flip


    def central_scale_images(X_imgs, scales):
        # Various settings needed for Tensorflow operation
        boxes = np.zeros((len(scales), 4), dtype=np.float32)
        for index, scale in enumerate(scales):
            x1 = y1 = 0.5 - 0.5 * scale  # To scale centrally
            x2 = y2 = 0.5 + 0.5 * scale
            boxes[index] = np.array([y1, x1, y2, x2], dtype=np.float32)
        box_ind = np.zeros((len(scales)), dtype=np.int32)
        crop_size = np.array([96, 96], dtype=np.int32)

        X_scale_data = []
        tf.reset_default_graph()
        X = tf.placeholder(tf.float32, shape=(1, 96, 96, 1))
        # Define Tensorflow operation for all scales but only one base image at a time
        tf_img = tf.image.crop_and_resize(X, boxes, box_ind, crop_size)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            for img_data in X_imgs:
                batch_img = np.expand_dims(img_data, axis=0)
                scaled_imgs = sess.run(tf_img, feed_dict={X: batch_img})
                X_scale_data.extend(scaled_imgs)

        X_scale_data = np.array(X_scale_data, dtype=np.float32)
        return X_scale_data

함수 실제 적용:

    # data augmentation 과정
    flipped = flip_images(images_train)
    centered = central_scale_images(images_train, [0.9])
    images_train = np.vstack((images_train, flipped, centered))
    labels_train_onehot = np.vstack((labels_train_onehot, labels_train_onehot, labels_train_onehot))
    images_train, labels_train_onehot = unison_shuffled_copies(images_train, labels_train_onehot)

---

### 5. Use seperate training set, validation set, and test set_random_set    --> O

    # test set을 test set과 validation set으로 분리
    images_test_count = images_test.shape[0]
    test_valid_point = int(images_test_count * 0.9)

    labels_valid_onehot = labels_test_onehot[test_valid_point:]
    images_valid = images_test[test_valid_point:]

    labels_test_onehot = labels_test_onehot[:test_valid_point]
    images_test = images_test[:test_valid_point]

    # train, test set의 개수, (8000 * 3, 5000 * 0.9)
    images_train_count = images_train.shape[0]
    images_test_count = images_test.shape[0]
    images_valid_count = images_valid.shape[0]

train 과정에서 validation 확인

    for epoch in range(training_epochs):
        avg_cost = 0
        avg_accuracy = 0
        total_batch = int(images_train_count / batch_size)
        valid_batch = int(images_valid_count / batch_size)

        for i in range(total_batch):
            batch_xs, batch_ys = images_train[i * batch_size:(i + 1) * batch_size], labels_train_onehot[
                                                                                    i * batch_size:(i + 1) * batch_size]
            c, _ = train(sess, batch_xs, batch_ys)
            avg_cost += c / total_batch

        for i in range(valid_batch):
            batch_xs, batch_ys = images_valid[i * batch_size:(i + 1) * batch_size], labels_valid_onehot[
                                                                                    i * batch_size:(i + 1) * batch_size]
            acc = get_accuracy(sess, batch_xs, batch_ys)
            avg_accuracy += acc / valid_batch

        print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost), str(datetime.now()),
              'Validation Set Accuracy:{:.4f}'.format(avg_accuracy))
