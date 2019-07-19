import tensorflow as tf
import matplotlib.pyplot as plt
from cifar10 import *

# ckpt2pb
def freeze_graph(input_checkpoint, output_graph):
    output_node_name = "finally/finally/_logits/BiasAdd"
    saver = tf.train.import_meta_graph(input_checkpoint+".meta", clear_devices=True)
    graph = tf.get_default_graph()
    input_graph_def = graph.as_graph_def()

    with tf.Session() as sess:
        saver.restore(sess, input_checkpoint)

        # fix batch norm nodes' bug
        # Using the tf.layers.batch_normalization rather the tf.contrib.layers.batch_norm may help? no test. 
        for node in input_graph_def.node:
            if node.op == 'RefSwitch':
                node.op = 'Switch'
                for index in range(len(node.input)):
                    # the 'and "Switch" not in node.input[index]' is important
                    if 'moving_' in node.input[index] and "Switch" not in node.input[index]:
                        node.input[index] = node.input[index] + '/read'
            elif node.op == 'AssignSub':
                node.op = 'Sub'
                if 'use_locking' in node.attr: del node.attr['use_locking']
            elif node.op == 'AssignAdd':
                node.op = 'Add'
                if 'use_locking' in node.attr: del node.attr['use_locking']

        output_graph_def = tf.graph_util.convert_variables_to_constants(
            sess=sess,
            input_graph_def=input_graph_def,
            output_node_names=output_node_name.split(",")
        )

        with tf.gfile.GFile(output_graph, "wb") as f:
            f.write(output_graph_def.SerializeToString())
        print("%d ops in the final graph." % len(output_graph_def.node))


input_ckpt = "ckpt path"
out_pb_path = "pb path"
# freeze_graph(input_ckpt, out_pb_path)

# use pb file
def freeze_graph_test(pb_path, batch_size=1, image_size=224):
    with tf.Graph().as_default():
        out_graph_def = tf.GraphDef()
        with open(pb_path, "rb") as f:
            out_graph_def.ParseFromString(f.read())
            tf.import_graph_def(out_graph_def, name="")

        # trainDataset = testpb.getDataset(testpb.valTfRecord, testpb.BATCH_SIZE)
        # train_iterator = trainDataset.make_one_shot_iterator()
        # train_images, train_labels = train_iterator.get_next()
        # print(train_labels.get_shape())
        
        # read cifar10 data
        train_x, train_y, test_x, test_y = prepare_data()
        train_x, test_x = color_preprocessing(train_x, test_x)
        test_batch_x = test_x[0: batch_size]
        test_batch_y = test_y[0: batch_size]
        print("test_batch_x.shape: {}, test_batch_y.shape: {}".format(
            np.shape(test_batch_x), np.shape(test_batch_y)))

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            input_image_tensor = sess.graph.get_tensor_by_name("input_x:0")
            # the label placeholder dose not need to feed
            # input_label_tensor = sess.graph.get_tensor_by_name("input_y:0")
            is_trianing_tensor = sess.graph.get_tensor_by_name("is_training_1:0")
            output_tensor_name = sess.graph.get_tensor_by_name("finally/finally/_logits/BiasAdd:0")
            arg_max_logit = tf.arg_max(output_tensor_name, 1)
            arg_max_label = tf.arg_max(test_batch_y, 1)

            correct = tf.equal(arg_max_logit, arg_max_label)
            accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

            # time_now = time.time()
            # train_imagess, train_labelss = sess.run([train_images, train_labels])
            # print(train_imagess.shape)
            # cost_time = time.time() - time_now
            # print("get image and label cost time: ", cost_time)

            time_now = time.time()
            logits, labels, accuracy = sess.run([arg_max_logit, arg_max_label, accuracy], feed_dict={
                input_image_tensor: test_batch_x, is_trianing_tensor: False})
            cost_time = time.time() - time_now
            print("sess.run() cost time: ", cost_time)
            print("accuracy: ", accuracy)
            # plot
            plt.figure(1)
            for i in range(batch_size):
                image = test_batch_x[i, :, :, :]
                image = np.reshape(image, (image_size, image_size, 3))
                plt.subplot(1, batch_size, i+1)
                plt.imshow(image.astype(np.uint8))
                plt.title(logits[i])
                plt.xlabel(labels[i])
            plt.show()
BATCH_SIZE = 16
IMG_SIZE = 32
freeze_graph_test(out_pb_path, batch_size=BATCH_SIZE, image_size=IMG_SIZE)

# change the BATCH_SIZE to test more data
