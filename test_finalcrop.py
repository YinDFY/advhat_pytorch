from align import detect_face
from skimage import transform as trans
import numpy as np
import cv2
import tensorflow as tf
from off_plane_transformation import projector_mask as projector
from stn import spatial_transformer_network as stn
import skimage.io as io
import torch



def preprocess(img, landmark):
    image_size = [600,600]
    src = 600./112.*np.array([
		[38.2946, 51.6963],
		[73.5318, 51.5014],
		[56.0252, 71.7366],
		[41.5493, 92.3655],
		[70.7299, 92.2041] ], dtype=np.float32)
    dst = landmark.astype(np.float32)
    tform = trans.SimilarityTransform()
    tform.estimate(dst, src)
    M = tform.params[0:2,:]

    warped = cv2.warpAffine(img,M,(image_size[1],image_size[0]), borderValue = 0.0)
    return warped


def draw_landmark(img_np,logo_np):
    """
    :param img_np: torch tensor [h,w,c]
    :param logo_np: torch tensor [b,c,h,w]
    :return: numpy [h,w,c]
    """

    tf.reset_default_graph()
    logo = logo_np.squeeze(0).permute(1,2,0).clone().cpu().detach().numpy().astype(np.float32)
    img = img_np.clone().cpu().detach().numpy().astype(np.float32)
    sess = tf.compat.v1.Session()
    pnet, rnet, onet = detect_face.create_mtcnn(sess, None)
    threshold = [0.6, 0.7, 0.7]
    factor = 0.709

    _minsize = min(min(img.shape[0] // 5, img.shape[1] // 5), 80)
    bounding_boxes, points = detect_face.detect_face(img, _minsize, pnet, rnet, onet, threshold, factor)
    assert bounding_boxes.size > 0
    points = points[:, 0]
    landmark = points.reshape((2, 5)).T
    warped = preprocess(img, landmark)

    #io.imsave("./test_align_img/" + '1_aligned.png', warped)
    alpha2 = np.random.uniform(-1., 1., size=(1, 1)) / 180. * np.pi
    x1 = np.random.uniform(0 - 600. / 112., 0 + 600. / 112., size=(1, 1))
    y1 = np.random.uniform(-15. - 600. / 112., -15. + 600. / 112., size=(1, 1))
    if True:
        logo_mask = logo
        logo_mask = np.expand_dims(logo_mask,axis=0)

        logo = tf.placeholder(tf.float32, shape=[1, 400, 900, 3])
        param = tf.placeholder(tf.float32, shape=[1, 1])
        ph = tf.placeholder(tf.float32, shape=[1, 1])
        result = projector(logo,param, ph)

        face_input = tf.placeholder(tf.float32, shape=[1, 600, 600, 3])
        theta = tf.placeholder(tf.float32, shape=[1, 6])
        prepared = stn(result, theta)
        united = prepared[:,300:,150:750]+(face_input * (1 - tf.clip_by_value(prepared[:, 300:, 150:750]*100,0., 1.)) )

        #theta2 = tf.placeholder(tf.float32, shape=[1, 6])

        final_crop = tf.clip_by_value(united, 0., 1.)
        img_with_mask = sess.run(final_crop, feed_dict={ph: [[17.]], logo: logo_mask, param: [[0.0013]], \
                                                    face_input: np.expand_dims(warped/255.0 , 0), \
                                                    theta: 1. / 0.465 * np.array(
                                                        [[1., 0., -0. / 450., 0., 1., 15. / 450.]])})[0]

        return img_with_mask
        #io.imsave("test_align_img/"+'_mask.png', img_with_mask)

if __name__ == "__main__":
    draw_landmark()