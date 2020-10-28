import tensorflow as tf


def sqrt_wh(box):
    """
    Take the square root of wh regardless of pred or true boxes given
    INPUTS:
    - box:
    OUTPUTS:
    - box_new:
    """
    print("box dtype: ")
    print(box.dtype)
    # if len(box.get_shape().as_list()) == 4:
    box_new = tf.concat([box[:, :, :, : 2], tf.sqrt(box[:, :, :, 2:])], 3)
    # else:
    # print("LABELS HAVE WRONG SHAPE !!!")
    return box_new


def calc_iou(boxes1, boxes2, scope='iou'):
    """calculate ious
    Args:
      boxes1: 5-D tensor [BATCH_SIZE, CELL_SIZE, CELL_SIZE, BOXES_PER_CELL, 4]  ====> (x_center, y_center, w, h)
      boxes2: 5-D tensor [BATCH_SIZE, CELL_SIZE, CELL_SIZE, BOXES_PER_CELL, 4] ===> (x_center, y_center, w, h)
    Return:
      iou: 4-D tensor [BATCH_SIZE, CELL_SIZE, CELL_SIZE, BOXES_PER_CELL]
    """
    with tf.variable_scope(scope):
        # transform (x_center, y_center, w, h) to (x1, y1, x2, y2)
        boxes1_t = tf.stack([boxes1[..., 0] - boxes1[..., 2] / 2.0,
                             boxes1[..., 1] - boxes1[..., 3] / 2.0,
                             boxes1[..., 0] + boxes1[..., 2] / 2.0,
                             boxes1[..., 1] + boxes1[..., 3] / 2.0],
                            axis=-1)

        boxes2_t = tf.stack([boxes2[..., 0] - boxes2[..., 2] / 2.0,
                             boxes2[..., 1] - boxes2[..., 3] / 2.0,
                             boxes2[..., 0] + boxes2[..., 2] / 2.0,
                             boxes2[..., 1] + boxes2[..., 3] / 2.0],
                            axis=-1)

        # calculate the left up point & right down point
        lu = tf.maximum(boxes1_t[..., :2], boxes2_t[..., :2])
        rd = tf.minimum(boxes1_t[..., 2:], boxes2_t[..., 2:])

        # intersection
        intersection = tf.maximum(tf.cast(0.0, tf.float64), tf.cast(rd - lu, tf.float64))
        inter_square = intersection[..., 0] * intersection[..., 1]

        # calculate the boxs1 square and boxs2 square
        square1 = boxes1[..., 2] * boxes1[..., 3]
        square2 = boxes2[..., 2] * boxes2[..., 3]

        union_square = tf.maximum(tf.cast(square1 + square2 - inter_square, tf.float64), tf.cast(1e-10, tf.float64))

    return tf.clip_by_value(inter_square / union_square, 0.0, 1.0)


def loss_fn(predict_array, labels_array, scope='loss_layer'):
    import tensorflow as tf
    batch_size = -1
    cell_size = 7
    boxes_per_cell = 2
    num_class = 20

    with tf.variable_scope(scope):
        predict_classes = predict_array[..., :7 * 7 * 20]
        predict_classes = tf.reshape(predict_classes, [batch_size, cell_size, cell_size, num_class])

        predict_confids = predict_array[..., 7 * 7 * 20:(7 * 7 * 2) + (7 * 7 * 20)]
        predict_confids = tf.reshape(predict_confids, [batch_size, cell_size, cell_size, boxes_per_cell])

        predict_box_1 = predict_array[:, 7 * 7 * 2 + 7 * 7 * 20:7 * 7 * 2 + 7 * 7 * 20 + 7 * 7 * 4]
        predict_box_1 = tf.reshape(predict_box_1, [batch_size, cell_size, cell_size, 4])

        predict_box_2 = predict_array[:, 7 * 7 * 2 + 7 * 7 * 20 + 7 * 7 * 4:]
        predict_box_2 = tf.reshape(predict_box_2, [batch_size, cell_size, cell_size, 4])

        g_th_confid = labels_array[..., 0]
        g_th_confid = tf.reshape(g_th_confid, [batch_size, cell_size, cell_size, 1])

        g_th_box = labels_array[..., 1:5]
        g_th_box = tf.reshape(g_th_box, [batch_size, cell_size, cell_size, 4])






        g_th_classes = labels_array[..., 5:]
        g_th_classes = tf.reshape(g_th_classes, [batch_size, cell_size, cell_size, 20])

        ###############################coord loss##########################################################

        ###############################coord loss##########################################################

        # now, we zero out those predict bboxes which belong to those grids which don't have object in them.
        # so mulitply bitwise by g_th_confidence
        # 7*7*1 --> 7*7*4
        g_th_confid_tiled = tf.tile(g_th_confid, [1, 1, 1, 4])

        # 7*7*4 x 7*7*4 --> 7*7*4
        select_predict_box_1 = tf.multiply(predict_box_1,
                                           g_th_confid_tiled)  # ZERO OUT ALL THE X,Y,W,H WHERE OBJECT DOES NOT EXIST
        select_predict_box_2 = tf.multiply(predict_box_2, g_th_confid_tiled)

        # Out of these selected bboxes, we choose the one box (out of each grid cell)
        # which has the highest IOU with g_th_box
        # 7*7*4 x 7*7*4 --> 7*7*1 MAKE SURE IT WORKS IN COMPUTE_IOU METHOD


        ################################  RECENTLY ADDED START ################################################

        select_predict_box_1 = tf.expand_dims(select_predict_box_1, axis=3)
        select_predict_box_2 = tf.expand_dims(select_predict_box_2, axis=3)

        g_th_box = tf.expand_dims(g_th_box, axis=3)

        ################################  RECENTLY ADDED END ################################################



        iou1 = calc_iou(select_predict_box_1, g_th_box)  # shape is [batch_size, 7 ,7, 1]
        iou2 = calc_iou(select_predict_box_2, g_th_box)

        # Now, we want to choose the select_predict_box which has the higher IOU
        # we get a boolean in return 7*7*1, it is 1 if iou of bbox1 is greater than that of bbox2

        ######################## START FROM HERE ###################################################################

        mask_if_box1_greater = tf.greater(iou1, iou2)
        mask_if_box2_greater = tf.logical_not(mask_if_box1_greater)

        # we tile the mask here
        tiled_mask_if_box1_greater = tf.tile(mask_if_box1_greater, [1, 1, 1, 4])
        tiled_mask_if_box2_greater = tf.tile(mask_if_box2_greater, [1, 1, 1, 4])

        # now, we further select the bboxes depending on the criteria of whether a bbox has a higher IOU
        # now, we zero out those predict bboxes which have lower IOU compared to the other bbox.
        # so mulitply bitwise by mask_if_box1_greater
        # 7*7*4 x 7*7*4 --> 7*7*4

        select_predict_box_1 = tf.squeeze(select_predict_box_1, [3])
        select_predict_box_2 = tf.squeeze(select_predict_box_2, [3])
        g_th_box = tf.squeeze(g_th_box, [3])

        further_select_predict_box_1 = tf.where(tiled_mask_if_box1_greater, select_predict_box_1,
                                                tf.zeros_like(select_predict_box_1))
        further_select_predict_box_2 = tf.where(tiled_mask_if_box2_greater, select_predict_box_2,
                                                tf.zeros_like(select_predict_box_2))

        # just making sure that those g_th_boxes which are not in the grid which has the gt_bbox are zeroed out
        g_th_confid_new = tf.tile(g_th_confid, [1, 1, 1, 4])

        g_th_box = tf.multiply(g_th_box, g_th_confid_new)

        ####square root w and h of these selected boxes
        # output is 7*7*4
        #         sqrt_wh_further_select_predict_box_1 = sqrt_wh(further_select_predict_box_1)
        #         sqrt_wh_further_select_predict_box_2 = sqrt_wh(further_select_predict_box_2)

        #         sqrt_wh_g_th_box = sqrt_wh(g_th_box)
##################################################################################################

        # finally, compute the loss due to bbox
        bbox1_of_loss = tf.subtract(further_select_predict_box_1, g_th_box)
        bbox2_of_loss = tf.subtract(further_select_predict_box_2, g_th_box)

        bbox1_of_loss = tf.where(tiled_mask_if_box1_greater, bbox1_of_loss,
                                 tf.zeros_like(bbox1_of_loss))
        bbox2_of_loss = tf.where(tiled_mask_if_box2_greater, bbox2_of_loss,
                                 tf.zeros_like(bbox2_of_loss))

        bbox1_of_loss = tf.reduce_sum(tf.square(bbox1_of_loss))  # RETURNS A SCALAR
        bbox2_of_loss = tf.reduce_sum(tf.square(bbox2_of_loss))  # RETURNS A SCALAR

        final_box_loss = 5 * tf.add(bbox1_of_loss, bbox2_of_loss)  # where 5 is lamda coord

        # bat_sz = tf.shape(labels_array)[0]
        # bat_sz = tf.cast(bat_sz, tf.float64)
        # final_box_loss = final_box_loss / bat_sz

        #NORMALIZE ALL LOSSES OR DON'T NORMALIZE AT ALL

        ##############################################################################################
        #################OBJECT LOSS#####################

        predict_confids_b1 = predict_confids[:, :, :, 0]
        predict_confids_b2 = predict_confids[:, :, :, 1]

        predict_confids_b1 = tf.expand_dims(predict_confids_b1, -1)
        predict_confids_b2 = tf.expand_dims(predict_confids_b2, -1)

        # Firstly, we zero out those pred_confids which do not lie in the grid cell which contains the object
        predict_confids_b1 = tf.multiply(predict_confids_b1, g_th_confid)
        predict_confids_b2 = tf.multiply(predict_confids_b2, g_th_confid)

        # Now we check if the predict_confids are indeed positive, for that we generate a boolean vector
        is_positive_1 = tf.greater(predict_confids_b1, tf.zeros_like(predict_confids_b1))
        is_positive_2 = tf.greater(predict_confids_b2, tf.zeros_like(predict_confids_b2))

        new_mask_1 = tf.logical_and(mask_if_box1_greater, is_positive_1)
        new_mask_2 = tf.logical_and(mask_if_box2_greater, is_positive_2)


        # now using this mask which makes sure we have values that are positive and have bboxes with higher IOUs
        predict_confids_b1 = tf.where(new_mask_1, predict_confids_b1, tf.zeros_like(predict_confids_b1))
        predict_confids_b2 = tf.where(new_mask_2, predict_confids_b2, tf.zeros_like(predict_confids_b2))


        #I imagine that the values with a hat are the real one read from the label and the one without hat are the predicted ones. So what is the real value from the label for the confidence score for each bbox Ĉ ij ? It is the intersection over union of the predicted bounding box with the one from the label.

        #The natural confidence score value is:

        #for a positive position, the intersection over union (IOU) of the predicted box with the ground truth

        #for a negative position, zero.


        # Now, we want to find the confidence to be used in the loss_function formula
        # first we find the ious for each type of bboxes using their respective masks
        ious_for_b1 = tf.where(mask_if_box1_greater, iou1, tf.zeros_like(iou1))
        ious_for_b2 = tf.where(mask_if_box2_greater, iou2, tf.zeros_like(iou2))
        # ious_for_b1 is now 1*7*7; we now convert it to 1*7*7*1
        #         ious_for_b1 = tf.expand_dims(ious_for_b1, -1)
        #         ious_for_b2 = tf.expand_dims(ious_for_b2, -1)

        g_th_confid_for_loss_1 = tf.multiply(ious_for_b1, g_th_confid)  ###############???????????????????????
        g_th_confid_for_loss_2 = tf.multiply(ious_for_b2, g_th_confid)

        gt_b1 = tf.where(new_mask_1, g_th_confid_for_loss_1, tf.zeros_like(g_th_confid_for_loss_1))
        gt_b2 = tf.where(new_mask_2, g_th_confid_for_loss_2, tf.zeros_like(g_th_confid_for_loss_2))

        loss_b1 = tf.square(predict_confids_b1 - gt_b1)
        loss_b2 = tf.square(predict_confids_b2 - gt_b2)

        total_loss_object = loss_b1 + loss_b2
        final_loss_object = tf.reduce_sum(total_loss_object)

        ##########################################################################
        ##########################################################################
        ###########     NO OBJECT LOSS####################################################

        cast_for_obj = tf.cast(g_th_confid, bool)
        cast_for_noobj = tf.logical_not(cast_for_obj)

        #doesn't include the check for positives
        b1_cast_for_noobj = tf.logical_or(cast_for_noobj, tf.logical_and(cast_for_obj, mask_if_box2_greater))
        b2_cast_for_noobj = tf.logical_or(cast_for_noobj, tf.logical_and(cast_for_obj, mask_if_box1_greater))

        predict_confids_b1 = predict_confids[..., 0]
        predict_confids_b2 = predict_confids[..., 1]

        predict_confids_b1_no_obj = tf.expand_dims(predict_confids_b1, -1)
        predict_confids_b2_no_obj = tf.expand_dims(predict_confids_b2, -1)



        # based on these masks, we want to select predict_confid based on these masks
        predict_confids_b1_no_obj = tf.where(b1_cast_for_noobj, predict_confids_b1_no_obj,
                                             tf.zeros_like(predict_confids_b1_no_obj))
        predict_confids_b2_no_obj = tf.where(b2_cast_for_noobj, predict_confids_b2_no_obj,
                                             tf.zeros_like(predict_confids_b2_no_obj))


        # now, we want to reverse the values of g_th_confid since we want the values according to the
        # no_object
        # /m. Ask vicent.

        #
        # ##############??????????????????
        # g_th_confid_reverse = tf.where(tf.greater(g_th_confid, tf.zeros_like(g_th_confid)),
        #                                tf.zeros_like(g_th_confid), tf.ones_like(g_th_confid))

        # # now, we want to find the confidence to be used in the loss_function formula
        # no_obj_g_th_confid_for_loss_1 = tf.multiply(iou1, g_th_confid_reverse)
        # no_obj_g_th_confid_for_loss_2 = tf.multiply(iou2, g_th_confid_reverse)

        gt_b1_no_obj = tf.where(b1_cast_for_noobj, iou1,
                                tf.zeros_like(iou1))
        gt_b2_no_obj = tf.where(b2_cast_for_noobj, iou2,
                                tf.zeros_like(iou2))

        loss_b1 = tf.square(predict_confids_b1_no_obj - gt_b1_no_obj)
        loss_b2 = tf.square(predict_confids_b2_no_obj - gt_b2_no_obj)

        total_loss_no_object = loss_b1 + loss_b2
        final_loss_no_object = 0.5 * tf.reduce_sum(total_loss_no_object)  # where 0.5 is lambda_no_obj












        # # First, we take the opposite of masks above
        # no_obj_new_mask_1 = tf.logical_not(new_mask_1)
        # no_obj_new_mask_2 = tf.logical_not(new_mask_2)
        #
        # predict_confids_b1 = predict_confids[..., 0]
        # predict_confids_b2 = predict_confids[..., 1]
        #
        # predict_confids_b1_no_obj = tf.expand_dims(predict_confids_b1, -1)
        # predict_confids_b2_no_obj = tf.expand_dims(predict_confids_b2, -1)
        #
        #
        # # based on these masks, we want to select predict_confid based on these masks
        # predict_confids_b1_no_obj = tf.where(no_obj_new_mask_1, predict_confids_b1_no_obj,
        #                                      tf.zeros_like(predict_confids_b1_no_obj))
        # predict_confids_b2_no_obj = tf.where(no_obj_new_mask_2, predict_confids_b2_no_obj,
        #                                      tf.zeros_like(predict_confids_b2_no_obj))
        #
        #
        # # now, we want to reverse the values of g_th_confid since we want the values according to the
        # # no_object
        # # /m. Ask vicent.
        # g_th_confid_reverse = tf.where(tf.greater(g_th_confid, tf.zeros_like(g_th_confid)),
        #                                tf.zeros_like(g_th_confid), tf.ones_like(g_th_confid))
        #
        # # now, we want to find the confidence to be used in the loss_function formula
        # no_obj_g_th_confid_for_loss_1 = tf.multiply(ious_for_b1, g_th_confid_reverse)
        # no_obj_g_th_confid_for_loss_2 = tf.multiply(ious_for_b2, g_th_confid_reverse)
        #
        # gt_b1_no_obj = tf.where(no_obj_new_mask_1, no_obj_g_th_confid_for_loss_1,
        #                         tf.zeros_like(no_obj_g_th_confid_for_loss_1))
        # gt_b2_no_obj = tf.where(no_obj_new_mask_2, no_obj_g_th_confid_for_loss_2,
        #                         tf.zeros_like(no_obj_g_th_confid_for_loss_2))
        #
        # loss_b1 = tf.square(predict_confids_b1_no_obj - gt_b1_no_obj)
        # loss_b2 = tf.square(predict_confids_b2_no_obj - gt_b2_no_obj)
        #
        # total_loss_no_object = loss_b1 + loss_b2
        # final_loss_no_object = 0.5 * tf.reduce_sum(total_loss_object)  # where 0.5 is lanbda_no_obj


        #######################################################
        ############ class loss #################################

        # 7*7*20 - 7*7*20 = 7*7*20

        loss_classes = tf.square(g_th_classes - predict_classes)

        # # we need to zero out those grid cell which do not have the object
        # g_th_confid_ind = tf.where(tf.greater(g_th_confid, tf.zeros_like(g_th_confid)), tf.ones_like(g_th_confid),
        #                            tf.zeros_like(g_th_confid))


        tiled_g_th_confid = tf.tile(g_th_confid, [1, 1, 1, 20])


        select_loss_classes = tf.multiply(loss_classes, tiled_g_th_confid)

        sum_class_losses = tf.reduce_sum(select_loss_classes)

        #############################################################################
        print("DONE WITH THE LOSS FUNCTION!!!!")
        # tf.losses.add_loss(final_box_loss)
        # tf.losses.add_loss(final_loss_object)
        # tf.losses.add_loss(final_loss_no_object)
        # tf.losses.add_loss(sum_class_losses)

        summ = final_box_loss + final_loss_object + final_loss_no_object + sum_class_losses

        tf.summary.scalar('class_loss', sum_class_losses)
        tf.summary.scalar('object_loss', final_loss_object)
        tf.summary.scalar('noobject_loss', final_loss_no_object)
        tf.summary.scalar('coord_loss', final_box_loss)

        return summ


# predict_array = tf.random_uniform((2, 7 * 7 * 30), dtype=tf.float64)
# labels_array = tf.random_uniform((2, 7, 7, 25), dtype=tf.float64)
#
# loss_fn(predict_array, labels_array, scope='loss_layer')








