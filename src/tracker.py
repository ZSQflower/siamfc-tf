
import tensorflow as tf
print('Using Tensorflow '+tf.__version__)
import matplotlib.pyplot as plt
import sys
# Uncommenting next line doesnt help when calling Tracker from C++
#sys.path.append('../')
import os
import csv
import numpy as np
from PIL import Image
import time

import src.siamese as siam
from src.visualization import show_frame, show_crops, show_scores


# gpu_device = 2
# os.environ['CUDA_VISIBLE_DEVICES'] = '{}'.format(gpu_device)

# read default parameters and override with custom ones
#def tracker(hp, run, design, frame_name_list, pos_x, pos_y, target_w, target_h, final_score_sz, filename, image, templates_z, scores, start_frame):
def tracker(hp, run, design, frame_name_list, objects, final_score_sz, filename, image, templates_z, scores, start_frame):
    num_frames = np.size(frame_name_list)
    # stores tracker's output for evaluation
    bboxes = [np.zeros((len(objects),4)) for i in range(0, num_frames)]
    
    # save first frame position (from ground-truth)
    for i in range(len(objects)):
        pos_x = objects[i][0]
        pos_y = objects[i][1]
        target_w = objects[i][2]
        target_h = objects[i][3]
        #bboxes[0][i] = objects[i]
        bboxes[0][i,:] = pos_x-target_w/2, pos_y-target_h/2, target_w, target_h
   
    scale_factors = hp.scale_step**np.linspace(-np.ceil(hp.scale_num/2), np.ceil(hp.scale_num/2), hp.scale_num)
    # cosine window to penalize large displacements
    hann_1d = np.expand_dims(np.hanning(final_score_sz), axis=0)
    penalty = np.transpose(hann_1d) * hann_1d
    penalty = penalty / np.sum(penalty)

    """    
    # I don't see this values in any part of the code, so I assume it's safe to comment them
    # thresholds to saturate patches shrinking/growing    
    min_z = hp.scale_min * z_sz
    max_z = hp.scale_max * z_sz
    min_x = hp.scale_min * x_sz
    max_x = hp.scale_max * x_sz
    """

    # This variables use the box data, so they should have different values per object
    # Object Box information
    pos_x = [0]*len(objects)
    pos_y = [0]*len(objects)
    target_w = [0]*len(objects)
    target_h = [0]*len(objects)
    # Other variables
    context = [0]*len(objects)
    z_sz = [0]*len(objects)
    x_sz = [0]*len(objects)
    for o in range(len(objects)):
        pos_x[o] = objects[o][0]
        pos_y[o] = objects[o][1]
        target_w[o] = objects[o][2]
        target_h[o] = objects[o][3]
        context[o] = design.context*(target_w[o]+target_h[o])
        z_sz[o] = np.sqrt(np.prod((target_w[o]+context[o])*(target_h[o]+context[o])))
        x_sz[o] = float(design.search_sz) / design.exemplar_sz * z_sz[o]
    
    # run_metadata = tf.RunMetadata()
    # run_opts = {
    #     'options': tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE),
    #     'run_metadata': run_metadata,
    # }
    run_opts = {}

    # with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        # Coordinate the loading of image files.
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        
        # save first frame position (from ground-truth)
        #bboxes[0,:] = pos_x-target_w/2, pos_y-target_h/2, target_w, target_h                
        
        scores_ = [0]*len(objects)
        templates_z_ = [0]*len(objects)
        for o in range(len(objects)):
            #print ('Box {} template! x: {}, y: {}, z_sz: {}'.format(o, pos_x[o], pos_y[o], z_sz[o]) )
            image_, templates_z_[o] = sess.run([image, templates_z], feed_dict={
            #image_, templates_z_res = sess.run([image, templates_z], feed_dict={
                                                                    siam.pos_x_ph: pos_x[o],
                                                                    siam.pos_y_ph: pos_y[o],
                                                                    siam.z_sz_ph: z_sz[o],
                                                                    filename: frame_name_list[0]})
            #templates_z_[o] = templates_z_res
        
        t_start = time.time()

        # Get an image from the queue
        for i in range(1, num_frames):
            for o in range(len(objects)):
                scaled_exemplar = z_sz[o] * scale_factors
                scaled_search_area = x_sz[o] * scale_factors
                scaled_target_w = target_w[o] * scale_factors
                scaled_target_h = target_h[o] * scale_factors
                
                image_, scores_ = sess.run(
                    [image, scores],
                    feed_dict={
                        siam.pos_x_ph: pos_x[o],
                        siam.pos_y_ph: pos_y[o],
                        siam.x_sz0_ph: scaled_search_area[0],
                        siam.x_sz1_ph: scaled_search_area[1],
                        siam.x_sz2_ph: scaled_search_area[2],
                        templates_z: np.squeeze(templates_z_[o]),
                        filename: frame_name_list[i],
                    }, **run_opts)
                scores_ = np.squeeze(scores_)
                # penalize change of scale
                scores_[0,:,:] = hp.scale_penalty*scores_[0,:,:]
                scores_[2,:,:] = hp.scale_penalty*scores_[2,:,:]
                # find scale with highest peak (after penalty)
                new_scale_id = np.argmax(np.amax(scores_, axis=(1,2)))
                # update scaled sizes
                x_sz[o] = (1-hp.scale_lr)*x_sz[o] + hp.scale_lr*scaled_search_area[new_scale_id]
                target_w[o] = (1-hp.scale_lr)*target_w[o] + hp.scale_lr*scaled_target_w[new_scale_id]
                target_h[o] = (1-hp.scale_lr)*target_h[o] + hp.scale_lr*scaled_target_h[new_scale_id]
                # select response with new_scale_id
                score_ = scores_[new_scale_id,:,:]
                score_ = score_ - np.min(score_)
                score_ = score_/np.sum(score_)
                # apply displacement penalty
                score_ = (1-hp.window_influence)*score_ + hp.window_influence*penalty
                pos_x[o], pos_y[o] = _update_target_position(pos_x[o], pos_y[o], score_, final_score_sz, design.tot_stride, design.search_sz, hp.response_up, x_sz[o])
                # convert <cx,cy,w,h> to <x,y,w,h> and save output
                #bboxes[i,:] = pos_x-target_w/2, pos_y-target_h/2, target_w, target_h
                bboxes[i][o,:] = pos_x[o]-target_w[o]/2, pos_y[o]-target_h[o]/2, target_w[o], target_h[o]
                if hp.z_lr>0:
                    new_templates_z_ = sess.run([templates_z], feed_dict={
                                                                    siam.pos_x_ph: pos_x[o],
                                                                    siam.pos_y_ph: pos_y[o],
                                                                    siam.z_sz_ph: z_sz[o],
                                                                    image: image_
                                                                    })
                    templates_z_[o]=(1-hp.z_lr)*np.asarray(templates_z_[o]) + hp.z_lr*np.asarray(new_templates_z_)
                
                # update template patch size
                z_sz[o] = (1-hp.scale_lr)*z_sz[o] + hp.scale_lr*scaled_exemplar[new_scale_id]
            
            if run.visualization:
                show_frame(image_, bboxes[i], 1)

        t_elapsed = time.time() - t_start
        speed = num_frames/t_elapsed

        # Finish off the filename queue coordinator.
        coord.request_stop()
        coord.join(threads) 

        # from tensorflow.python.client import timeline
        # trace = timeline.Timeline(step_stats=run_metadata.step_stats)
        # trace_file = open('timeline-search.ctf.json', 'w')
        # trace_file.write(trace.generate_chrome_trace_format())

    plt.close('all')

    return bboxes, speed


def _update_target_position(pos_x, pos_y, score, final_score_sz, tot_stride, search_sz, response_up, x_sz):
    # find location of score maximizer
    p = np.asarray(np.unravel_index(np.argmax(score), np.shape(score)))
    # displacement from the center in search area final representation ...
    center = float(final_score_sz - 1) / 2
    disp_in_area = p - center
    # displacement from the center in instance crop
    disp_in_xcrop = disp_in_area * float(tot_stride) / response_up
    # displacement from the center in instance crop (in frame coordinates)
    disp_in_frame = disp_in_xcrop *  x_sz / search_sz
    # *position* within frame in frame coordinates
    pos_y, pos_x = pos_y + disp_in_frame[0], pos_x + disp_in_frame[1]
    return pos_x, pos_y


