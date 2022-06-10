import cv2
import os
import numpy as np
import polyscope as ps

from utils.data_utils import read_obj

def get_texts_colors_locs(mode, h, w, T=None):
    if mode == 'compare_all':
        texts = ['Source', 'NeuralMLS', 'MLS', 'KPD']
        colors = [(0, 0, 204), (204, 0, 0), (51, 179, 128), (179, 128, 51)]
        locs = [(int(w/10), int(h/10)), (int(w/10), int(h/10)+60), (int(w/10), int(h/10)+110), (int(w/10), int(h/10)+160)]

    if mode == 'compare_mls':
        # texts = ['Source shape', 'NeuralMLS', 'MLS']
        # colors = [(0, 0, 204), (204, 0, 0), (51, 179, 128)]
        # locs = [(int(w/10), int(h/10)), (int(w/10), int(h/10)+35), (int(w/10), int(h/10)+70)]
        texts = ['NeuralMLS', 'Source', 'MLS'] # for chair1 and plane1
        colors = [(179, 115, 0), (0, 94, 214), (51, 179, 128)] # for chair1 and plane1
        locs = [(int(w/2 - w/3 - w/30), int(h/8)), (int(w/2 - w/13), int(h/8)), (int(w/2 + w/3 - w/7), int(h/8))] # for chair1
        
    # if mode == 'weight_animation':
    #     texts = ['NeuralMLS', 'MLS'] # for chair2 weight animation
    #     colors = [(179, 115, 0), (51, 179, 128)] # for chair2 and plane2 weight animation
    #     locs = [(int(w/2 - w/4), int(h/8)), (int(w/2 + w/12), int(h/8))] # for chair2 weight animation
        
    if mode == 'temperature':
        texts = ['T = {:1.2}'.format(T)]
        colors = [(179, 115, 0)]
        locs = [(int(w/2 - w/8), int(h/8))]
        
    return texts, colors, locs

def putLegendOnImage(img_path, mode, T=None):
    # font
    font = cv2.FONT_HERSHEY_SIMPLEX
    # fontScale
    fontScale = 2
    # Line thickness of 2 px
    thickness = 3
    # read image
    image = cv2.imread(img_path)
    h, w, _ = image.shape
    texts, colors, locs = get_texts_colors_locs(mode, h, w, T)

    # Using cv2.putText() method
    for i in range(len(texts)):
        text = texts[i]
        color = colors[i]
        loc = locs[i]
        image = cv2.putText(image, text, loc, font, fontScale, color, thickness, cv2.LINE_AA, False)

    if mode == 'compare_all':
        h_offset = 170
        w_offset = 550

        # represents the top left corner of rectangle
        start_point = (int(w/10)-5, int(h/10)-60)
        # represents the bottom right corner of rectangle
        end_point = (int(w/10)+w_offset, int(h/10)+h_offset)
        # Black color in BGR
        color = (0, 0, 0)
        # Thickness
        thickness = 2
        # Using cv2.rectangle() method
        image = cv2.rectangle(image, start_point, end_point, color, thickness)

    cv2.imwrite(img_path, image)
    
def take_screenshots(base_fn, image_num, mode, num_of_times=1, addLegend=False, T=None):
    for _ in range(num_of_times):
        file_name = base_fn.format(image_num)
        ps.screenshot(filename=file_name, transparent_bg=False)
        if addLegend:
            putLegendOnImage(file_name, mode, T)
        image_num += 1

    return image_num

def create_files_names(files_dir, step, shape, is_test, is_interactive):
    if is_test:
        source_path = files_dir + '{}-Sa.obj'.format(shape)
        deformed_path = files_dir + '{}-Sab.obj'.format(shape)
        target_path = files_dir + '{}-Sab_MLS.obj'.format(shape)
        arap_path = files_dir + '{}-Sab_ARAP.obj'.format(shape)
        kpd_path = files_dir + '{}-Sab_KPD.obj'.format(shape)
        source_cp_path = files_dir + 'source_points.obj'
        target_cp_path = files_dir + 'target_points.obj'
    elif is_interactive:
        source_path = files_dir + '{}-Sa.obj'.format(shape)
        deformed_path = files_dir + '{}-Sab_0.obj'.format(shape)
        target_path = files_dir + '{}-Sab_CLASSIC_0.obj'.format(shape)
        source_cp_path = files_dir + 'source_points.obj'
        target_cp_path = files_dir + 'target_points_0.obj'
    else:
        source_path = files_dir + 'step-{:06}-shape-{}-Sa.obj'.format(step, shape)
        deformed_path = files_dir + 'step-{:06}-shape-{}-Sab.obj'.format(step, shape)
        target_path = files_dir + 'shape-{}-Sab_CLASSIC.obj'.format(shape)
        arap_path = files_dir + 'shape-{}-Sab_ARAP.obj'.format(shape)
        kpd_path = files_dir + 'shape-{}-Sab_KPD.obj'.format(shape)
        source_cp_path = files_dir + 'step-{:06}-control_points.obj'.format(step)
        target_cp_path = files_dir + 'step-{:06}-deformed_control_points.obj'.format(step)
    
    return source_path, deformed_path, target_path, arap_path, kpd_path, source_cp_path, target_cp_path

def read_files(files_dir, step, shape, is_test, with_faces, is_interactive=False):
    source_path, deformed_path, target_path, arap_path, kpd_path, source_cp_path, target_cp_path = create_files_names(files_dir, step, shape, is_test, is_interactive)

    s_face = None
    d_face = None
    t_face = None
    if with_faces:
        source, s_face = read_obj(source_path)
        source, s_face = np.array(source), np.array(s_face)   
        deformed, d_face = read_obj(deformed_path)    
        deformed, d_face = np.array(deformed), np.array(d_face)   
        target, t_face = read_obj(target_path)    
        target, t_face = np.array(target), np.array(t_face)   
        if os.path.isfile(arap_path):
            arap, a_face = read_obj(arap_path)    
            arap, a_face = np.array(arap), np.array(a_face)   
        else:
            arap, a_face = None, None
        if os.path.isfile(kpd_path):
            kpd, k_face = read_obj(kpd_path)    
            kpd, k_face = np.array(kpd), np.array(k_face)   
        else:
            kpd, k_face = None, None
        source_cp, _ = read_obj(source_cp_path)    
        source_cp = np.array(source_cp)   
        target_cp, _ = read_obj(target_cp_path)    
        target_cp = np.array(target_cp)   
    else:
        source = np.genfromtxt(source_path)    
        deformed = np.genfromtxt(deformed_path)    
        target = np.genfromtxt(target_path)    
        source_cp = np.genfromtxt(source_cp_path)    
        target_cp = np.genfromtxt(target_cp_path)    

    return source, deformed, target, arap, kpd, source_cp, target_cp, source_path, s_face, d_face, t_face, a_face, k_face
    
def parse_and_normalize(source, deformed, target, arap, kpd, source_cp, target_cp, with_faces):
    if with_faces:
        offset = 0
    else:
        offset = 1
        
    # parse
    s_color = source[:, offset+3:].astype('float32')
    source = source[:, offset:offset+3]
    d_color = deformed[:, offset+3:].astype('float32')
    deformed = deformed[:, offset:offset+3]    
    t_color = target[:, offset+3:].astype('float32')
    target = target[:, offset:offset+3]    
    if arap is not None:
        a_color = arap[:, offset+3:].astype('float32')
        arap = arap[:, offset:offset+3]    
    else:
        a_color = None
    if kpd is not None:
        k_color = kpd[:, offset+3:].astype('float32')
        kpd = kpd[:, offset:offset+3]    
    else:
        k_color = None
    scp_color = source_cp[:, offset+3:].astype('float32')
    source_cp = source_cp[:, offset:offset+3]    
    tcp_color = target_cp[:, offset+3:].astype('float32')
    target_cp = target_cp[:, offset:offset+3]

    # normalize - center to middle of bounding box
    s_max = np.max(source, axis=0)
    s_min = np.min(source, axis=0)
    s_mid = (s_max + s_min)/2
    source = source - s_mid
    deformed = deformed - s_mid
    target = target - s_mid
    if arap is not None:
        arap = arap - s_mid
    if kpd is not None:
        kpd = kpd - s_mid
    source_cp = source_cp - s_mid
    target_cp = target_cp - s_mid

    return s_color, source, d_color, deformed, t_color, target, a_color, arap, k_color, kpd, \
    scp_color, source_cp, tcp_color, target_cp

def register_pointclouds(s_color, source, d_color, deformed, t_color, target, scp_color, source_cp, tcp_color, target_cp, source_radius=0.002):
    # source always enabled
    s_cloud = ps.register_point_cloud("source shape", source, radius=source_radius, enabled=True)
    s_cloud.add_color_quantity("source shape colors", s_color, enabled=True)
    # deformed
    d_cloud = ps.register_point_cloud("deformed shape", deformed, radius=2*source_radius, enabled=False)
    d_cloud.add_color_quantity("deformed shape colors", d_color, enabled=True)
    # target
    t_cloud = ps.register_point_cloud("target shape", target, radius=2*source_radius, enabled=False)
    t_cloud.add_color_quantity("target shape colors", t_color, enabled=True)
    # source CP
    ps_cloud = ps.register_point_cloud("source cp", source_cp, radius=5*source_radius, enabled=True)
    ps_cloud.add_color_quantity("source cp colors", scp_color, enabled=True)
    # target CP
    pt_cloud = ps.register_point_cloud("target cp", target_cp, radius=5*source_radius, enabled=False)
    pt_cloud.add_color_quantity("target cp colors", tcp_color, enabled=True)

    return s_cloud, d_cloud, t_cloud, ps_cloud, pt_cloud

def register_meshes(s_color, source, d_color, deformed, t_color, target, a_color, arap, k_color, kpd, scp_color, source_cp, tcp_color, target_cp, s_face, d_face, t_face, a_face, k_face, source_radius=0.002):
    # source always enabled
    s_cloud = ps.register_surface_mesh("source shape", source, s_face, enabled=True)
    s_cloud.add_color_quantity("source shape colors", s_color, enabled=True)
    # source CP
    ps_cloud = ps.register_point_cloud("source cp", source_cp, radius=5*source_radius, enabled=True)
    ps_cloud.add_color_quantity("source cp colors", scp_color, enabled=True)
    # deformed
    d_cloud = ps.register_surface_mesh("deformed shape", deformed, d_face, enabled=False)
    d_cloud.add_color_quantity("deformed shape colors", d_color, enabled=True)
    # target
    t_cloud = ps.register_surface_mesh("target shape", target, t_face, enabled=False)
    t_cloud.add_color_quantity("target shape colors", t_color, enabled=True)
    # arap
    if arap is not None:
        a_cloud = ps.register_surface_mesh("arap shape", arap, a_face, enabled=False)
        a_cloud.add_color_quantity("arap shape colors", a_color, enabled=True)
    else:
        a_cloud = None
    # keypointDeformer
    if kpd is not None:
        k_cloud = ps.register_surface_mesh("KPD shape", kpd, k_face, enabled=False)
        k_cloud.add_color_quantity("KPD shape colors", k_color, enabled=True)
    else:
        k_cloud = None
    # target CP
    pt_cloud = ps.register_point_cloud("target cp", target_cp, radius=5*source_radius, enabled=False)
    pt_cloud.add_color_quantity("target cp colors", tcp_color, enabled=True)

    return s_cloud, d_cloud, t_cloud, a_cloud, k_cloud, ps_cloud, pt_cloud

def init_camera(i, num_images, init_cam_pos, init_cam_trg, cam_radius, axis='vertical'):
    offset_cam_trg = np.array([0., 0., 0.])
    theta = 2*np.pi * (i/(num_images-1))

    init_cam_pos = np.array(init_cam_pos)
    init_cam_pos = cam_radius * (init_cam_pos / np.linalg.norm(init_cam_pos))

    if axis == 'vertical':
        cam_pos = np.matmul(np.array([[np.cos(theta), 0., np.sin(theta)], [0., 1., 0.], [-np.sin(theta), 0, np.cos(theta)]]), init_cam_pos)
        # cam_pos = np.matmul(np.array([[np.cos(theta), np.sin(theta), 0.], [-np.sin(theta), np.cos(theta), 0.], [0., 0., 1.]]), init_cam_pos)
        # cam_pos = np.matmul(np.array([[1., 0., 0.], [0., np.cos(theta), np.sin(theta)], [0., -np.sin(theta), np.cos(theta)]]), init_cam_pos)
        cam_trg = init_cam_trg + offset_cam_trg
    elif axis == 'horizontal':
        # cam_pos = init_cam_pos + cam_radius * np.array([0., np.sin(theta), np.abs(np.cos(theta))])
        cam_pos = np.matmul(np.array([[1., 0., 0.], [0., np.cos(theta), np.sin(theta)], [0., -np.sin(theta), np.cos(theta)]]), init_cam_pos)
        cam_trg = init_cam_trg + offset_cam_trg

    ps.look_at(cam_pos, cam_trg)

def generate_cp_movement(ps_cloud, source_cp, offsets, base_fn, image_num, num_images_cp=20):
    for j in range(num_images_cp):
        ps_cloud.update_point_positions(source_cp + offsets * (j/(num_images_cp-1)))
        image_num = take_screenshots(base_fn, image_num, 1)

    return image_num

def generate_first_frames(s_cloud, d_cloud, ps_cloud, pt_cloud, source_cp, offsets, o_color, base_fn, image_num, num_fps):
    image_num = generate_cp_movement(ps_cloud, source_cp, offsets, base_fn, image_num, 3*num_fps)
    ps_cloud.update_point_positions(source_cp)
    pt_cloud.set_enabled()
    # offsets
    ps_cloud.add_vector_quantity("offsets", offsets, vectortype='ambient', color=o_color, radius=0.007, enabled=True)
    image_num = take_screenshots(base_fn, image_num, 2*num_fps)

    ps_cloud.set_enabled(False)
    pt_cloud.set_enabled(False)
    d_cloud.set_enabled()
    image_num = take_screenshots(base_fn, image_num, num_fps)

    s_cloud.set_enabled(False)

    return image_num

def generate_double_flip(d_cloud, t_cloud, a_cloud, k_cloud, ps_cloud, pt_cloud, base_fn, image_num, mode, num_overlap=9):
    ps_cloud.set_enabled()
    pt_cloud.set_enabled()

    d_cloud.set_enabled()
    t_cloud.set_enabled(False)
    if a_cloud is not None:
        a_cloud.set_enabled(False)
    if k_cloud is not None:
        k_cloud.set_enabled(False)
    image_num = take_screenshots(base_fn, image_num, mode, num_overlap)

    d_cloud.set_enabled(False)
    t_cloud.set_enabled()
    if a_cloud is not None:        
        a_cloud.set_enabled(False)
    if k_cloud is not None:
        k_cloud.set_enabled(False)
    image_num = take_screenshots(base_fn, image_num, mode, num_overlap)

    d_cloud.set_enabled()
    t_cloud.set_enabled(False)
    if a_cloud is not None:
        a_cloud.set_enabled(False)        
    if k_cloud is not None:
        k_cloud.set_enabled(False)
    image_num = take_screenshots(base_fn, image_num, mode, num_overlap)

    if a_cloud is not None:
        d_cloud.set_enabled(False)
        t_cloud.set_enabled(False)
        a_cloud.set_enabled()
        if k_cloud is not None:
            k_cloud.set_enabled(False)
        image_num = take_screenshots(base_fn, image_num, mode, num_overlap)

        d_cloud.set_enabled()
        t_cloud.set_enabled(False)
        a_cloud.set_enabled(False)
        if k_cloud is not None:
            k_cloud.set_enabled(False)
        image_num = take_screenshots(base_fn, image_num, mode, num_overlap)

    if k_cloud is not None:
        d_cloud.set_enabled(False)
        t_cloud.set_enabled(False)
        if a_cloud is not None:
            a_cloud.set_enabled(False)
        k_cloud.set_enabled()
        image_num = take_screenshots(base_fn, image_num, mode, num_overlap)

        d_cloud.set_enabled()
        t_cloud.set_enabled(False)
        if a_cloud is not None:
            a_cloud.set_enabled(False)
        k_cloud.set_enabled(False)
        image_num = take_screenshots(base_fn, image_num, mode, num_overlap)

    d_cloud.set_enabled(False)
    t_cloud.set_enabled()
    if a_cloud is not None:
        a_cloud.set_enabled(False)
    if k_cloud is not None:
        k_cloud.set_enabled(False)
    image_num = take_screenshots(base_fn, image_num, mode, num_overlap)

    d_cloud.set_enabled()
    t_cloud.set_enabled(False)
    if a_cloud is not None:
        a_cloud.set_enabled(False)
    if k_cloud is not None:
        k_cloud.set_enabled(False)
    image_num = take_screenshots(base_fn, image_num, mode, num_overlap)

    if a_cloud is not None:
        d_cloud.set_enabled(False)
        t_cloud.set_enabled(False)
        a_cloud.set_enabled()
        if k_cloud is not None:
            k_cloud.set_enabled(False)
        image_num = take_screenshots(base_fn, image_num, mode, num_overlap)

        t_cloud.set_enabled(False)
        a_cloud.set_enabled(False)
        d_cloud.set_enabled()
        if k_cloud is not None:
            k_cloud.set_enabled(False)

    if k_cloud is not None:
        d_cloud.set_enabled(False)
        t_cloud.set_enabled(False)
        if a_cloud is not None:
            a_cloud.set_enabled(False)
        k_cloud.set_enabled()
        image_num = take_screenshots(base_fn, image_num, mode, num_overlap)

        d_cloud.set_enabled()
        t_cloud.set_enabled(False)
        if a_cloud is not None:
            a_cloud.set_enabled(False)
        k_cloud.set_enabled(False)
        image_num = take_screenshots(base_fn, image_num, mode, num_overlap)

    ps_cloud.set_enabled(False)
    pt_cloud.set_enabled(False)

    return image_num

def create_mp4(output_dir, num_fps, delete_images):
    video_name = output_dir + 'visualization.mp4'

    images = sorted([img for img in os.listdir(output_dir) if img.endswith(".png")])
    frame = cv2.imread(os.path.join(output_dir, images[0]))
    height, width, _ = frame.shape

    fourcc = cv2.VideoWriter_fourcc(*'avc1') # Be sure to use lower case
    video = cv2.VideoWriter(video_name, fourcc, num_fps, (width,height))

    for image in images:
        img = cv2.imread(os.path.join(output_dir, image))
        video.write(img)

    cv2.destroyAllWindows()
    video.release()

    if delete_images:
        for image in images:
            os.remove(os.path.join(output_dir, image))

def rotate_meshes(i, num_images, mesh, change_interval=None):
    if change_interval is not None:
        c_i = num_images // change_interval
        if change_interval == 4:
            if i >= c_i:
                if i >= c_i*2:
                    if i >= c_i*3:
                        h_i = -(c_i - 1)+ (i % c_i)
                    else:
                        h_i = -(i % c_i)
                else:
                    h_i = c_i - 1 - (i % c_i)
            else:
                h_i = i
        else:
            if i >= c_i:
                h_i = c_i-1 - ((c_i-1)//2) 
                if i >= c_i*2:
                    v_i = c_i-1
                    mh_i = -(i % c_i)
                else:
                    v_i = i % c_i
                    mh_i = 0
            else:
                h_i = i - ((c_i-1)//2)
                v_i = 0
                mh_i = 0
        h_theta = 2*np.pi * (h_i/(num_images-1))

        h_rotmat = np.array([[np.cos(h_theta), 0., np.sin(h_theta)], [0., 1., 0.], [-np.sin(h_theta), 0, np.cos(h_theta)]]) 
        rot_mat = h_rotmat

    else:
        theta = 2*np.pi * (i/(num_images-1))
        rot_mat = np.array([[np.cos(theta), 0., np.sin(theta)], [0., 1., 0.], [-np.sin(theta), 0, np.cos(theta)]])

    mesh = np.matmul(mesh, rot_mat)

    return mesh



