import os
import argparse
import torch
import numpy as np
import polyscope as ps

from utils.data_utils import read_obj
from experiments.visualizations.visualization_utils import take_screenshots, read_files, parse_and_normalize, register_pointclouds, register_meshes, init_camera, \
    generate_first_frames, generate_double_flip, create_mp4, rotate_meshes

    
def create_visualization(files_dir, step=150, shape="101", cam_radius=4, num_images=30, num_fps=10, delete_images=True, is_test=False, with_faces=False):

    output_dir = files_dir + 'screenshots/'
    os.makedirs(output_dir, exist_ok=True)
    init_cam_pos = np.array([0., 0.8, 1.])
    init_cam_trg = np.array([0., 0., 0.])

    source, deformed, target, arap, kpd, source_cp, target_cp, _, s_face, d_face, t_face, a_face, k_face = read_files(files_dir, step, shape, is_test, with_faces)    

    s_color, source, d_color, deformed, t_color, target, a_color, arap, k_color, kpd, \
    scp_color, source_cp, tcp_color, target_cp = parse_and_normalize(source, deformed, target, arap, kpd, source_cp, target_cp, with_faces)
        
    o_color = (0.3, 0.3, 0.3)     
    offsets = target_cp - source_cp

    ps.init()
    ps.set_ground_plane_mode("none")

    if with_faces:
        s_cloud, d_cloud, t_cloud, a_cloud, k_cloud, ps_cloud, pt_cloud = register_meshes(\
            s_color, source, d_color, deformed, t_color, target, a_color, arap, k_color, kpd, scp_color, source_cp, tcp_color, target_cp, s_face, d_face, t_face, a_face, k_face, source_radius=0.002)
    else:
        s_cloud, d_cloud, t_cloud, ps_cloud, pt_cloud = register_pointclouds(\
            s_color, source, d_color, deformed, t_color, target, scp_color, source_cp, tcp_color, target_cp, source_radius=0.002)

    image_num = 0
    base_fn = output_dir + "image{:04}.png"

    for i in range(num_images):
        # Position the camera
        init_camera(i, num_images, init_cam_pos, init_cam_trg, cam_radius, 'vertical')

        # first view show the movement of CP
        if i == 0:
            image_num = generate_first_frames(s_cloud, d_cloud, ps_cloud, pt_cloud, source_cp, offsets, o_color, base_fn, image_num, num_fps)

        # every a while show both results
        if i % (num_images//5) == 0:        
            image_num = generate_double_flip(d_cloud, t_cloud, a_cloud, k_cloud, ps_cloud, pt_cloud, base_fn, image_num, num_fps)

        # Take a screenshot
        image_num = take_screenshots(base_fn, image_num, 1)

    # create video
    create_mp4(output_dir, num_fps, delete_images)

def create_basic_visualization(opt, files_dir, shape="101", cam_radius=4, num_images=30, num_fps=12, delete_images=True):
    
    init_cam_trg = np.array([0., 0., 0.])
    source_radius = 0.002

    output_dir = files_dir + 'screenshots/'
    os.makedirs(output_dir, exist_ok=True)

    ps.init()
    ps.set_ground_plane_mode("shadow_only")

    image_num = 0
    base_fn = output_dir + "image{:04}.png"
    num_changes = len(opt.cp_order)
    for curr_idx in range(num_changes):
        # read, parse and normalize files
        if curr_idx == 0:   
            # source         
            source_path     = files_dir + '{}-Sa.obj'.format(shape)
            source, s_face = read_obj(source_path)
            source, s_face = np.array(source), np.array(s_face)
            s_color = source[:, 3:].astype('float32')
            source = source[:, :3]
            s_max = np.max(source, axis=0)
            s_min = np.min(source, axis=0)
            s_mid = (s_max + s_min)/2
            source = source - s_mid
            s_offset = [0, 0, (np.max(source, axis=0) - np.min(source, axis=0))[2] * 1.1]
            # source CP
            source_cp_path = files_dir + 'source_points.obj'
            source_cp, _ = read_obj(source_cp_path)    
            source_cp = np.array(source_cp)   
            scp_color = source_cp[:, 3:].astype('float32')
            source_cp = source_cp[:, :3]    
            source_cp = source_cp - s_mid
            updated_source_cp = np.copy(source_cp)
        # ours   
        deformed_path   = files_dir + '{}-Sab_{}.obj'.format(shape, curr_idx)
        deformed, d_face = read_obj(deformed_path)    
        deformed, d_face = np.array(deformed), np.array(d_face)   
        d_color = deformed[:, 3:].astype('float32')
        deformed = deformed[:, :3]    
        c_deformed = deformed - s_mid
        deformed = c_deformed + s_offset
        # classic
        target_path     = files_dir + '{}-Sab_CLASSIC_{}.obj'.format(shape, curr_idx)
        target, t_face = read_obj(target_path)    
        target, t_face = np.array(target), np.array(t_face)   
        t_color = target[:, 3:].astype('float32')
        target = target[:, :3]    
        c_target = target - s_mid
        target =  c_target - s_offset
        # target cp
        target_cp_path  = files_dir + 'target_points_{}.obj'.format(curr_idx)
        target_cp, _ = read_obj(target_cp_path)    
        target_cp = np.array(target_cp)   
        tcp_color = target_cp[:, 3:].astype('float32')
        target_cp = target_cp[:, :3]
        target_cp = target_cp - s_mid

        # register meshes
        if curr_idx == 0:
            # source
            s_cloud = ps.register_surface_mesh("source shape", source, s_face, enabled=False)
            s_cloud.add_color_quantity("source shape colors", s_color, enabled=True)
            # source CP
            ps_cloud = ps.register_point_cloud("source cp", source_cp, radius=5*source_radius, enabled=False)
            ps_cloud.add_color_quantity("source cp colors", scp_color, enabled=True)
            # ours
            d_cloud = ps.register_surface_mesh("deformed shape", deformed, d_face, enabled=False)
            d_cloud.add_color_quantity("deformed shape colors", d_color, enabled=True)
            # classic
            t_cloud = ps.register_surface_mesh("target shape", target, t_face, enabled=False)
            t_cloud.add_color_quantity("target shape colors", t_color, enabled=True)
            # target CP
            pt_cloud = ps.register_point_cloud("target cp", target_cp, radius=5*source_radius, enabled=False)
            pt_cloud.add_color_quantity("target cp colors", tcp_color, enabled=True)
        
        o_color = (0.3, 0.3, 0.3)     
        offsets = target_cp - source_cp

        init_camera(0, 2, opt.camera_pos[curr_idx], init_cam_trg, cam_radius, 'vertical')

        # 1st rotation - around vertical axis
        for i in range(num_images):
            # Rotate the meshes            
            source_n = rotate_meshes(i, num_images, source)
            source_cp_n = rotate_meshes(i, num_images, source_cp)
            target_cp_n = rotate_meshes(i, num_images, target_cp)
            deformed_n = rotate_meshes(i, num_images, c_deformed) + s_offset
            target_n = rotate_meshes(i, num_images, c_target) - s_offset

            # update meshes
            # source
            s_cloud = ps.register_surface_mesh("source shape", source_n, s_face, enabled=False)
            s_cloud.add_color_quantity("source shape colors", s_color, enabled=True)
            # source CP
            ps_cloud = ps.register_point_cloud("source cp", source_cp_n, radius=5*source_radius, enabled=False)
            ps_cloud.add_color_quantity("source cp colors", scp_color, enabled=True)
            # ours
            d_cloud = ps.register_surface_mesh("deformed shape", deformed_n, d_face, enabled=False)
            d_cloud.add_color_quantity("deformed shape colors", d_color, enabled=True)
            # classic
            t_cloud = ps.register_surface_mesh("target shape", target_n, t_face, enabled=False)
            t_cloud.add_color_quantity("target shape colors", t_color, enabled=True)
            # target CP
            pt_cloud = ps.register_point_cloud("target cp", target_cp_n, radius=5*source_radius, enabled=False)
            pt_cloud.add_color_quantity("target cp colors", tcp_color, enabled=True) 

            # first view show the movement of CP
            if i == 0:
                if curr_idx == 0:
                    s_cloud.set_enabled()
                else:
                    s_cloud.set_enabled()
                    d_cloud.set_enabled()
                    t_cloud.set_enabled()
                ps_cloud.set_enabled()

                curr_offsets = np.zeros_like(offsets)
                curr_offsets[opt.cp_order[curr_idx]] = offsets[opt.cp_order[curr_idx]]
                ps_cloud.add_vector_quantity("offsets", np.zeros_like(offsets), vectortype='ambient', color=o_color, radius=0.003, enabled=True)
                # CP that are moved visualization
                movedCP_cloud = ps.register_point_cloud("moved cp", source_cp[opt.cp_order[curr_idx]], radius=15*source_radius, enabled=True)
                movedCP_cloud.add_color_quantity("moved cp colors", torch.zeros_like(torch.tensor(scp_color))[opt.cp_order[curr_idx]], enabled=True)
                for j in range(num_fps):
                    ps_cloud.update_point_positions(updated_source_cp + curr_offsets * (j/(num_fps-1)))
                    movedCP_cloud.update_point_positions((updated_source_cp + curr_offsets * (j/(num_fps-1)))[opt.cp_order[curr_idx]])
                    if curr_idx == 0:
                        image_num = take_screenshots(base_fn, image_num, 'compare_mls', 1)
                    else:
                        image_num = take_screenshots(base_fn, image_num, 'compare_mls', 1, addLegend=True)
                        # image_num = take_screenshots(base_fn, image_num, 'compare_mls', 1)
                movedCP_cloud.set_enabled(False)

                # target CP
                pt_cloud = ps.register_point_cloud("target cp", target_cp_n, radius=5*source_radius, enabled=False)
                pt_cloud.add_color_quantity("target cp colors", tcp_color, enabled=True) 

                ps_cloud.update_point_positions(updated_source_cp)
                pt_cloud.set_enabled()
                # offsets
                ps_cloud.add_vector_quantity("offsets", curr_offsets, vectortype='ambient', color=o_color, radius=0.003, enabled=True)
                # image_num = take_screenshots(base_fn, image_num, 'compare_mls', 2*num_fps)
                if curr_idx == 0:
                    image_num = take_screenshots(base_fn, image_num, 'compare_mls', int(0.5*num_fps))
                else:
                   image_num = take_screenshots(base_fn, image_num, 'compare_mls', int(0.5*num_fps), addLegend=True)
                # image_num = take_screenshots(base_fn, image_num, 'compare_mls', int(0.5*num_fps))
                # update source cp
                updated_source_cp[opt.cp_order[curr_idx]] = target_cp[opt.cp_order[curr_idx]]
                ps_cloud.update_point_positions(updated_source_cp)
                ps_cloud.add_vector_quantity("offsets", np.zeros_like(offsets), vectortype='ambient', color=o_color, radius=0.003, enabled=True)

                # ours
                d_cloud = ps.register_surface_mesh("deformed shape", deformed_n, d_face, enabled=False)
                d_cloud.add_color_quantity("deformed shape colors", d_color, enabled=True)
                # classic
                t_cloud = ps.register_surface_mesh("target shape", target_n, t_face, enabled=False)
                t_cloud.add_color_quantity("target shape colors", t_color, enabled=True)

                d_cloud.set_enabled()
                t_cloud.set_enabled()
                image_num = take_screenshots(base_fn, image_num, 'compare_mls', num_fps, addLegend=True)
                # image_num = take_screenshots(base_fn, image_num, 'compare_mls', 2*num_fps)

            # # every a while show both results
            # if i % (num_images//2) == 0:
            #     image_num = generate_double_flip(d_cloud, t_cloud, None, None, ps_cloud, pt_cloud, base_fn, image_num, 'compare_mls', num_fps)
            #     d_cloud.set_enabled(False)

            # update meshes
            # source
            s_cloud = ps.register_surface_mesh("source shape", source_n, s_face, enabled=False)
            s_cloud.add_color_quantity("source shape colors", s_color, enabled=True)
            # source CP
            ps_cloud = ps.register_point_cloud("source cp", source_cp_n, radius=5*source_radius, enabled=False)
            ps_cloud.add_color_quantity("source cp colors", scp_color, enabled=True)
            # ours
            d_cloud = ps.register_surface_mesh("deformed shape", deformed_n, d_face, enabled=False)
            d_cloud.add_color_quantity("deformed shape colors", d_color, enabled=True)
            # classic
            t_cloud = ps.register_surface_mesh("target shape", target_n, t_face, enabled=False)
            t_cloud.add_color_quantity("target shape colors", t_color, enabled=True)
            # target CP
            pt_cloud = ps.register_point_cloud("target cp", target_cp_n, radius=5*source_radius, enabled=False)
            pt_cloud.add_color_quantity("target cp colors", tcp_color, enabled=True) 

            # Take a screenshot
            s_cloud.set_enabled()
            ps_cloud.set_enabled()
            d_cloud.set_enabled()
            t_cloud.set_enabled()
            pt_cloud.set_enabled()
            image_num = take_screenshots(base_fn, image_num, 'compare_mls', 1, addLegend=True)
            # image_num = take_screenshots(base_fn, image_num, 'compare_mls', 1)

            # d_cloud.set_enabled(False)

    # create video
    create_mp4(output_dir, num_fps, delete_images)

def create_temp_visualization(opt, files_dir, shape="101", cam_radius=4, num_fps=10, delete_images=True, is_mls_ablation=False):
    
    init_cam_trg = np.array([0., 0., 0.])
    source_radius = 0.002

    output_dir = files_dir + 'screenshots/'
    os.makedirs(output_dir, exist_ok=True)

    ps.init()
    ps.set_ground_plane_mode("none")

    image_num = 0
    base_fn = output_dir + "image{:04}.png"
    num_changes = len(opt.cp_order)
    if is_mls_ablation:
        num_temps = 1
    else:
        num_temps = len(opt.temps)
        
    for curr_idx in range(num_changes):
        for T_idx in range(num_temps):
            if is_mls_ablation:
                T = opt.MLS_alpha
            else:
                T = opt.temps[T_idx]
            # read, parse and normalize files
            if curr_idx == 0 and T_idx == 0:   
                # source         
                source_path     = files_dir + '{}-Sa.obj'.format(shape)
                source, s_face = read_obj(source_path)
                source, s_face = np.array(source), np.array(s_face)
                s_color = source[:, 3:].astype('float32')
                source = source[:, :3]
                s_max = np.max(source, axis=0)
                s_min = np.min(source, axis=0)
                s_mid = (s_max + s_min)/2
                source = source - s_mid
                # source CP
                source_cp_path = files_dir + 'source_points.obj'
                source_cp, _ = read_obj(source_cp_path)    
                source_cp = np.array(source_cp)   
                scp_color = source_cp[:, 3:].astype('float32')
                source_cp = source_cp[:, :3]    
                source_cp = source_cp - s_mid
                updated_source_cp = np.copy(source_cp)
            if is_mls_ablation:
                # classic
                deformed_path     = files_dir + '{}-Sab_CLASSIC_{}.obj'.format(shape, curr_idx)
                deformed, d_face = read_obj(deformed_path)    
                deformed, d_face = np.array(deformed), np.array(d_face)   
                d_color = deformed[:, 3:].astype('float32')
                deformed = deformed[:, :3]    
                deformed = deformed - s_mid
            else:
                # ours   
                deformed_path   = files_dir + '{}-Sab_{}_T_{:.3f}.obj'.format(shape, curr_idx, T)
                deformed, d_face = read_obj(deformed_path)    
                deformed, d_face = np.array(deformed), np.array(d_face)   
                d_color = deformed[:, 3:].astype('float32')
                deformed = deformed[:, :3]    
                deformed = deformed - s_mid            
            # target cp
            target_cp_path  = files_dir + 'target_points_{}.obj'.format(curr_idx)
            target_cp, _ = read_obj(target_cp_path)    
            target_cp = np.array(target_cp)   
            tcp_color = target_cp[:, 3:].astype('float32')
            target_cp = target_cp[:, :3]
            target_cp = target_cp - s_mid

            # register meshes
            if curr_idx == 0 and T_idx == 0:
                # source
                s_cloud = ps.register_surface_mesh("source shape", source, s_face, enabled=False)
                s_cloud.add_color_quantity("source shape colors", s_color, enabled=True)
                # source CP
                ps_cloud = ps.register_point_cloud("source cp", source_cp, radius=5*source_radius, enabled=False)
                ps_cloud.add_color_quantity("source cp colors", scp_color, enabled=True)
                # ours
                d_cloud = ps.register_surface_mesh("deformed shape", deformed, d_face, enabled=False)
                d_cloud.add_color_quantity("deformed shape colors", d_color, enabled=True)
                # # classic
                # t_cloud = ps.register_surface_mesh("target shape", target, t_face, enabled=False)
                # t_cloud.add_color_quantity("target shape colors", t_color, enabled=True)
                # target CP
                pt_cloud = ps.register_point_cloud("target cp", target_cp, radius=5*source_radius, enabled=False)
                pt_cloud.add_color_quantity("target cp colors", tcp_color, enabled=True)
            
            o_color = (0.3, 0.3, 0.3)     
            offsets = target_cp - source_cp

            # Position the camera
            init_camera(0, 2, opt.camera_pos[curr_idx], init_cam_trg, cam_radius, 'vertical')

            # first view show the movement of CP
            if curr_idx == 0 and T_idx == 0:
                s_cloud.set_enabled()
            else:
                d_cloud.set_enabled()
            ps_cloud.set_enabled()

            curr_offsets = np.zeros_like(offsets)
            curr_offsets[opt.cp_order[curr_idx]] = offsets[opt.cp_order[curr_idx]]
            ps_cloud.add_vector_quantity("offsets", np.zeros_like(offsets), vectortype='ambient', color=o_color, radius=0.003, enabled=True)
            if curr_idx == 0 and T_idx == 0:
                for j in range(3*num_fps):
                    ps_cloud.update_point_positions(updated_source_cp + curr_offsets * (j/(3*num_fps-1)))
                    image_num = take_screenshots(base_fn, image_num, 1, addLegend=False)

            # update meshes
            # ours
            d_cloud = ps.register_surface_mesh("deformed shape", deformed, d_face, enabled=False)
            d_cloud.add_color_quantity("deformed shape colors", d_color, enabled=True)
            # # classic
            # t_cloud = ps.register_surface_mesh("target shape", target, t_face, enabled=False)
            # t_cloud.add_color_quantity("target shape colors", t_color, enabled=True)
            # target CP
            pt_cloud = ps.register_point_cloud("target cp", target_cp, radius=5*source_radius, enabled=False)
            pt_cloud.add_color_quantity("target cp colors", tcp_color, enabled=True) 

            ps_cloud.update_point_positions(updated_source_cp)
            pt_cloud.set_enabled()
            # offsets
            ps_cloud.add_vector_quantity("offsets", curr_offsets, vectortype='ambient', color=o_color, radius=0.003, enabled=True)
            # if curr_idx > 0: 
            d_cloud.set_enabled()
            s_cloud.set_enabled(False)

            image_num = take_screenshots(base_fn, image_num, 'temperature', num_fps, addLegend=(not is_mls_ablation), T=T)

            # if curr_idx == 0:
            ps_cloud.set_enabled(False)
            pt_cloud.set_enabled(False)
            # d_cloud.set_enabled()
            # image_num = take_screenshots(base_fn, image_num, num_fps)
            d_cloud.set_enabled(False)

            # # Take a screenshot
            # d_cloud.set_enabled()
            # image_num = take_screenshots(base_fn, image_num, 1)
            # d_cloud.set_enabled(False)

        # update source cp
        updated_source_cp[opt.cp_order[curr_idx]] = target_cp[opt.cp_order[curr_idx]]
        ps_cloud.update_point_positions(updated_source_cp)
        ps_cloud.add_vector_quantity("offsets", np.zeros_like(offsets), vectortype='ambient', color=o_color, radius=0.003, enabled=True)

    # create video
    create_mp4(output_dir, num_fps, delete_images)

        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, description='visualization parser')

    parser.add_argument("--files_dir", type=str, help="path to all reesult files", required=True)
    parser.add_argument("--step", type=int, help="from which step to generate the visualization", default=150)
    parser.add_argument("--shape", type=str, help="name of shape", default="101")
    parser.add_argument("--cam_radius", type=float, help="radius for the camera (distance from object)", default=4)
    parser.add_argument("--num_images", type=int, help="how many images per 360deg rotation to capture", default=32)
    parser.add_argument("--num_fps", type=int, help="number of fps for the video", default=10)
    parser.add_argument("--delete_images", action="store_true", help="whether to delete the images after the creation of the video")
    parser.add_argument("--is_test", action="store_true", help="whether the visualization is for test results (different naming)")
    parser.add_argument("--with_faces", action="store_true", help="whether the visualization is for meshes")

    opt = parser.parse_args()

    create_visualization(opt.files_dir, opt.step, opt.shape, opt.cam_radius, opt.num_images, opt.num_fps, opt.delete_images, opt.is_test, opt.with_faces)