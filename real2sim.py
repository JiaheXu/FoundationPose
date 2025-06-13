# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


from estimater import *
from datareader import *
import argparse
from pathlib import Path

if __name__=='__main__':
  parser = argparse.ArgumentParser()
  code_dir = os.path.dirname(os.path.realpath(__file__))
  # parser.add_argument('--mesh_file', type=str, default=f'{code_dir}/demo_data/mustard0/mesh/textured_simple.obj')
  # parser.add_argument('--test_scene_dir', type=str, default=f'{code_dir}/demo_data/mustard0')
  
  parser.add_argument('--mesh_file', type=str, default= f'{code_dir}/demo_data/mug0/mesh/scaled.obj')
  parser.add_argument('--task_name', type=str, default='mug')
  parser.add_argument('--data_index', type=int, default=1)  

  parser.add_argument('--est_refine_iter', type=int, default=10)
  parser.add_argument('--track_refine_iter', type=int, default=4)
  parser.add_argument('--debug', type=int, default=2)
  parser.add_argument('--debug_dir', type=str, default=f'{code_dir}/debug')
  args = parser.parse_args()

  set_logging_format()
  set_seed(0)

  mesh = trimesh.load(args.mesh_file)

  debug = args.debug
  debug_dir = args.debug_dir
  os.system(f'rm -rf {debug_dir}/* && mkdir -p {debug_dir}/track_vis {debug_dir}/ob_in_cam')

  to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
  bbox = np.stack([-extents/2, extents/2], axis=0).reshape(2,3)

  scorer = ScorePredictor()
  refiner = PoseRefinePredictor()
  glctx = dr.RasterizeCudaContext()
  est = FoundationPose(model_pts=mesh.vertices, model_normals=mesh.vertex_normals, mesh=mesh, scorer=scorer, refiner=refiner, debug_dir=debug_dir, debug=debug, glctx=glctx)
  logging.info("estimator initialization done")

  # test_scene_dir = f'/ws/data/real2sim/{args.task_name}/traj/{args.data_index}.npy'
  # print("processing: ",   test_scene_dir)
  reader = CustomReader( task_name = args.task_name, data_index = args.data_index, shorter_side=None, zfar=np.inf)
  print("!!!!!!!!!!!!!!! len: ", len(reader))
  print("!!!!!!!!!!!!!!! len: ", len(reader))
  print("!!!!!!!!!!!!!!! len: ", len(reader))
  
  video_images = []
  center_poses = []
  traj_pose = []
  keyposes = [190, 226, 266]
  for i in range(len(reader)):
    logging.info(f'i:{i}')
    color = reader.get_color(i)
    depth = reader.get_depth(i)
    if i==0:
      mask = reader.get_mask(0).astype(bool)
      pose = est.register(K=reader.K, rgb=color, depth=depth, ob_mask=mask, iteration=args.est_refine_iter)

      if debug>=3:
        m = mesh.copy()
        m.apply_transform(pose)
        m.export(f'{debug_dir}/model_tf.obj')
        xyz_map = depth2xyzmap(depth, reader.K)
        valid = depth>=0.001
        pcd = toOpen3dCloud(xyz_map[valid], color[valid])
        o3d.io.write_point_cloud(f'{debug_dir}/scene_complete.ply', pcd)
    else:
      pose = est.track_one(rgb=color, depth=depth, K=reader.K, iteration=args.track_refine_iter)
    traj_pose.append( pose.reshape(4,4) )
    os.makedirs(f'{debug_dir}/ob_in_cam', exist_ok=True)
    np.savetxt(f'{debug_dir}/ob_in_cam/{i}.txt', pose.reshape(4,4))

    if debug>=1:
      center_pose = pose@np.linalg.inv(to_origin)
      vis = draw_posed_3d_box(reader.K, img=color, ob_in_cam=center_pose, bbox=bbox)
      vis = draw_xyz_axis(color, ob_in_cam=center_pose, scale=0.1, K=reader.K, thickness=3, transparency=0, is_input_rgb=True)
      
      video_images.append(vis)
      center_poses.append(center_pose)

      cv2.imshow('1', vis)
      cv2.waitKey(1)

    save_data_dir = f'/ws/data/real2sim/{args.task_name}/obj1_traj/'
    OUTPUT_DIR = Path(save_data_dir)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    np.save(f'/ws/data/real2sim/{args.task_name}/obj1_traj/{args.data_index}.npy', traj_pose, allow_pickle=True)

  current_goal_idx = 0
  

  # final_video_images = []

  # for idx in range(len(reader)):
  #   print("idx: ", idx)
  #   print("current_goal_idx: ", current_goal_idx)
  #   if(current_goal_idx >= len(keyposes) ):
  #     final_video_images.append(video_images[idx][...,::-1])
  #     continue
  #   if(idx > keyposes[current_goal_idx]):
  #     current_goal_idx+=1

  #   if(current_goal_idx >= len(keyposes) ):
  #     final_video_images.append(video_images[idx][...,::-1])
  #     continue

  #   current_goal = center_poses[ keyposes[current_goal_idx] ]
  #   # vis = draw_posed_3d_box(reader.K, img=video_images[idx], ob_in_cam=current_goal, bbox=bbox)
  #   vis = draw_xyz_axis(video_images[idx], ob_in_cam=current_goal, scale=0.1, K=reader.K, thickness=3, transparency=0, is_input_rgb=True)
  #   cv2.imshow('1', vis[...,::-1])
  #   cv2.waitKey(1)
  #   final_video_images.append(vis[...,::-1])

  video_name = f'video{args.data_index:02}.avi'
  height, width, layers = video_images[0].shape
  video = cv2.VideoWriter(video_name, 0, 15, (width,height))
  for idx, image in enumerate( video_images ):
    video.write(image)
  video.release()
