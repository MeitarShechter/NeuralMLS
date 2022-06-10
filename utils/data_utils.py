import torch
import numpy as np
import pywavefront

def read_off(file_path):
    with open(file_path, 'r') as file:
        if 'OFF' != file.readline().strip():
            raise('Not a valid OFF header')
        n_verts, n_faces, _ = tuple([int(s) for s in file.readline().strip().split(' ')])
        verts = [[float(s) for s in file.readline().strip().split(' ')] for _ in range(n_verts)]
        faces = [[int(s) for s in file.readline().strip().split(' ')][1:] for _ in range(n_faces)]
    return verts, faces

def read_obj(file_path):
    obj = pywavefront.Wavefront(file_path, collect_faces=True, parse=True)
    verts = obj.vertices
    faces = []
    for mesh_name in obj.meshes:
        part = obj.meshes[mesh_name]
        faces += part.faces

    return verts, faces

def save_obj(filename, points, labels=None):
    assert(points.ndim==2)
    if points.shape[-1] == 2:
        points = np.concatenate([points, np.zeros_like(points)[:, :1]], axis=-1)
    if labels is not None:
        if labels.ndim == 1:
            labels = labels[:, None]
        all_colors = np.array([
            [0, 0, 0],
            [1, 0, 0],
            [0.5, 0, 0],
            [0.6, 0.3, 0],
            [0.5, 0.6, 0],
            [0.7, 0.7, 0], 
            [0.3, 0.6, 0], 
            [0.6, 1, 0.2],
            [0, 0.6, 0.6], 
            [0, 1, 1], 
            [0, 0, 1], 
            [0, 0, 0.4], 
            [0.3, 0.9, 0.1], 
            [0.6, 0, 0.6], 
            [1, 0, 1]
        ])
        colors = np.take_along_axis(all_colors, np.repeat(labels, 3, axis=1), axis=0)

        points = np.concatenate([points, colors], axis=1)
        f = " %f" * points.shape[1]
        np.savetxt(filename, points, fmt="v{}".format(f))
        np.savetxt(filename.replace(".obj", ".seg"), labels)
    else:
        print("GOT NO LABELS TO SAVE")

def get_control_points_for_shape(shape_name, phase='train'):
    control_points = None
    offsets = None
    
    if 'airplane1' in shape_name:
        control_points = torch.tensor([
            [0.335, -0.065,  -0.004],  # nose
            [-0.117, -0.037, -0.347],  # left wing edge
            [-0.128, -0.038,  0.347], # right wind edge 
            [-0.322, -0.022, -0.141],  # tail left
            [-0.338, 0.123,   0.000],   # tail top
            [-0.322, -0.022,  0.141],  # tail right
            [0.061, -0.094,   0.004]   # body 
        ])
        offsets = torch.tensor([
            [0.3, 0, 0],    # nose
            [0.1, 0, -0.1], # left wing edge
            [0.1, 0, 0.1],  # right wind edge 
            [0, 0, 0.2],    # tail left
            [0, 0.2, 0],    # tail top
            [0, 0, -0.2],   # tail right
            [0, 0, 0]      # body 
        ])  

    if 'airplane2' in shape_name:
        control_points = torch.tensor([
            [0.398926, -0.031972, 0.004864],  # nose
            [-0.080740, -0.043164, -0.280274],  # left wing edge
            [-0.084838, -0.043082, 0.280274], # right wind edge 
            [-0.370298, 0.072130, -0.094802],  # tail left
            [-0.314968, -0.020781, 0.005430],   # tail bottom
            [-0.370298, 0.072130, 0.094802],  # tail right
            [0.133974, -0.069945, -0.000134]   # body 
        ])
        offsets = torch.tensor([
            [0.3, 0, 0],    # nose
            [0.1, 0, -0.1], # left wing edge
            [0.1, 0, 0.1],  # right wind edge 
            [0, 0, 0.2],    # tail left
            [0, 0.2, 0],    # tail top
            [0, 0, -0.2],   # tail right
            [0, 0, 0]      # body 
        ])   
                 
    if 'chair1' in shape_name:
        control_points = torch.tensor([
            [0.227311, -0.368148, 0.219803],  # front right leg
            [0.224086, -0.368448, -0.221803], # front left leg
            [-0.194602, -0.368659, 0.206416], # back right leg
            [-0.216803, -0.368659, -0.208891],  # back left leg
            [0.221453, 0.004787, 0.223799],   # front right leg top
            [0.224086, 0.004787, -0.221803],  # front left leg top
            [-0.194602,  0.004787, 0.206416], # back right leg top
            [-0.216803,  0.004787, -0.208891],  # back left leg top
            [-0.211474, 0.368422, 0.206416], #top right backrest
            [-0.191050, 0.368422, -0.208891] #top left backrest
        ])
        offsets = torch.tensor([
            [0, -0.2, 0], # front right leg
            [0, -0.2, 0], # front left leg
            [0, -0.2, 0], # back right leg
            [0, -0.2, 0], # back left leg
            [0, 0, 0], # front right leg top
            [0, 0, 0], # front left leg top
            [0, 0, 0], # back right leg top
            [0, 0, 0], # back left leg top
            [0, 0.2, 0], #top right backrest
            [0, 0.2, 0]  #top left backrest
        ])    
                
    if 'chair2' in shape_name:
        control_points = torch.tensor([
            [0.227311, -0.368148, 0.194050],  # front right leg
            [0.224086, -0.368448, -0.196050], # front left leg
            [-0.238082, -0.396731, 0.148738], # back right leg
            [-0.238082, -0.396731, -0.124322],  # back left leg
            [0.217936, -0.013012, 0.189030],   # front right leg top
            [0.217798, -0.010910, -0.189016],  # front left leg top
            [-0.106441, 0.003103, 0.153791], # back right leg top
            [-0.106441, 0.003103, -0.153792],  # back left leg top
            [-0.182134, 0.373994, 0.145569], #top right backrest
            [-0.194336, 0.392061, -0.144243] #top left backrest
        ])
        offsets = torch.tensor([
            [0, -0.2, 0], # front right leg
            [0, -0.2, 0], # front left leg
            [0, -0.2, 0], # back right leg
            [0, -0.2, 0], # back left leg
            [0, 0, 0], # front right leg top
            [0, 0, 0], # front left leg top
            [0, 0, 0], # back right leg top
            [0, 0, 0], # back left leg top
            [0, 0.2, 0], #top right backrest
            [0, 0.2, 0]  #top left backrest
        ]) 
        
    if 'Male_Low_Poly' in shape_name:
        control_points = torch.tensor([
            [-4.955569,  1.726851,  4.279628],  # right foot
            [-5.124431,  17.020964, 0.654930],  # right knee
            [-5.324431,  24.820964, -2.654930], # right thigh
            [-5.602841,  32.606380, -4.187675], # right hip
            [5.480736,   1.629017,   4.245317], # left foot
            [5.280736,   10.029017,  1.045317], # left shin
            [5.025230,   18.181957, 1.131380],   # left knee
            [5.525230,   25.281957, 2.131380],   # left thigh
            [6.347088,   33.415478, 3.366078],   # left hip
            [0.006791,   45.710350, 3.808482],   # stomach
            [-8.900441,  57.974167, -2.781189],  # right shoulder
            [-10.899905, 46.998695, -1.204356],  # right albow
            [-15.095141, 34.430336, 1.732888],   # right hand
            [8.900959,   57.974045, -2.781122],  # left shoulder
            [11.234556,  46.401623, -1.373343],  # left albow
            [15.237267,  35.420891, 1.551461],   # left hand
            [-0.012136,  71.635742, -1.121795]   # head
        ])
        offsets = torch.tensor([
            [0., 0., 0.], # right foot
            [0., 0., 0.], # right knee
            [0., 0., 0.], # right thigh
            [0., 0., 0.], # right hip
            [0., 0., 0.], # left foot
            [0., 0., 0.], # left shin
            [0., 0., 0.], # left knee
            [0., 0., 0.], # left thigh
            [0., 0., 0.], # left hip
            [0., 0., 0.], # stomach
            [0., 0., 0.], # right shoulder
            [-2.5, 0., 0.], # right albow
            [-5., 0., 0.], # right hand
            [0., 0., 0.], # left shoulder
            [0., 0., 0.], # left albow
            [0., 0., 0.], # left hand
            [0., 0., 0.] # head
        ])          

    if 'ShapeNetPlane1' in shape_name:
        control_points = torch.tensor([                        
            [-0.4202, -0.0600, -0.0123], # rear body
            [-0.0184, -0.1265, -0.2205], # mid left wing
            [ 0.5033, -0.0597,  0.0140], # body
            [-0.0440, -0.1351,  0.2842], # mid right wing
            [-0.2270, -0.1081,  0.6706], # tip right wing
            [-0.8417,  0.1607,  0.0022], # rear wings
            [-0.2079, -0.1060, -0.6440], # tip left wing
            [ 0.9681, -0.0888, -0.0052]  # nose      
        ])
        offsets = torch.tensor([
            [0, 0, 0], # rear body
            [0, 0, 0], # mid left wing
            [0, 0, 0], # body
            [0, 0, 0], # mid right wing
            [-0.3, 0, 0.3], # tip right wing
            [-0.2, 0, 0], # rear wings
            [-0.3, 0, -0.3], # tip left wing
            [0.3, 0, 0]  # nose     
        ])

    if 'ShapeNetPlane2' in shape_name:
        control_points = torch.tensor([                        
            [-0.2301, -0.1012,  0.0190], # rear body
            [-0.4048, -0.0965, -0.3947], # mid left wing
            [ 0.3637, -0.1305, -0.0042], # body
            [-0.4251, -0.0915,  0.3983], # mid right wing
            [-0.8570, -0.0813,  0.5987], # tip right wing
            [-0.7717,  0.0082, -0.0121], # rear wings
            [-0.8309, -0.1073, -0.5838], # tip left wing
            [ 0.9671, -0.1295,  0.0290]  # nose      
        ])
        offsets = torch.tensor([
            [0, 0, 0], # rear body
            [0, 0, 0], # mid left wing
            [0, 0, 0], # body
            [0, 0, 0], # mid right wing
            [0, 0, 0], # tip right wing
            [0, 0, 0], # rear wings
            [0, 0, 0], # tip left wing
            [0, 0, 0]  # nose  
        ])
     
    if 'ShapeNetPlane3' in shape_name:
        control_points = torch.tensor([                        
            [-0.2677, -0.0628,  0.0046], # rear body
            [ 0.0428, -0.0346, -0.4365], # mid left wing
            [ 0.3498, -0.0350,  0.0179], # body
            [ 0.0537, -0.0476,  0.4665], # mid right wing
            [ 0.0748,  0.0012,  0.9809], # tip right wing
            [-0.7053,  0.0585,  0.0266], # rear wings
            [ 0.0304, -0.0069, -0.9920], # tip left wing
            [ 0.7634, -0.0736, -0.0075]  # nose              
        ])
        offsets = torch.tensor([
            [0, 0, 0], # rear body
            [0, 0, 0], # mid left wing
            [0, 0, 0], # body
            [0, 0, 0], # mid right wing
            [0, 0, 0], # tip right wing
            [0, 0, 0], # rear wings
            [0, 0, 0], # tip left wing
            [0, 0, 0]  # nose  
        ])

    if 'ShapeNetChair1' in shape_name:
        control_points = torch.tensor([                        
            [-0.3375, -0.0153,  0.3923], # back right seat
            [-0.5256, -0.8768,  0.4932], # back right leg
            [ 0.4754, -0.8766, -0.4397], # front left leg
            [-0.5193, -0.9160, -0.4505], # back left leg
            [-0.4537,  0.8270,  0.3364], # right backrest
            [-0.3361, -0.1112, -0.3430], # back left seat
            [ 0.3962, -0.0550,  0.0186], # middle seat
            [ 0.3707, -0.0701,  0.4355], # front right seat
            [ 0.4049, -0.0945, -0.4208], # front left seat
            [ 0.4629, -0.8980,  0.4438], # front right leg
            [-0.4633,  0.9375, -0.2891], # middle backrest
            [-0.4530,  0.4096, -0.3266]  # left backrest     
        ])
        offsets = torch.tensor([
            [0, 0, 0], # back right seat
            [0, 0, 0], # back right leg
            [0, 0, 0], # front left leg
            [0, 0, 0], # back left leg
            [0, 0, 0], # right backrest
            [0, 0, 0], # back left seat
            [0, 0, 0], # middle seat
            [0, 0, 0], # front right seat
            [0, 0, 0], # front left seat
            [0, 0, 0], # front right leg
            [0, 0, 0], # middle backrest
            [0, 0, 0]  # left backrest     
        ])

    if 'ShapeNetChair2' in shape_name:
        control_points = torch.tensor([                        
            [-0.3142, -0.0377,  0.3888], # back right seat
            [-0.3517, -0.9024,  0.3800], # back right leg
            [ 0.4670, -0.9080, -0.4221], # front left leg
            [-0.3466, -0.9266, -0.3528], # back left leg
            [-0.4296,  0.8312,  0.3312], # right backrest
            [-0.3302, -0.1383, -0.3444], # back left seat
            [ 0.2810, -0.1747,  0.0061], # middle seat
            [ 0.3966, -0.1512,  0.4257], # front right seat
            [ 0.4146, -0.1740, -0.4270], # front left seat
            [ 0.4379, -0.9278,  0.4436], # front right leg
            [-0.4102,  0.9636, -0.3277], # middle backrest
            [-0.4097,  0.3937, -0.3202]  # left backrest               
        ])
        offsets = torch.tensor([
            [0, 0, 0], # back right seat
            [0, 0, 0], # back right leg
            [0, 0, 0], # front left leg
            [0, 0, 0], # back left leg
            [0, 0, 0], # right backrest
            [0, 0, 0], # back left seat
            [0, 0, 0], # middle seat
            [0, 0, 0], # front right seat
            [0, 0, 0], # front left seat
            [0, 0, 0], # front right leg
            [0, 0, 0], # middle backrest
            [0, 0, 0]  # left backrest     
        ])

    if 'ShapeNetChair3' in shape_name:
        control_points = torch.tensor([                        
            [-0.3481, -0.1014,  0.4927], # back right seat
            [-0.4754, -0.9311,  0.4470], # back right leg
            [ 0.4797, -0.9426, -0.4388], # front left leg
            [-0.4948, -0.9416, -0.4335], # back left leg
            [-0.4429,  0.7741,  0.3777], # right backrest
            [-0.3578, -0.1980, -0.3624], # back left seat
            [ 0.2826, -0.3235,  0.0040], # middle seat
            [ 0.4036, -0.1739,  0.4590], # front right seat
            [ 0.4355, -0.1992, -0.4621], # front left seat
            [ 0.4692, -0.9686,  0.4564], # front right leg
            [-0.4387,  0.9274, -0.3233], # middle backrest
            [-0.4180,  0.3463, -0.4188]  # left backrest              
        ])
        offsets = torch.tensor([
            [0, 0, 0], # back right seat
            [0, 0, 0], # back right leg
            [0, 0, 0], # front left leg
            [0, 0, 0], # back left leg
            [0, 0, 0], # right backrest
            [0, 0, 0], # back left seat
            [0, 0, 0], # middle seat
            [0, 0, 0], # front right seat
            [0, 0, 0], # front left seat
            [0, 0, 0], # front right leg
            [0, 0, 0], # middle backrest
            [0, 0, 0]  # left backrest     
        ])

    if 'ShapeNetCar1' in shape_name:
        control_points = torch.tensor([                        
            [ 0.1930,  0.0705,  0.2810], # middle right
            [-0.3956,  0.2238, -0.1837], # top rear left
            [ 0.1174, -0.0925, -0.3307], # middle left
            [ 0.8237, -0.1391,  0.2739], # front right
            [-0.8779, -0.0576,  0.2733], # rear right
            [-0.8085, -0.1152, -0.3109], # rear left
            [-0.3759, -0.1877,  0.2772], # bottom rear right
            [ 0.8109, -0.1211, -0.2944]  # front left       
        ])
        offsets = torch.tensor([
            [0, 0, 0], # middle right
            [0, 0, 0], # top rear left
            [0, 0, 0], # middle left
            [0, 0, 0], # front right
            [0, 0, 0], # rear right
            [0, 0, 0], # rear left
            [0, 0, 0], # bottom rear right
            [0, 0, 0]  # front left  
        ])

    if 'ShapeNetCar2' in shape_name:
        control_points = torch.tensor([                        
            [ 0.2121,  0.1117,  0.2598], # middle right
            [-0.3865,  0.1987, -0.1924], # top rear left
            [ 0.1771, -0.0738, -0.3023], # middle left
            [ 0.8656, -0.1426,  0.2812], # front right
            [-0.9299, -0.0307,  0.2831], # rear right
            [-0.8548, -0.0858, -0.3049], # rear left
            [-0.4244, -0.1976,  0.3005], # bottom rear right
            [ 0.8599, -0.1186, -0.2835]  # front left                
        ])
        offsets = torch.tensor([
            [0, 0, 0], # middle right
            [0, 0, 0], # top rear left
            [0, 0, 0], # middle left
            [0, 0, 0], # front right
            [0, 0, 0], # rear right
            [0, 0, 0], # rear left
            [0, 0, 0], # bottom rear right
            [0, 0, 0]  # front left  
        ])

    if 'ShapeNetCar3' in shape_name:
        control_points = torch.tensor([                        
            [ 0.2366,  0.0014,  0.2729], # middle right
            [-0.3335,  0.0429, -0.1801], # top rear left
            [ 0.2372, -0.0226, -0.3235], # middle left
            [ 0.8777, -0.1395,  0.2735], # front right
            [-0.9790,  0.1143,  0.2909], # rear right
            [-0.9031,  0.0034, -0.2807], # rear left
            [-0.5062, -0.1612,  0.3459], # bottom rear right
            [ 0.8811, -0.1108, -0.2552]  # front left       
        ])
        offsets = torch.tensor([
            [0, 0, 0], # middle right
            [0, 0, 0], # top rear left
            [0, 0, 0], # middle left
            [0, 0, 0], # front right
            [0, 0, 0], # rear right
            [0, 0, 0], # rear left
            [0, 0, 0], # bottom rear right
            [0, 0, 0]  # front left  
        ])

    if "create_user_study" in phase and \
        ('ShapeNetPlane1' in shape_name or \
         'ShapeNetPlane2' in shape_name or \
         'ShapeNetPlane3' in shape_name):
        offsets = torch.tensor([
            [
                [0, 0, 0], # rear body
                [0, 0, 0], # mid left wing
                [0, 0, 0], # body
                [0, 0, 0], # mid right wing
                [-0.2, 0, 0.2], # tip right wing
                [0, 0, 0], # rear wings
                [-0.2, 0, -0.2], # tip left wing
                [0.4, 0, 0]  # nose    
            ],
            [
                [0, 0, 0], # rear body
                [0.2, 0, 0], # mid left wing
                [-0.05, 0, 0], # body
                [0.2, 0, 0], # mid right wing
                [0.2, 0, 0], # tip right wing
                [0, 0.3, 0], # rear wings
                [0.2, 0, 0], # tip left wing
                [-0.2, 0, 0]  # nose    
            ],
            [
                [0, 0, 0], # rear body
                [0.3, 0, 0], # mid left wing
                [0, 0, 0], # body
                [0.3, 0, 0], # mid right wing
                [0.4, 0, 0], # tip right wing
                [0, 0, 0], # rear wings
                [0.4, 0, 0], # tip left wing
                [0, 0, 0]  # nose    
            ]
        ])
        
    if "create_user_study" in phase and \
        ('ShapeNetCar1' in shape_name or \
         'ShapeNetCar2' in shape_name or \
         'ShapeNetCar3' in shape_name):
         offsets = torch.tensor([
            [
                [0, 0, 0], # middle right
                [0, 0, 0], # top rear left
                [0, 0, 0], # middle left
                [0.3, 0, 0], # front right
                [-0.2, 0, 0], # rear right
                [-0.2, 0, 0], # rear left
                [0, 0, 0], # bottom rear right
                [0.3, 0, 0]  # front left     
            ],
            [
                [0, 0.2, 0], # top middle right
                [0, 0.2, 0], # top rear left
                [0, 0, 0], # middle left
                [0, 0, 0], # front right
                [0, 0, 0], # rear right
                [0, 0, 0], # rear left
                [0, 0, 0], # bottom rear right
                [0, 0, 0]  # front left    
            ],
            [
                [0, 0, 0], # middle right
                [0, 0, -0.2], # top rear left
                [0, 0, -0.2], # middle left
                [0, 0, 0], # front right
                [0, 0, 0], # rear right
                [0, 0, -0.2], # rear left
                [0, 0, 0], # bottom rear right
                [0, 0, -0.2]  # front left  
            ]
        ])

    if "create_user_study" in phase and \
        ('ShapeNetChair1' in shape_name or \
         'ShapeNetChair2' in shape_name or \
         'ShapeNetChair3' in shape_name):
         offsets = torch.tensor([
            [
                [0, 0, 0], # back right seat
                [0, -0.2, 0], # back right leg
                [0.2, -0.2, 0], # front left leg
                [0, -0.2, 0], # back left leg
                [0, 0, 0], # right backrest
                [0, 0, 0], # back left seat
                [-0.2, 0, 0], # middle seat
                [0, 0, 0], # front right seat
                [0, 0, 0], # front left seat
                [0.2, -0.2, 0], # front right leg
                [0, 0, 0], # middle backrest
                [0, 0, 0]  # left backrest   
            ],
            [
                [0, 0, 0], # back right seat
                [0, 0, 0], # back right leg
                [0.5, 0, 0], # front left leg
                [0, 0, 0], # back left leg
                [0, 0, 0], # right backrest
                [0, 0, 0], # back left seat
                [0.25, 0, 0], # middle seat
                [0.5, 0, 0], # front right seat
                [0.5, 0, 0], # front left seat
                [0.5, 0, 0], # front right leg
                [0, 0, 0], # middle backrest
                [0, 0, 0]  # left backrest    
            ],
            [
                [0, 0, 0], # back right seat
                [0, 0.2, 0], # back right leg
                [0, 0.2, 0], # front left leg
                [0, 0.2, 0], # back left leg
                [0, -0.2, 0], # right backrest
                [0, 0, 0], # back left seat
                [0, 0, 0], # middle seat
                [0, 0, 0], # front right seat
                [0, 0, 0], # front left seat
                [0, 0.2, 0], # front right leg
                [0, -0.2, 0], # middle backrest
                [0, 0, 0]  # left backrest  
            ]
        ])

    return control_points, offsets