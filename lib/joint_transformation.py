import numpy as np 
import yaml

def robot_config_loader(robot_config_path, robot_name = 'UR10'):
    robot_config = yaml.load(open(robot_config_path), Loader=yaml.FullLoader)[robot_name]
    joint_angles_dict = robot_config['theta']
    a_dict = robot_config['a']
    d_dict = robot_config['d']
    alpha_dict = robot_config['alpha']
    mass_dict = robot_config['mass']
    com_dict = robot_config['com']
    
    def convert_values(config_dict):
        for key in config_dict.keys():
            if config_dict[key] == 'pi':
                config_dict[key] = np.pi
            elif config_dict[key] == 'pi/2':
                config_dict[key] = np.pi/2
            elif config_dict[key] == '-pi/2':
                config_dict[key] = -np.pi/2
            elif config_dict[key] == '-pi':
                config_dict[key] = -np.pi
            else:
                config_dict[key] = float(config_dict[key])
        return config_dict
    
    # If there is a pi or pi/2 in any configuration - value is converted to np.pi or np.pi/2
    joint_angles_dict = convert_values(joint_angles_dict)
    alpha_dict = convert_values(alpha_dict)
    a_dict = convert_values(a_dict)
    d_dict = convert_values(d_dict)
    mass_dict = convert_values(mass_dict)
    #com_dict = convert_values(com_dict)
    
    return joint_angles_dict, a_dict, d_dict, alpha_dict, mass_dict, com_dict

def find_DH_transform(joint_angles_dict, a_dict, d_dict, alpha_dict):
    
    T_01 = np.array([
        [np.cos(joint_angles_dict['j1']) , -np.sin(joint_angles_dict['j1'])* np.cos(alpha_dict['j1']), np.sin(joint_angles_dict['j1'])* np.sin(alpha_dict['j1']), a_dict['j1']*np.cos(joint_angles_dict['j1'])],
        [np.sin(joint_angles_dict['j1']) , np.cos(joint_angles_dict['j1'])* np.cos(alpha_dict['j1']), -np.cos(joint_angles_dict['j1'])* np.sin(alpha_dict['j1']), a_dict['j1']*np.sin(joint_angles_dict['j1'])],
        [0, np.sin(alpha_dict['j1']), np.cos(alpha_dict['j1']), d_dict['j1']],
        [0, 0, 0, 1]
    ])
    
    T_12 = np.array([
        [np.cos(joint_angles_dict['j2']) , -np.sin(joint_angles_dict['j2'])* np.cos(alpha_dict['j2']), np.sin(joint_angles_dict['j2'])* np.sin(alpha_dict['j2']), a_dict['j2']*np.cos(joint_angles_dict['j2'])],
        [np.sin(joint_angles_dict['j2']) , np.cos(joint_angles_dict['j2'])* np.cos(alpha_dict['j2']), -np.cos(joint_angles_dict['j2'])* np.sin(alpha_dict['j2']), a_dict['j2']*np.sin(joint_angles_dict['j2'])],
        [0, np.sin(alpha_dict['j2']), np.cos(alpha_dict['j2']), d_dict['j2']],
        [0, 0, 0, 1]
    ])
    
    T_23 = np.array([
        [np.cos(joint_angles_dict['j3']) , -np.sin(joint_angles_dict['j3'])* np.cos(alpha_dict['j3']), np.sin(joint_angles_dict['j3'])* np.sin(alpha_dict['j3']), a_dict['j3']*np.cos(joint_angles_dict['j3'])],
        [np.sin(joint_angles_dict['j3']) , np.cos(joint_angles_dict['j3'])* np.cos(alpha_dict['j3']), -np.cos(joint_angles_dict['j3'])* np.sin(alpha_dict['j3']), a_dict['j3']*np.sin(joint_angles_dict['j3'])],
        [0, np.sin(alpha_dict['j3']), np.cos(alpha_dict['j3']), d_dict['j3']],
        [0, 0, 0, 1]
    ])
    
    T_34 = np.array([
        [np.cos(joint_angles_dict['j4']) , -np.sin(joint_angles_dict['j4'])* np.cos(alpha_dict['j4']), np.sin(joint_angles_dict['j4'])* np.sin(alpha_dict['j4']), a_dict['j4']*np.cos(joint_angles_dict['j4'])],
        [np.sin(joint_angles_dict['j4']) , np.cos(joint_angles_dict['j4'])* np.cos(alpha_dict['j4']), -np.cos(joint_angles_dict['j4'])* np.sin(alpha_dict['j4']), a_dict['j4']*np.sin(joint_angles_dict['j4'])],
        [0, np.sin(alpha_dict['j4']), np.cos(alpha_dict['j4']), d_dict['j4']],
        [0, 0, 0, 1]
    ])
    
    T_45 = np.array([
        [np.cos(joint_angles_dict['j5']) , -np.sin(joint_angles_dict['j5'])* np.cos(alpha_dict['j5']), np.sin(joint_angles_dict['j5'])* np.sin(alpha_dict['j5']), a_dict['j5']*np.cos(joint_angles_dict['j5'])],
        [np.sin(joint_angles_dict['j5']) , np.cos(joint_angles_dict['j5'])* np.cos(alpha_dict['j5']), -np.cos(joint_angles_dict['j5'])* np.sin(alpha_dict['j5']), a_dict['j5']*np.sin(joint_angles_dict['j5'])],
        [0, np.sin(alpha_dict['j5']), np.cos(alpha_dict['j5']), d_dict['j5']],
        [0, 0, 0, 1]
    ])
    
    T_56 = np.array([
        [np.cos(joint_angles_dict['j6']) , -np.sin(joint_angles_dict['j6'])* np.cos(alpha_dict['j6']), np.sin(joint_angles_dict['j6'])* np.sin(alpha_dict['j6']), a_dict['j6']*np.cos(joint_angles_dict['j6'])],
        [np.sin(joint_angles_dict['j6']) , np.cos(joint_angles_dict['j6'])* np.cos(alpha_dict['j6']), -np.cos(joint_angles_dict['j6'])* np.sin(alpha_dict['j6']), a_dict['j6']*np.sin(joint_angles_dict['j6'])],
        [0, np.sin(alpha_dict['j6']), np.cos(alpha_dict['j6']), d_dict['j6']],
        [0, 0, 0, 1]
    ])
    
    end_effector_transform = (((((T_01@T_12)@T_23)@T_34)@T_45)@T_56)
    return end_effector_transform

def test():
    robot_config_path = "config/robot.yaml"
    robot_name = 'UR10'
    joint_angles_dict, a_dict, d_dict, alpha_dict, mass_dict, com_dict = robot_config_loader(robot_config_path, robot_name)
    end_effector_transform = find_DH_transform(joint_angles_dict, a_dict, d_dict, alpha_dict)
    print("End effector transformation matrix:")
    print(end_effector_transform)
    
if __name__ == "__main__":
    test()