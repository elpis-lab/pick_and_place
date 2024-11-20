import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm  # For displaying progress
import numpy as np
from tqdm import trange
import argparse
import time
from ur10_env import UR10
from main import HardcodedAgent, evaluate
import torch
from torchvision import datasets, transforms
from model import PerceptionThrowingModel 
import random
from controller import range, velocity
# Training function
def loss_fn(output,R_e,v,p_f):
    g = 9.8 
    R = range(g,p_f[2],v)
    R = R_e - p_f[0]
    v_e = range(g,h,R)
    g_delta = v_e - v
    p_delta = output - v
    Lt = torch.where(torch.abs(g_delta - p_delta) < 1, 
                    1/2 * (g_delta - p_delta)**2, 
                    torch.abs(g_delta - p_delta) - 1/2)
    Lt = grasp_sucess*Lt
    return Lt

def train_cnn(model, RGB_D, v,p_f,R_e, optimizer, device, num_epochs=10):
    model = model.to(device)
    model.train()  # Set model to training mode

    for epoch in range(num_epochs):
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0

        # Iterate over the data loader  # Move the single image to the device
        RGB_D = np.expand_dims(RGB_D, axis=0) # Add batch dimension
        images = torch.tensor(RGB_D).permute(0, 3, 1, 2)
        optimizer.zero_grad()

        # Forward pass
        images = images.float()  # Convert images to float type
        output = model(images, v)
        loss = loss_fn(output, R_e[0],v,p_f)

        # Backward pass and optimization step
        loss.backward()
        optimizer.step()

        # Statistics
        running_loss += loss.item() * images.size(0)  # Accumulate loss
        _, predicted = torch.max(outputs, 1)
        correct_predictions += (predicted == labels).sum().item()
        total_samples += labels.size(0)

        # Calculate average loss and accuracy for the epoch
        epoch_loss = running_loss / total_samples
        epoch_accuracy = correct_predictions / total_samples

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}")

    print("Training complete!")


# Example usage
# def evaluate(policy, env, nepisodes=100, viz=True):
#     success = []
#     reward = []
#     for episode in trange(nepisodes):
#         state,rgb_d = env.reset(episode)
#         policy.reset()
#         reward.append(0)

#         while True:
#             if env.step_id <= 300:
#                 #print(env.step_id)
#                 env.call_camera(episode)
#                 pos,prev_pos = policy.act(state)
#                 #state, rew, done, info = env.step(pos)
#                 env.pick(pos,prev_pos)
#                 rew = 1
#                 done =0
#             else:
#                 current_pos,prev_pos = policy.act(state)
#                 env.call_camera(episode)
#                 state,rew,done,info = env.curl(pos)
#             reward[-1] += rew
#             if viz:
#                 time.sleep(0.01)
#             if done:
#                 success.append(info['is_success'])
#                 print(success[-1])
#                 break
#                                             #Throwing Starts HERE"
#         # Calculate the angular velocity in radians per second
#         throw_duration_seconds = 1  # Duration of throw in seconds
#         max_angle_radians = np.random.uniform(np.pi/2,np.pi) # Maximum angle for the throw (180 degrees)
#         #max_angle_radians = np.pi
#         angular_velocity_rad_per_sec = max_angle_radians / throw_duration_seconds
#         print(angular_velocity_rad_per_sec)
#         # Calculate the number of simulation steps
#         simulation_steps = int(throw_duration_seconds / env.dt)  # Assuming env.dt is the simulation time step
#         # Perform the throwing motion
#         grip_val=10
#         done= 0
#         vel = [0,0,0]
#         gripper_position = [0,0,0]
#         while True:
#             # Convert angular velocity from rad/sec to rad/step
#             env.call_camera(episode)
#             # Move the wrist joint
#             if done==0:
#                 vel,gripper_position, rew, done = env.throw(angular_velocity_rad_per_sec,grip_val)
#             elif done==1:
#                 vel, p_f, Range = env.step_sim(vel, gripper_position,angular_velocity_rad_per_sec,done)
#             else:
#                 time.sleep(2)
#                 break
#             if viz:
#                 time.sleep(0)
#         env.close()
#     print(reward)
#     return rgb_d, vel,p_f,Range

# class DepthRGBDataset(Dataset):
#         def __init__(self, root_dir, transform=None):
#             self.root_dir = root_dir
#             self.transform = transform
#             self.depth_files = [f for f in os.listdir(root_dir) if f.endswith('.jpg') and 'depth' in f]
#             self.rgb_files = [f for f in os.listdir(root_dir) if f.endswith('.jpg') and 'rgb' in f]

#         def __len__(self):
#             return len(self.depth_files)

#         def __getitem__(self, index):
#             depth_file = self.depth_files[index]
#             rgb_file = self.rgb_files[index]
#             depth_path = os.path.join(self.root_dir, depth_file)
#             rgb_path = os.path.join(self.root_dir, rgb_file)

#             depth_img = Image.open(depth_path).convert('L')  # Grayscale depth
#             rgb_img = Image.open(rgb_path).convert('RGB')

#             if self.transform:
#                 depth_img = self.transform(depth_img)
#                 rgb_img = self.transform(rgb_img)

#             # Concatenate depth and RGB images
#             combined_img = torch.cat((rgb_img, depth_img), dim=0)

#             return combined_img

if __name__ == "__main__":
    env = UR10(is_train='viz'=='eval', is_dense=True)
    nepisodes =100
    agent = HardcodedAgent(UR10.position_bounds, UR10.orientation_bounds)
    model = PerceptionThrowingModel()
    optimizer =  optim.Adam(model.parameters(), lr=1e-5)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize images to 224x224
        transforms.ToTensor(),  # Convert PIL images to tensors
        transforms.Normalize(mean=[0.485, 0.456, 0.406, 0.5], std=[0.229, 0.224, 0.225, 0.5])  # Normalize the images
    ])
    for episode in trange(nepisodes):
        state,rgb_d = env.reset(episode)
        objects =3
        fruits = ['strawberry','peach','lemon']
        while objects>0:
            success = []
            selected_fruit = random.choice(fruits)
            fruits.remove(selected_fruit)
            print(selected_fruit)
            if selected_fruit == 'strawberry':
                env.object = env.strawberry
            elif selected_fruit == 'peach':
                env.object = env.peach
            elif selected_fruit == 'lemon':
                env.object = env.lemon
            env.step_id =0
            vel,p_f,R_e = evaluate(agent, env,episode,objects, viz='eval')
            vel = 1.53
            p_f = [0,0,1.5]
            R_e = [0.5,0.5]
              # Example with 10 classes
         # Loss function for classification
 # SGD or ADAM(prefered) Optimizer
    
    # Define the data transformation
    
    # Load the dataset
            # dataset = DepthRGBDataset(root_dir=r"C:/Users/mark1/Downloads/pick_and_place/RGB", transform=transform)
            # train_loader = DataLoader(
            #     dataset=dataset,  # Replace with actual dataset
            #     batch_size=1,
            #     shuffle=True
            # )
    # Device configuration
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Train the model
            train_cnn(model, rgb_d,vel,p_f,R_e, optimizer, device, num_epochs=10)
