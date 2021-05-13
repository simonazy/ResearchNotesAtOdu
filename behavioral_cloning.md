# Behavioral Cloning
This work is inspired by the Nvidia DAVE-2 neural network. Most people do not have the luxury of testing on a real self-driving car, so CARLA could be a good alternative. In principle, to teach a neural network to drive, we need to drive CARLA, recording images of the road and corresponding steering angle that a human driver applied, the process called **behavioral cloning or imitation learning.**

The task can be divided into three steps:
  
- Collecting the dataset
- Designing and training the neural network
- Integrating the neural network in Carla

The following diagram shows the system:

<img src="Dave2_diagram.jpg" alt="Markdown Monster icon" style="float: center; margin-right:20px;">

<span style="color:lightblue">**Getting to know manual_control.py in CARLA API**</span>: read this note.

## Part 1: Recording the dataset
By default, CARLA has five cameras: 
```python
bound_y = 0.5 + self._parent.bounding_box.extent.y
Attachment = carla.AttachmentType
self._camera_transforms = [
    #third-person view
    (carla.Transform(carla.Location(x=-5.5, z=2.5), carla.Rotation(pitch=8.0)), Attachment.SpringArm),
    #From the front of the car,toward the road
    (carla.Transform(carla.Location(x=1.6, z=1.7)), Attachment.Rigid),
    #From the front of the car, toward the car
    (carla.Transform(carla.Location(x=5.5, y=1.5, z=1.5)), Attachment.SpringArm),
    #From far above
    (carla.Transform(carla.Location(x=-8.0, z=6.0), carla.Rotation(pitch=6.0)), Attachment.SpringArm),
    #From the left
    (carla.Transform(carla.Location(x=-1, y=-bound_y, z=0.5)), Attachment.Rigid)]
```
Carla has been written using a very famous engine for video-games: **Unreal Engine 4.** In Unreal Engine, the z axis is the vertical one (up and down), the x axis is for forward and backward, and the y axis is for lateral movements, left and right.

- **Rigid**

    With this fixed attachment the object follow its parent position strictly. This is the recommended attachment to retrieve precise data from the simulation.
- **SpringArm**
    An attachment that expands or retracts the position of the actor, depending on its parent. This attachment is only recommended to record videos from the simulation where a smooth movement is needed. 

So, we could end up adjusting the three cameras to the following locations(under the CameraManager class):
```python
#main,same as the original
(carla.Transform(carla.Location(x=1.6, z = 1.7)), Attachment.Rigid),
#left
(carla.Transform(carla.Location(x=1.6, y=-bound_y, z=1.7)), Attachment.Rigid)
#right
(carla.Transform(carla.Location(x=1.6, y=bound_y, z=1.7)), Attachment.Rigid) #bound_y = 4
```

To add the three cameras, we need to add them in the World.restart(), as follows:
```python
self.camera_manager.add_camera(1)
self.camera_manager.add_camera(2)
```
The **CameraManager.add_camera()** method is defined as follows:
```python
def add_camera(self, camera_index):
    camera_name = self.get_camera_name(camera_index)

    if not (camera_index in self.sensors_added_indexes):
        sensor = self._parent.get_world().spawn_actor(
            self.sensors[self.index][-1],
            self._camera_transforms[camera_index][0],
            attach_to=self._parent,
            attachment_type=self._camera_transforms[camera_index][1]
        self.sensors_added_indexes.add(camera_index)
        self.sensors_added.append(sensor)
        #pass the lambda a weak reference 
        weak_self = weakref.ref(self)
        sensor.listen(lambda image: CameraManager._save_image(weak_self, image, camera_name))
```
What the add_camera() function did is:
- Sets up a sensor, using the specified camera.
- Adds the sensor to a list.
- Instructs the sensor to call a lambda function that invokes the save_image() method.

The following code is to add names to the three cameras:
```python
def get_camera_name(self,index):
    return "MAIN" if index == 0 else ("LEFT" if index == 1 else ("RIGHT" if index == 2 else "UNK"))
```

The camera used by Nvidia was recording at 30 FPS, but they decided to skip most of the frames, recording only at 10 FPS, because the frames were very similar, increasing the training time without adding much information. 

To solve this problem, we will record only one camera view for each frame, then we rotate to the next camera view for the next frame, and we will cycle through all three camera views during recording.

```python
@staticmethod
def _save_image(weak_self,image,camera_name):
    self = weak_self
    if not self:
        return 
    if self.recording:
        #Check the frame
        n = image.frame % 3
        #Save only one camera out of 3, to increase fluidity
        if (n==0 and camera_name == "MAIN") or (n == 1 and camera_name == "LEFT") or (n == 2 and and camera_name == "RIGHT"):
            #Interpret a buffer as a 1-dimensional array.
            img = np.frombuffer(image.raw_data, dtype=np.dtype("unit8"))
            img = np.reshape(img,(img.height, image.width, 4))
            #Drop the last channel: RGB(A)
            img = img[:,:,:3]
            #Resize 
            img = cv2.resize(img, (200, 133))
            #Crop
            img = img[67:, :, :]
            cv2.imwrite('_out/%08d_%s_%f_%f_%f.jpg' % (
                    image.frame, camera_name, self.last_steer, self.last_throttle, self.last_brake), img)
            image.save_to_disk('_out/%08d_%s_%f_%f_%f.jpg' % (
                   image.frame, camera_name, self.last_steer, self.last_throttle, self.last_brake))
```
The following method should be defined to get the steering, brake, and throttle value: 
```python
def set_last_controls(self, control):
    self.last_steer = control.steer
    self.last_throttle = control.throttle
    self.last_brake = control.brake
```

Other places we could change to customize the data collection process:

In the **game_loop()** method:
```python
    client.load_world('Town04')
    client.reload_world()
```
Anchor the spawn point:
```python
spawn_point = spawn_points[0] if spawn_points else carla.Transform()
```
Limit the throttle value:
```python
self._control.throttle = min(self._control.throttle + 0.01, 0.5)
```
Lower the resolution of the server:
```
./CarlaUE4.sh  -ResX=480-ResY=320
```
After we drive the car in CARLA for several loops, we will have the file "_out" in the directory. And the data is name with, for instance: "00003047_RIGHT_0.000000_0.010000_0.000000.png"

<center><img src="00003047_RIGHT_0.000000_0.010000_0.000000.png" alt="Markdown Monster icon" style="width:200px; height:133px"></center>

To prepare the data for train, we need to preprocess the data.
- A convenience to convert -0 to 0.
```python
def to_float(str):
    f = float(str)
    return 0 if f == -0.0 else f
```
- extract the data from the name
```python
def expand_name(file):
    idx = int(max(file.rfind('/'),file.rfind('\\')))
    prefix = file[0:idx]
    file = file[idx:].replace('.png','').replace('.jpg','')
    parts = file.split('_')
    (seq,camera,steer,throttle,brake,img_type) = parts

    return (prefix + seq, camera, to_float(steer), to_float(throttle), to_float(brake), img_type)
```

- change the steering angle and return the new values: The most important thing to do is to correct the steering angle for the left and right camera.
```python
def fix_name(file_name, extension, steering_correction, create_dir):
    (seq, camera, steer, throttle, brake, img_type) = expand_name(file_name)
    #expand the dataset
    if camera == 'LEFT':
        steer = steer + steering_correction
    if camera == 'RIGHT':
        steer = steer - steering_correction
    if img_type == "MIRROR":
        steer = -steer
    
    new_name = seq + "_" + camera + "_" + str(steer) + "_" + str(throttle) + "_" + str(brake) + "_" + img_type + "." + extension

    if create_dir:
        directory = os.path.dirname(new_name)
        if not os.path.exists(directory):
            os.makedirs(directory)
    
    return new_name
```
After processing, we get a file named **"dataset_out"**. 
For instance: 
- "00001289_RIGHT_-0.25_0_0_ORIG.jpg"
- "00001290_MAIN_0_0_0_MIRROR.jpg"

## PART 2: Modeling the neural network

Take a look at the network architecture: 
<center><img src="CNN.png" width="250px" height ="300px"></center>

```python
def Dave2():
    model = Sequential()
    model.add(Lambda(lambda x: x/127.5 - 1., input_shape=(66, 200, 3)))

    model.add(Conv2D(24, (5, 5), strides=(2, 2), activation='elu'))
    model.add(Conv2D(36, (5, 5), strides=(2, 2), activation='relu'))
    model.add(Conv2D(48, (5, 5), strides=(2, 2), activation='relu'))

    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))

    model.add(Flatten())
    model.add(Dense(1164, activation='relu'))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1, activation='tanh'))

    return model
```
Alternative: Pytorch version

```python
class Dave2(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_layers=nn.Sequential(
            nn.Conv2d(1,24,5,stride=2),
            nn.ELU(),
            nn.Conv2d(24,36,5,stride=2),
            nn.ELU(),
            nn.Conv2d(36,48,5, stride=2),
            nn.ELU(),
            nn.Conv2d(48, 64, 3),
            nn.ELU(),
            nn.Conv2d(64, 64, 3),      
        )
        self.dense_layers=nn.Sequential(
            nn.Linear(in_features=1152, out_features=100),
            nn.ELU(),
            nn.Dropout(0.2),
            nn.Linear(in_features=100, out_features=50),
            nn.ELU(),
            nn.Dropout(0.2),
            nn.Linear(in_features=50, out_features=10),
            nn.ELU(),
            nn.Linear(in_features=10, out_features=2)
        )
    def forward(self,data):          
        #data = data.reshape(data.size(0), 1, 60, 120)
        #print(data.shape)
        output = self.conv_layers(data)
        # print(output.shape)
        output = output.view(output.size(0), -1)
        #print(output.shape)
        output = self.dense_layers(output)
        return output
```
Preprocess the images:
- Extract images and labels
```python
def extract_image(file_name):
    return cv2.imread(file_name)

def extract_label(file_name):
    (seq, camera, steer, throttle, brake, img_type) = expand_name(file_name)
    return steer
```
- Generate image-label pairs
```python
def generator(ids, fn_image, fn_label, batch_size=32):
    num_samples = len(ids)
    while 1: # Loop forever so the generator never terminates
        samples_ids = shuffle(ids)  # New epoch

        for offset in range(0, num_samples, batch_size):
            batch_samples_ids = samples_ids[offset:offset + batch_size]
            batch_samples = [fn_image(x) for x in batch_samples_ids]
            batch_labels = [fn_label(x) for x in batch_samples_ids]

            yield np.array(batch_samples), np.array(batch_labels)
```
Train the neural network

Usually, when we train a classifier, as in the case of MNIST or CIFAR-10, we use categorical_crossentropy as a loss and accuracy as a metric. However, for regression, we need to use **mse** for the loss and we can optionally use cosine_proximity as a metric.

```python
model.compile(loss=mse, optimizer=Adam(), metrics=    ['cosine_proximity'])
```
After this process, we get a "*.h5" file as the network parameters. Next, we can test it in CARLA to see performance. 

## Part 3: Integrating the neural network in Carla

In principle, letting our network take control of the steering wheel is quite simple, as we just need to analyze the current frame and set the steering. Meanwhile, to apply some throttle or the car will not move. 

In KeyboardControl.parse_events(), we will intercept the D key and switch the self-driving functionality on and off:
```python
elif event.key == K_d:
    self.self_driving = not self.self_driving
    if self.self_driving:
        world.hud.notification('Self-driving with Neural Network')
    else:
        world.hud.notification('Self-driving OFF')
```
In the CameraManager._parse_image() method, we resize and save the last image received from the server.

```python
array_bgr = cv2.resize(array, (200, 133))
            self.last_image = array_bgr[67:, :, :]
            array = array[:, :, ::-1] #convert from BGR to RGB
```
Next, we can load the "h5" model in the **game_loop()**. 
```python 
while True:
    clock.tick_busy_loop(60)
    if controller.parse_events(client, world, clock):
        return

    if world.camera_manager.last_image is not None:
        image_array = np.asarray(world.camera_manager.last_image)

        controller.self_driving_steer = model.predict(image_array[None, :, :, :], batch_size=1)[0][0].astype(float)

    world.tick(clock)
    world.render(display)
    pygame.display.flip()
```
Now, we have the steering from the network, the last thing is to set a fix throttle value from the KeyBoardControl class under the _parse_vehicle_keys() function.
```python
def _parse_vehicle_keys(self, keys, milliseconds):
    if self.self_driving:
        self.player_max_speed = 0.3
        self.player_max_speed_fast = 0.3
        self._control.throttle = 0.3
        self._control.steer = self.self_driving_steer
        return
```
At this point, we have finished all the steps and ready to see the car in the self-driving mode!

## Summary
DAVE2 is a good start for learning to train a steering model controlled by neural networks. The main idea is about imitation learning or behavioral cloning that to train a self-driving car by imitating another agent's behavior. The general steps for imitation learning are: data collection, training the network, testing the network on simulator.  

References:

Hands-On Vision and Behavior for Self-Driving Cars [<a href='https://www.amazon.com/dp/1800203586'>Copy</a>]

CARLA: An open urban driving simulator[<a href="https://github.com/carla-simulator">repo</a>]