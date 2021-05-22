If you run manual_control.py, the first line you will see is the block of code: 
```python
try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass
```
This will load an egg file containing the code for Carla module. You can also install Carla at local using the following command:
```
python -m easy_install carla-0.9.10-py3.7-linux-x86_64.egg
```

My own preference is to use **conda** to create different virtual environments, so you can switch to different environments based on the code.

To create an environment: 
```
conda create --name myenv
```
To create an environment with a specific version of Python:
```
conda create -n myenv python=3.7
```
Verify that the new environment was installed correctly:
```
conda env list
```
To activate an environment:
```
conda activate myenv
```
For more commands about <a href="https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html"> managing environments</a>, click :grin:!

After this, the code is organized into the following classes:

- **World**: The virtual world where our vehicle moves, including the map and all the actors: vehicles, pedestrians, and sensors(camera, gnss, radar,camera,imu).
- **KeyboardControl**: This part reacts to the keys pressed by users. You can also set it to auto-pilot mode. It has some logic to convert the binary on/off keys for steering, braking, and accelerating to a wider range of values, based on how long they are pressed for.
- **HUD**: This renders all the information related to the simulation process.
- **FadingText**: This class is used by the HUD class to show notifications that disappear after a few seconds.
- **HelpText**: This class displays some text using ```pygame```.
- **CollisionSensor**: As name indicated.
- **LaneInvasionSensor**: As name indicated.
- **GnssSensor**: This is a GPS/GNSS sensor that provides the GNSS position inside the OpenDRIVE map
- **IMUSensor**: This is the *inertial measurement unit*, which uses a gyroscope to detect the accelerations applied to the car.
- **RadarSensor**: A radar, providing a two-dimensional map of the elements detected, including their speed.
- **CameraManager**: This is a class that manages the camera and parses it.

Two important methods:
- **game_loop()**: This mostly initializes ```pygame```, the Carla client, and all the related objects, and it also implements the game loop, where, **60 times per second**, the keys are analyzed and the most updated image is shown on the screen.
- **main()**: This is mostly dedicated to parsing the arguments received by the OS.

Reference:

Hands-On Vision and Behavior for Self-Driving Cars [<a href='https://www.amazon.com/dp/1800203586'>Copy</a>]

CARLA: An open urban driving simulator[<a href="https://github.com/carla-simulator">repo</a>]
