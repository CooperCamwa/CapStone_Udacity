# Capstone Project

### Installation & Deployment

#### Docker:

clone our repo

```git clone https://github.com/morishuz/udi-capstone-selfie.git```

go inside the directory

```cd udi-capstone-selfie```

build and launch the docker container (bulid only needs to be run once!)

```make build```
```make run```

This will now run docker in bash mode. Run project as usual, i.e.:

```
cd capstone/ros
catkin_make
source devel/setup.sh
roslaunch launch/styx.launch
```

Launch the external simulator ([download here](https://github.com/udacity/CarND-Capstone/releases))

1. enable the "camera" check-box.  
2. disable the "manual" check-box. 




## From original README:


### Real world testing
1. Download [training bag](https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/traffic_light_bag_file.zip) that was recorded on the Udacity self-driving car.
2. Unzip the file
```bash
unzip traffic_light_bag_file.zip
```
3. Play the bag file
```bash
rosbag play -l traffic_light_bag_file/traffic_light_training.bag
```
4. Launch your project in site mode
```bash
cd CarND-Capstone/ros
roslaunch launch/site.launch
```
5. Confirm that traffic light detection works on real life images
