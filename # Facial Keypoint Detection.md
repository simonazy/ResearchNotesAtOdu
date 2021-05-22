# Facial Keypoint Detection

This project will be all about defining and training a convolutional neural network to perform facial keypoint detection, and using computer vision techniques to transform images of faces. 

Facial keypoints (also called facial landmarks) are the small magenta dots shown on each of the faces in the image above. In each training and test image, there is a single face and 68 keypoints, with coordinates (x, y), for that face. These keypoints mark important areas of the face: the eyes, corners of the mouth, the nose, etc.

## Load and visualize data

Training and Testing Data
This facial keypoints dataset consists of 5770 color images. All of these images are separated into either a training or a test set of data.

- 3462 of these images are training images, for you to use as you create a model to predict keypoints.
- 2308 are test images, which will be used to test the accuracy of your model.

Download the data by it's URL and unzip the data in a /data/ directory
```
!mkdir /data
!wget -P /data/ https://s3.amazonaws.com/video.udacity-data.com/topher/2018/May/5aea1b91_train-test-data/train-test-data.zip
!unzip -n /data/train-test-data.zip -d /data
```

```python
key_pts_frame = pd.read_csv('data/train-test-data/training_frames_keypoints.csv')

# choose the first image
n = 0
image_name = key_pts_frame.iloc[n,0]  #slice
key_pts = key_pts_frame.iloc[n,1:].values

key_pts = key_pts.astype('float').reshape(-1,2) # -1 means an unknown dimension

print('image name: ', image_name)
print('Landmarks shape: ', key_pts.shape)
print('First 4 key pts:{}'.format(key_pts[:4]))
```
out:
image name:  Luis_Fonsi_21.jpg

Landmarks shape:  (68, 2)

First 4 key pts:[[ 45.  98.]
 [ 47. 106.]
 [ 49. 110.]
 [ 53. 119.]]

 We could write a function to show an image
 ```python
def show_image(image_name, key_pts):
    plt.imshow(image)
    plt.scatter(key_pts[:,0], key_pts[:,1], s=20, marker='.', c = 'm')

plt.figure(figsize=(5, 5))
show_keypoints(mpimg.imread(os.path.join('data/train-test-data/training/', image_name)), key_pts)
plt.show()
 ```
 <p align='center'>
 <img src='images/example_1.png' width='150px'>
 </p>

## Dataset class and Transformations
Dataset class

`torch.utils.data.Dataset` is an abstract class representing a dataset. This class will allow us to load batches of image/keypoint data, and uniformly apply transformations to our data, such as rescaling and normalizing images for training a neural network.

Your custom dataset should inherit `Dataset` and override the following methods:
- __len__ so that len(dataset) returns the size of the dataset.
- __getitem__ to support the indexing such that dataset[i] can be used to get the i-th sample of image/keypoint data.
  

Let's create a dataset class for our face keypoints dataset. We will read the CSV file in __init__ but leave the reading of images to __getitem__. This is memory efficient because all the images are not stored in the memory at once but read as required.

A sample of our dataset will be a dictionary {'image': image, 'keypoints': key_pts}. 

```python
from torch.utils.data import Dataset, DataLoader

class FacialKeypointsDataset(Dataset):
    def __init__(self,csv_file, root_dir,transform=None):
        """Args
            csv_file: Path to the annotations
            root_dir: Dictionary with all the images
            transform: Optional transform to be applied
        """
        self.key_pts_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.key_pts_frame)

    def __getitem__(self, idx):
        #concat path with image name
        image_name = os.path.join(self.root_dir, self.key_pts_frame.iloc[idx,0])
        image = mpimg.imread(image_name)

        # if image has an alpha color channel, get rid of it
        if(image.shape[2]==4):
            image = image[:,:,0:3]

        key_pts = self.key_pts_frame.iloc[idx, 1:].values
        key_pts = key_pts.astype('float').reshape(-1,2)
        sample = {'image':image, 'keypoints': key_pts}

        if self.transform:
            sample = self.transform(sample)
        # format is image, keypoints. Data with annotations. 
        return sample
```
Display some images:
```python
num_to_play = 3 

for i in range(num_to_play):
    fig = plt.figure(figsize=(10,5))
    rand_i = np.random.randint(0, len(face_dataset))
    sample = face_dataset[rand_i]

    print(i, sample['image'].shape, sample['keypoints'].shape)

    ax = plt.subplot(1, num_to_play, i+1)
    ax.set_title('Sample #{}'.format(i))
    show_keypoints(sample['image'],sample['keypoints'])
```
output: 

    0 (191, 227, 3) (68, 2)

    1 (100, 94, 3) (68, 2)

    2 (203, 196, 3) (68, 2)

Our dataset will take an optional argument transform so that any required processing can be applied on the sample.

## Transform
Now, the images above are not of the same size, and neural networks often expect images that are standardized; a fixed size, with a normalized range for color ranges and coordinates, and (for PyTorch) converted from numpy lists and arrays to Tensors.

Therefore, we will need to write some pre-processing code. Let's create four transforms:
- `Normalize:` to convert a color image to grayscale values with a range of [0,1] and normalize the keypoints to be in a range of about [-1, 1].
- `Rescale: `to rescale an image to a desired size.
- `RandomCrop:` to crop an image randomly.
- `ToTensor:` to convert numpy images to torch images.

We will write them as callable classes instead of simple functions so that parameters of the transform need not be passed everytime it's called. For this, we just need to implement `__call__` method and (if we require parameters to be passed in), the `__init__` method.

- Normalize
  
```python
class Normalize(object):

    def __call__(self, sample):
        image, key_pts = sample['image'],  sample['keypoints']

        image_copy = np.copy(image)
        key_pts_copy = np.copy(key_pts)
        
        # convert image to grayscale
        image_copy = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # scale color range from [0, 255] to [0, 1]
        image_copy=  image_copy/255.0

        # scale keypoints to be centered around 0 with a range of [-1, 1]
        # mean = 100, sqrt = 50, so, pts should be (pts - 100)/50
        key_pts_copy = (key_pts_copy - 100)/50.0

        return {'image': image_copy, 'keypoints': key_pts_copy}

```
- Rescale
```python
class Rescale(object):
    """Args:
        output_size(tuple or int): Desired output size
    """
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, key_pts = sample['image'], sample['keypoints']

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size*h/w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size*w/h 
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        image = cv2.resize(image, (new_w, new_h))

        # scale the pts, too
        key_pts = key_pts * [new_w / w, new_h / h]

        return {'image': image, 'keypoints': key_pts}
```
- Crop
  
```python
class RandomCrop(object):
    """Crop randomly the image in a sample.
    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """
    def __init__(self, output_size):
        assert isinstance(output_size,(int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size
    def __call__(self, sample):
        image, key_pts = sample['image'], sample['keypoints']

        h,w = image.shape[:2]
        new_h, new_w = self.output_size 

        top = np.random.randint(0, h-new_h)
        left = np.random.randint(0, w - new_w)
        image = image[top: top+new_h, left:left+new_w]

        key_pts = key_pts - [left, top]

        return {'image': image, 'keypoints': key_pts}
```

- To tensor
```python
class ToTensor(object):
    def __call__(self, sample):
        image, key_pts = sample['image'], sample['keypoints']
        # if image has no grayscale color channel, add one
        if (len(image.shape) == 2):
            # add that third color dim
            image = image.reshape(image.shape[0], image.shape[1],1)
        
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose(2, 0, 1)

        return {'image':torch.from_numpy(image), 
                'keypoints':torch.from_numpy(key_pts)}
```
### Test transforms
As you look at each transform, note that, in this case, order does matter. For example, you cannot crop a image using a value smaller than the original image (and the original images vary in size!), but, if you first rescale the original image, you can then crop it to any size smaller than the rescaled size.

```python
import torch
from torchvision import transforms, utils

rescale = Rescale(100)
crop = RandomCrop(50)
composed = transform.Compose([Rescale(250), RandomCrop(224)])

# apply the transforms to a sample image
test_num = 500
sample = face_dataset[test_num]

fig = plt.figure()
for i, tx in enumerate([rescale, crop, composed]):
    transformed_sample = tx(sample)

    ax = plt.subplot(1, 3, i+1)
    plt.tight_layout()
    ax.set_title(type(tx).__name__)
    show_keypoints(transformed_sample['image'], transformed_sample['keypoints'])
plt.show()
```
<p align='center'>
<img src='example_2.png' width='350px'>
</p>

Now, we can create the transformed dataset
```python
data_transform = transform.Compose([Rescale(250),
                                     RandomCrop(224),
                                     Normalize(),
                                     ToTensor()])

transformed_dataset = FacialKeypointsDataset(
    csv_file='/data/train-test-data/training_frames_keypoints.csv',
    root_dir='/data/train-test-data/training/',
    transform=data_transform)
```

## Data Iteration and Batching
Right now, we are iterating over this data using a for loop, but we are missing out on a lot of PyTorch's dataset capabilities, specifically the abilities to:

- Batch the data
- Shuffle the data
- Load the data in parallel using multiprocessing workers.

torch.utils.data.DataLoader is an iterator which provides all these features, Next, we will load data in batches to train a neural network!

### Batch size
Decide on a good batch size for training your model.Too large a batch size may cause your model to crash and/or run out of memory while training.
```python
batch_size = 24
train_loader = DataLoader(transformed_dataset, 
                          batch_size=batch_size,
                          shuffle=True, 
                          num_workers=4)
```
### Before training
Take a look at how this model performs before it trains. You should see that the keypoints it predicts start off in one spot and don't match the keypoints on a face at all! 
```python
batch_size = 24

test_loader = DataLoader(test_dataset, 
                          batch_size=batch_size,
                          shuffle=True, 
                          num_workers=4)
```

### Apply the model on a test sample

To test the model on a test sample of data, you have to follow these steps:

- Extract the image and ground truth keypoints from a sample
- Wrap the image in a Variable, so that the net can process it as input and track how it changes as the image moves through the network.
- Make sure the image is a FloatTensor, which the model expects.
- Forward pass the image through the net to get the predicted, output keypoints.

```python
# test the model on a batch of test images
def net_sample_output():
    
    # iterate through the test dataset
    for i, sample in enumerate(test_loader):
        
        # get sample data: images and ground truth keypoints
        images = sample['image']
        key_pts = sample['keypoints']

        # convert images to FloatTensors
        images = images.type(torch.FloatTensor)

        # forward pass to get net output
        output_pts = net(images)
        
        # reshape to batch_size x 68 x 2 pts
        output_pts = output_pts.view(output_pts.size()[0], 68, -1)
        
        # break after first image is tested
        if i == 0:
            return images, output_pts, key_pts     
```
### Visualize the predicted keypoints
Once we've had the model produce some predicted output keypoints, we can visualize these points in a way that's similar to how we've displayed this data before, only this time, we have to `"un-transform"` the image/keypoint data to display it.
```python
def show_all_keypoints(image, predicted_key_pts, gt_pts=None):
    """Show image with predicted keypoints"""
    # image is grayscale
    plt.imshow(image, cmap='gray')
    plt.scatter(predicted_key_pts[:, 0], predicted_key_pts[:, 1], s=20, marker='.', c='m')
    # plot ground truth points as green pts
    if gt_pts is not None:
        plt.scatter(gt_pts[:, 0], gt_pts[:, 1], s=20, marker='.', c='g')
```
### Un-transformation
```python
def visualize_output(test_images, test_outputs, gt_pts=None, batch_size=24):
    for i in range(batch_size):
        plt.figure(figsize=(20,10))
        ax = plt.subplot(1, batch_size, i+1)
        
        # un-transform the image data
        image = test_images[i].data   # get the image from it's Variable wrapper
        image = image.numpy()   # convert to numpy array from a Tensor
        image = np.transpose(image, (1, 2, 0))   # transpose to go from torch to numpy image

        # un-transform the predicted key_pts data
        predicted_key_pts = test_outputs[i].data
        predicted_key_pts = predicted_key_pts.numpy()
        # undo normalization of keypoints  
        predicted_key_pts = predicted_key_pts*50.0+100
        
        # plot ground truth points for comparison, if they exist
        ground_truth_pts = None
        if gt_pts is not None:
            ground_truth_pts = gt_pts[i]         
            ground_truth_pts = ground_truth_pts*50.0+100
        # call show_all_keypoints
        show_all_keypoints(np.squeeze(image), predicted_key_pts, ground_truth_pts)

           plt.axis('off')

    plt.show()
# call it
visualize_output(test_images, test_outputs, gt_pts)
```

### Training
**Loss function**

Training a network to predict keypoints is different than training a network to predict a class; instead of outputting a distribution of classes and using cross entropy loss, you may want to choose a loss function that is suited for regression, which directly compares a predicted value and target value. 

```python
import torch.optim as optim

criterion = nn.SmoothL1Loss()

optimizer = optim.Adam(net.parameters(), lr=0.001)
```
Define the training and process and initial observations.

```python
def train_net(n_epochs):
    # prepare the net for training
    net.train()

    for epoch in range(n_epochs):  # loop over the dataset multiple times
        
        running_loss = 0.0

        # train on batches of data, assumes you already have train_loader
        for batch_i, data in enumerate(train_loader):
            # get the input images and their corresponding labels
            images = data['image']
            key_pts = data['keypoints']

            # flatten pts
            key_pts = key_pts.view(key_pts.size(0), -1)

            # convert variables to floats for regression loss
            key_pts = key_pts.type(torch.FloatTensor)
            images = images.type(torch.FloatTensor)

            # forward pass to get outputs
            output_pts = net(images)

            # calculate the loss between predicted and target keypoints
            loss = criterion(output_pts, key_pts)

            # zero the parameter (weight) gradients
            optimizer.zero_grad()

             # backward pass to calculate the weight gradients
            loss.backward()

            # update the weights
            optimizer.step()

            # print loss statistics
            running_loss += loss.item()

            if batch_i % 24 == 23:
                print('Epoch: {}, Batch:{}, Avg. Loss:{}'.format(epoch+1, batch_i+1, running_loss/24))
                running_loss = 0

    print('Finished Training!')
```
## Save 
After testing the net on the test dataset, finally we can save the model!
```python
!model_dir = 'saved_models/'
model_name = 'keypoints_model_1.pt'

# after training, save your model parameters in the dir 'saved_models'
torch.save(net.state_dict(), model_dir+model_name)
```
