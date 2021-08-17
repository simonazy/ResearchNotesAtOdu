# Attack Methodology


Basic Iterative Method (BIM)

The Basic Iterative Method is presented by Kurakin and Goodfellow. It is the iterative version of Fast Gradient Method (FGM), also known as "Fast Gradient Sign Method (FGSM)，which is one of the easiest methods to generate adversarial images. FGM was firstly implemented by Good fellow.

There are four key elements in the Fast Gradient Method. 1)Input images. An input image is typically a 3-D tensor (width * height * channel). The value of image pixels are integers falling into the range of [0,255]. 2) Ground-truth label: Let y represents the ground-truth label. 3) J (, , y): J is the cost function of the neural network, given the input image  and ground-truth y, and  is the network weights. 4): correspond to the maximum magnitude of modifications for the image. 

![image](https://user-images.githubusercontent.com/56880104/128085599-48bd751e-ad08-4b15-abd5-fdbe67bca8a4.png)

During FGSM attack, we first calculate the gradient of input image  with respect to cost function, then we update the input image  with the calculated gradient to maximize cost function. To bound the magnitude of pixel modification, x is update by  = +() where   is the maximum step size allowed pixel change. 

When it comes to BIM, it is a straightforward way to extend the FGSM method by applying it multiple times with small step size, so BIM would be stronger than FGSM.  


ADVERSARIAL PATCH

Adversarial patch was firstly implemented by Brown. It is introduced as an universal, robust patch that fools DNN regardless of the scale or location of the patch. They are universal because they can be used to attack any scene, robust because they work under a wide variety of transformations. In realistic world, we can print out the patches and add it to any scene, while in simulator, we can try different size and location of the patch attached to the input images before they are passed to DNN.  
![image](https://user-images.githubusercontent.com/56880104/128085696-6d01cd43-64be-48b5-bf2d-23dff55d0f9b.png)

We use the patch application operator A that takes the patch, original image, location, scale as input and generate the tampered image. Then the image will be processed by the DNN to generate vehicle controls. This method is also called black-box attack because the patch attacker has no idea about the neural network’s architecture and parameters. 
