# Week 4 deliverables
Jason Yim

### Cell counting image generation tool
- Based on Ailey-dev, created a Clarity-like image generation tool. 
- Tool is very basic for now. Able to create ellipsoidal like objects with different properties like size, locations, colors, fade, intensity.
- It is very customizeable and anyone on the team can use it easily.
### Sample generated images
![](https://user-images.githubusercontent.com/8682187/30778558-c164613e-a0a6-11e7-8f56-059be39c0da6.png)
![](https://user-images.githubusercontent.com/8682187/30778560-c56f44ce-a0a6-11e7-9d98-0d2991814f1c.png)
![](https://user-images.githubusercontent.com/8682187/30778554-bac3d7ba-a0a6-11e7-8ac3-edcec7664e22.png)
![](https://user-images.githubusercontent.com/8682187/30778556-be434128-a0a6-11e7-9920-24d73dea02fc.png)

#### Uses
- We plan on using these images to evaluate and validate our algorithms. We can control how many cells are in each image, the overlapping of cells, 
their distribution, their shapes, the sizes of the volumes, etc.
- if we go to deep learning then suing generated training data like this is invaluable. 
#### Further work
- Right now it can only generate ellipsoidal cell objects. I plan on allowing for different cell shapes like gaussian blobs with varying 
covariances and different irregular cell shapes.
#### Other things
- Set up my environment more by creating lots of helper classes/functions

### Next week
- Will pick an algorithm and start implementing it.
- As the needs arise I will work more on this image generation tool.
