## Pytorch_Resnet ##
This is the implementation of ResNet-20 for cifar-10 dataset. The architecture of the network  
and initialization of weights refers to [the ResNet paper][1]. Due to the lack of computation  
resources, I can only do the experiment of ResNet-20 for cifar-10 dataset. The network do not  
rely on the pretrain model and can be trained from scratch.

### Requirements ###
> pytorch0.2.0  
> python3.5

### Train and Test ###
> This model would be trained for 160 epoches, and would be tested after training time. If you  
> want to save the model, you could add `torch.save(res_net.state_dict(), PATH)` after training.  
#### Run the model ####
> `python model.py` 
  
[1]:https://arxiv.org/pdf/1512.03385.pdf
