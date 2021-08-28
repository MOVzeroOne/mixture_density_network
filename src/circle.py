import torch 
import torch.nn as nn 
import torch.optim as optim 
import torch.distributions as D 
import torch.functional as F 
import matplotlib.pyplot as plt 
import math 


class gaussian_mix(nn.Module):
    def __init__(self,input_size,num_gaussians,dimensions_gaussian=1):
        super().__init__()
        self.num_gaussians = num_gaussians
        self.dimensions_gaussian = dimensions_gaussian
        self.layer_means = nn.Linear(input_size,num_gaussians*dimensions_gaussian)
        self.layer_std = nn.Linear(input_size,num_gaussians)
        self.layer_weights = nn.Linear(input_size,num_gaussians)

    def forward(self,x):
        
        means = self.layer_means(x).view(x.size(0),self.num_gaussians,self.dimensions_gaussian) #(batch,num_gaussians,dimensions_gaussian)
        std = (nn.ELU()(self.layer_std(x))+1+1e-15).view(x.size(0),self.num_gaussians,self.dimensions_gaussian)
        weights = nn.Softmax(dim=1)(self.layer_weights(x))

        mix = D.Categorical(weights)
        comp = D.Independent(D.Normal(loc=means,scale= std), 1) 
        
        dist = D.MixtureSameFamily(mix, comp)
    
        return dist

class net(nn.Module):
    def __init__(self,input_size=1,input_size_mix=10,hidden_size=100,num_gaussians=10,dimensions_gaussian=1):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(input_size,hidden_size),nn.ReLU(),nn.Linear(hidden_size,hidden_size),nn.ReLU(),nn.Linear(hidden_size,input_size_mix))
        self.mix = gaussian_mix(input_size_mix,num_gaussians,dimensions_gaussian)

    def forward(self,x):
        return self.mix(self.layers(x))


def circle(number_samples,scale_noise=0):
    t = torch.linspace(0,2*math.pi,number_samples)
    radius = 1 + torch.randn(number_samples)*scale_noise
    y = radius*torch.sin(t)
    x = radius*torch.cos(t)
    return x,y 

if __name__ == "__main__":
    batch_size = 1000
    network = net()
    optimizer = optim.Adam(network.parameters(),lr=0.001)

    plt.ion()
    for i in range(1000):
        x,y = circle(batch_size)
        optimizer.zero_grad()
        distribution = network(x.view(-1,1))
        loss = torch.sum(-distribution.log_prob(y.view(-1,1)))
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            x,_ = circle(1000)
            plt.cla()
            plt.scatter(x,network(x.view(-1,1)).sample())
            plt.pause(0.1)
    plt.show()