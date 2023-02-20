import torch 
import torch.nn as nn 
import torch.optim as optim 
import torch.distributions as D 
import torch.functional as F 
import matplotlib.pyplot as plt 

class gaussian_mix(nn.Module):
    def __init__(self,input_size:int,num_gaussians:int,dimensions_gaussian:int) -> None:
        super().__init__()
        self.num_gaussians = num_gaussians
        self.dimensions_gaussian = dimensions_gaussian
        self.layer_means = nn.Linear(input_size,num_gaussians*dimensions_gaussian)
        self.layer_std = nn.Linear(input_size,num_gaussians*dimensions_gaussian)
        self.layer_weights = nn.Linear(input_size,num_gaussians)

    def forward(self,x:torch.Tensor) -> D.MixtureSameFamily:
        batch_size = x.size(0)
        means = self.layer_means(x).view(batch_size,self.num_gaussians,self.dimensions_gaussian) #(batch,num_gaussians,dimensions_gaussian)
        std = (nn.ELU()(self.layer_std(x))+1+1e-15).view(batch_size,self.num_gaussians,self.dimensions_gaussian)
        weights = nn.Softmax(dim=1)(self.layer_weights(x))
        
        mix = D.Categorical(weights)
        comp = D.Independent(D.Normal(loc=means,scale= std), 1) 
        dist = D.MixtureSameFamily(mix, comp)

        return dist


if __name__ == "__main__":
    batch_size = 1 
    input_size = 10
    amount_gaussians = 5
    dimensions_gaussian = 1

    mix = gaussian_mix(input_size,amount_gaussians,dimensions_gaussian)
    input = torch.randn(batch_size,input_size)
    targets = torch.tensor([[5.0],[-5.0]])
    
    distribution = mix(input)
    
    optimizer = optim.Adam(mix.parameters(),lr=0.005)

    plt.ion()
    for i in range(1000):
        optimizer.zero_grad()
        distribution = mix(input)
        noise = torch.randn_like(targets)
        loss = torch.sum(-distribution.log_prob(targets+noise))
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            x = torch.linspace(-10,10,10000).view(-1,1)
            plt.cla()
            plt.plot(x,torch.exp(distribution.log_prob(x)))
            plt.pause(0.1)
    plt.show()
