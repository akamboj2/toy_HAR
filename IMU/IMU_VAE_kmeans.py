from IMU_VAE import *

"""
What's the goal here?
Try to learn k-means clustering on latent space of VAE
with k=28

and see if those k clusters correspond to the different actions
"""


if __name__=='__main__':
    """ NOTE: This main function was just for debugging let's import and train this in train.py"""
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    epochs= 500 #1000
    latent_dims= 4 #512
    batch_size = 32 #lower bs seems to decrease mse loss and increase kl loss
    beta = 1/2000 #weight of kl divergence
    """
    total 370, 93, 277: 100,50, 32
    150: 100,10,32
    140, 95, 44: 100,8,32
    104, mse 93, kl 11: 100,2, 32
    """
     # Load dataset
    dir = "/home/akamboj2/data/utd-mhad/Inertial_splits/action_80_20_#1/train.txt"
    # dir = "/home/abhi/data/USC-HAD/splits/train.txt"
    train_dataset = IMUDataset(dir, dataset_name="UTD")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    vae = VariationalAutoencoder(latent_dims).to(device).float() # GPU
    # vae = Autoencoder(latent_dims).to(device).float() # GPU
    opt = optim.Adam(vae.parameters())
    for epoch in range(epochs):
        for x, y in train_loader:
            x = x.to(device) # GPU
            x = F.normalize(x, dim=1) # prevents output x_hat from going to infinity
            opt.zero_grad()
            x_hat = vae(x)
            loss_mse = ((torch.flatten(x, start_dim=1).float() - torch.flatten(x_hat, start_dim=1))**2).sum()
            loss_kl =  vae.encoder.kl
            loss = loss_mse + beta*loss_kl
            loss.backward()
            opt.step()
            
            # exit()
        if epoch % 10 == 0:
            print("MSE Loss:", loss_mse.item(), "KL Loss:", loss_kl.item())
            # print("MSE Loss:", loss_mse.item())
            print(f"Epoch {epoch+1}/{epochs}")

    print("Finished train VAE")

    # Now perform kmeans clustering on latent space:
    k = 27 #number of classes
    means = torch.rand((27,latent_dims)).to(device)
    # assignments = torch.zeros((len(train_dataset)))
    counts = torch.zeros((27)).to(device)
    for x, y in train_dataset:
        x = x.unsqueeze(0).to(device)
        # x = F.normalize(x, dim=1)
        x = F.normalize(x, dim=1) # prevents output x_hat from going to infinity
        z = vae.encoder(x)
        means[y] = means[y] + z
        counts[y] += 1

    for i in range(len(means)):
        means[i] = means[i]/counts[i]
    print("Means:", means.shape)

    #Now test on test set
    # Test model
    datapath = "Inertial_splits/action_80_20_#1"
    val_dir = os.path.join("/home/akamboj2/data/utd-mhad/",datapath,"val.txt")
    val_dataset = IMUDataset(val_dir, time_invariance_test=False)
    # val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    print("Testing model")
    correct = 0
    for x,y in val_dataset:
        # input = vae.encoder(F.normalize(inputs, dim=1).to(device))
        x = x.unsqueeze(0).to(device)
        x = F.normalize(x, dim=1) # prevents output x_hat from going to infinity
        z = vae.encoder(x)
        min_dist = torch.inf 
        for i in range(len(means)):
            # print("z",z.shape)
            # print("Means:", means[i].shape)
            dist = torch.norm(z-means[i])
            if dist < min_dist:
                min_dist = dist
                predicted = i
        if y == predicted:
            correct += 1

    print("Accuracy: ", 100*correct/len(val_dataset), "%")
    # During validation getting a 40% accuracy
    
    



        
    
# From Wikipedia: (Also downloaded pdf in downloads if needed)
"""
Standard algorithm (naive k-means)

Convergence of k-means
The most common algorithm uses an iterative refinement technique. Due to its ubiquity, it is often called "the k-means algorithm"; it is also referred to as Lloyd's algorithm, particularly in the computer science community. It is sometimes also referred to as "naïve k-means", because there exist much faster alternatives.[6]

Given an initial set of k means m1(1), ..., mk(1) (see below), the algorithm proceeds by alternating between two steps:[7]

Assignment step: Assign each observation to the cluster with the nearest mean: that with the least squared Euclidean distance.[8] (Mathematically, this means partitioning the observations according to the Voronoi diagram generated by the means.)

,
{\displaystyle S_{i}^{(t)}=\left\{x_{p}:\left\|x_{p}-m_{i}^{(t)}\right\|^{2}\leq \left\|x_{p}-m_{j}^{(t)}\right\|^{2}\ \forall j,1\leq j\leq k\right\},}
where each 

x_{p} is assigned to exactly one 

S^{(t)}, even if it could be assigned to two or more of them.
Update step: Recalculate means (centroids) for observations assigned to each cluster.

{\displaystyle m_{i}^{(t+1)}={\frac {1}{\left|S_{i}^{(t)}\right|}}\sum _{x_{j}\in S_{i}^{(t)}}x_{j}}
The algorithm has converged when the assignments no longer change. The algorithm is not guaranteed to find the optimum.[9]

The algorithm is often presented as assigning objects to the nearest cluster by distance. Using a different distance function other than (squared) Euclidean distance may prevent the algorithm from converging. Various modifications of k-means such as spherical k-means and k-medoids have been proposed to allow using other distance measures.
"""