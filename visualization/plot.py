import matplotlib.pyplot as plt
import torch
import numpy as np

def plot_prediction(N: int, 
                    dimensions: tuple,
                    inputs: torch.Tensor, 
                    outputs: torch.Tensor, 
                    prediction: torch.Tensor,
                    path: str,
                    is_cifar: bool = False):
    print(inputs.shape[0])
    assert inputs.shape[0]>=N
    inputs = inputs.detach().cpu().numpy()
    outputs = outputs.detach().cpu().numpy()
    prediction = prediction.detach().cpu().numpy()

    image_lists = [inputs[:N], outputs[:N], prediction[:N]]  # Only take first N images from each list
    channels = inputs.shape[1] + outputs.shape[1] + prediction.shape[1]
    
    if not is_cifar:
        # Create a figure with 3 rows and N columns
        fig, axes = plt.subplots(channels, N, figsize=(4 * N, channels * 3))

        # Plot images
        for col in range(N):
            cnt = 0
            for which, data in enumerate(image_lists):
                for c in range(data.shape[1]):
                    to_plot = data[col,c]
                    axes[cnt, col].imshow(to_plot, cmap = 'gist_ncar')
                    cnt+=1
                    
        plt.tight_layout()
        plt.show()
        plt.savefig(path, dpi = 400)
        return fig
    else:
        fig = show_images(prediction[:, :3], N, path)
        return fig

def show_images(images, N, path):
    images = (255*(images*0.5 + 0.5)).astype("uint8")
    images = np.transpose(images, axes=[0,2,3,1])
    fig = plt.figure(figsize=(25, 4))
    for idx in np.arange(N):
        ax = fig.add_subplot(2, N//2, idx+1, xticks=[], yticks=[])
        plt.imshow(images[idx])

    plt.tight_layout()
    plt.show()
    plt.savefig(path, dpi = 400)
    return fig