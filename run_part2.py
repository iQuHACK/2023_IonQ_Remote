#generates a specified number of images and their respective predictions
def generate_images(n_desired):

    count = 0
    fig, axes = plt.subplots(nrows=1, ncols=n_desired, figsize=(10, 3))

    model.eval()
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            if count == n_desired:
                break
            output = model(data)
        
            pred = output.argmax(dim=1, keepdim=True) 

            axes[count].imshow(data[0].numpy().squeeze())

            axes[count].set_xticks([])
            axes[count].set_yticks([])
            axes[count].set_title('Predicted {}'.format(pred.item()))
        
            count += 1
generate_images(3)