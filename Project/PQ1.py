import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import random
import torch
import torch.linalg
import numpy as np
from scipy import ndimage

TORCH_DEVICE = torch.device('cuda')

# Load images
images = [[mpimg.imread(f"data/{i}a.jpg"), mpimg.imread(f"data/{i}b.jpg")] for i in range(1,201)]
fig = plt.figure(figsize=(15, 10))
for i in range(5):
    index = random.randint(0, len(images)-1)
    # First image
    fig.add_subplot(5, 2, i*2+1)
    plt.imshow(images[index][0], cmap='gray')
    plt.axis('off')
    # Second
    fig.add_subplot(5, 2, i*2+2)
    plt.imshow(images[index][1], cmap='gray')
    plt.axis('off')
#plt.show()

# Convert first 190 into matrix
gamma_matrix = np.array([images[i][0].flatten() for i in range(190)], dtype=np.float32).T
gamma_matrix = torch.tensor(gamma_matrix, dtype=torch.float32, device=TORCH_DEVICE)

# Mean image
psi = torch.mean(gamma_matrix, 1)
fig = plt.figure()
fig.add_subplot(1, 1, 1)
plt.imshow(psi.cpu().detach().numpy().reshape(images[0][0].shape), cmap='gray')
plt.axis('off')
#plt.show()

# Covariance matrix
a_matrix = torch.t(torch.t(gamma_matrix) - psi)
eigenvectors, eigenvalues, v = torch.linalg.svd(a_matrix)
del(a_matrix)
del(v) # No need for this
eigenvectors = eigenvectors[:,0:len(eigenvalues)]
eigenvalues = torch.pow(eigenvalues, 2)
# Show egenvalues
fig = plt.figure()
plt.plot(np.array(list(range(1,len(eigenvalues)+1))), eigenvalues.cpu().detach().numpy())
#plt.show()
# Show first 5
fig = plt.figure()
for i in range(5):
    fig.add_subplot(1, 5, i+1)
    plt.imshow(eigenvectors[:,i].cpu().detach().numpy().reshape(images[0][0].shape), cmap='gray')
    plt.axis('off')
#plt.show()
# Cutoff
K = 100

# Reconstruct
def reconstruct_image(orig: torch.Tensor, pcs: torch.Tensor):
    """
    A function to reconstruct the original image with the given PCs
    Arguments
    orig - Original image to perform reconstruction on
    pcs  - The principal components to use for the reconstruction
    
    Returns
    recon - The reconstruction of the orig from pcs
    mse   - The Mean Squared Error of the recon with orig
    """
    
    w = torch.matmul(torch.t(pcs), orig - psi)
    recon = torch.matmul(pcs, w) + psi
    mse = torch.nn.MSELoss()
    return recon, float(mse(recon, orig).item())

mses = []
img = torch.tensor(images[random.randint(0, len(images)-11)][0].flatten(), dtype=torch.float32, device=TORCH_DEVICE)
for i in range(1,len(eigenvalues)):
    pcs = eigenvectors[:,0:i]
    mses.append(reconstruct_image(img, pcs)[1])
del(img)
print("Max mse in index:", mses.index(max(mses)), "; value of it:", max(mses))
print("Min mse in index:", mses.index(min(mses)), "; value of it:", min(mses))
# Show them in plot
fig = plt.figure()
plt.plot(np.array(list(range(1,len(eigenvalues)))), np.array(mses))
#plt.show()
del(mses)
# Reconstruct using k with 30 intervals
fig = plt.figure()
img = torch.tensor(images[random.randint(0, len(images)-11)][0].flatten(), dtype=torch.float32, device=TORCH_DEVICE)
for i in range(5):
    k = 30 + i * 30
    subplot = fig.add_subplot(1, 6, i+1)
    subplot.title.set_text("K = " + str(k))
    pcs = eigenvectors[:,0:k]
    plt.imshow(reconstruct_image(img, pcs)[0].cpu().detach().numpy().reshape(images[0][0].shape), cmap='gray')
    plt.axis('off')
subplot = fig.add_subplot(1, 6, 6)
subplot.title.set_text("Original")
plt.imshow(img.cpu().detach().numpy().reshape(images[0][0].shape), cmap='gray')
plt.axis('off')
#plt.show()
del(subplot)
del(img)

# Smiling images
random_index = random.randint(0, len(images)-11)
normal_img = torch.tensor(images[random_index][0].flatten(), dtype=torch.float32, device=TORCH_DEVICE)
smiling_img = torch.tensor(images[random_index][1].flatten(), dtype=torch.float32, device=TORCH_DEVICE)
mses = []
for i in range(1,len(eigenvalues)):
    pcs = eigenvectors[:,0:i]
    mses.append([reconstruct_image(normal_img, pcs)[1], reconstruct_image(smiling_img, pcs)[1]])
mses = np.array(mses)
# Show them in plot
fig = plt.figure()
plt.plot(np.array(list(range(1,len(eigenvalues)))), mses[:,0], label = "non smiling")
plt.plot(np.array(list(range(1,len(eigenvalues)))), mses[:,1], label = "smiling")
plt.legend()
#plt.show()
del(mses)
# Reconstruct using k with 30 intervals
fig = plt.figure()
for i in range(5):
    k = 30 + i * 30
    subplot = fig.add_subplot(2, 6, i+1)
    subplot.title.set_text("K = " + str(k))
    pcs = eigenvectors[:,0:k]
    plt.imshow(reconstruct_image(normal_img, pcs)[0].cpu().detach().numpy().reshape(images[0][0].shape), cmap='gray')
    plt.axis('off')
    fig.add_subplot(2, 6, 6+i+1)
    plt.imshow(reconstruct_image(smiling_img, pcs)[0].cpu().detach().numpy().reshape(images[0][0].shape), cmap='gray')
    plt.axis('off')
subplot = fig.add_subplot(2, 6, 6)
subplot.title.set_text("Original")
plt.imshow(normal_img.cpu().detach().numpy().reshape(images[0][0].shape), cmap='gray')
plt.axis('off')
fig.add_subplot(2, 6, 12)
plt.imshow(smiling_img.cpu().detach().numpy().reshape(images[0][0].shape), cmap='gray')
plt.axis('off')
#plt.show()
del(subplot)
del(random_index)
del(normal_img)
del(smiling_img)

# Other images
mses = []
img = torch.tensor(images[190 + random.randint(0, 9)][0].flatten(), dtype=torch.float32, device=TORCH_DEVICE)
for i in range(1,len(eigenvalues)):
    pcs = eigenvectors[:,0:i]
    mses.append(reconstruct_image(img, pcs)[1])
print("Max mse in index:", mses.index(max(mses)))
print("Min mse in index:", mses.index(min(mses)))
# Show them in plot
fig = plt.figure()
plt.plot(np.array(list(range(1,len(eigenvalues)))), np.array(mses))
#plt.show()
del(mses)
# Reconstruct using k with 30 intervals
fig = plt.figure()
for i in range(5):
    k = 30 + i * 30
    subplot = fig.add_subplot(1, 6, i+1)
    subplot.title.set_text("K = " + str(k))
    pcs = eigenvectors[:,0:k]
    plt.imshow(reconstruct_image(img, pcs)[0].cpu().detach().numpy().reshape(images[0][0].shape), cmap='gray')
    plt.axis('off')
subplot = fig.add_subplot(1, 6, 6)
subplot.title.set_text("Original")
plt.imshow(img.cpu().detach().numpy().reshape(images[0][0].shape), cmap='gray')
plt.axis('off')
#plt.show()
del(subplot)
del(img)

# Non human image
# convert sticker.webp -colorspace Gray -resize 162x193\! sticker.jpg
img1 = mpimg.imread("data/non-human1.jpg")
img1 = torch.tensor(img1.flatten(), dtype=torch.float32, device=TORCH_DEVICE)
img2 = mpimg.imread("data/non-human2.jpg")
img2 = torch.tensor(img2.flatten(), dtype=torch.float32, device=TORCH_DEVICE)
reconstructed_image1, mse1 = reconstruct_image(img1, eigenvectors)
reconstructed_image2, mse2 = reconstruct_image(img2, eigenvectors)
print("MSE for first image is:", mse1)
print("MSE for second image is:", mse2)
fig = plt.figure()
fig.add_subplot(2, 2, 1)
plt.imshow(img1.cpu().detach().numpy().reshape(images[0][0].shape), cmap='gray')
plt.axis('off')
fig.add_subplot(2, 2, 2)
plt.imshow(reconstructed_image1.cpu().detach().numpy().reshape(images[0][0].shape), cmap='gray')
plt.axis('off')
fig.add_subplot(2, 2, 3)
plt.imshow(img2.cpu().detach().numpy().reshape(images[0][0].shape), cmap='gray')
plt.axis('off')
fig.add_subplot(2, 2, 4)
plt.imshow(reconstructed_image2.cpu().detach().numpy().reshape(images[0][0].shape), cmap='gray')
plt.axis('off')
#plt.show()
del(img1)
del(img2)
del(reconstructed_image1)
del(reconstructed_image2)
del(mse1)
del(mse2)

# Rotate
# Draw the plot of mses
main_image = images[random.randint(0, len(images)-11)][0]
mses = []
for degree in range(0, 360 + 1):
    rotated_image = ndimage.rotate(main_image, degree, reshape=False)
    rotated_image = torch.tensor(rotated_image.flatten(), dtype=torch.float32, device=TORCH_DEVICE)
    mses.append(reconstruct_image(rotated_image, eigenvectors)[1])
fig = plt.figure()
plt.plot(np.array(list(range(0, 360 + 1))), np.array(mses))
plt.show()
print("MSE in 0 degree rotate: ", mses[0])
print("MSE in 360 degree rotate: ", mses[360])
del(mses)
# Reconstruct images
images_in_plot = []
for degree in range(0, 360 + 1, 30):
    rotated_image = ndimage.rotate(main_image, degree, reshape=False)
    rotated_image_gpu = torch.tensor(rotated_image.flatten(), dtype=torch.float32, device=TORCH_DEVICE)
    images_in_plot.append([rotated_image, reconstruct_image(rotated_image_gpu, eigenvectors)[0].cpu().detach().numpy().reshape(images[0][0].shape)])

fig = plt.figure(figsize=(8, 6), dpi=200)
for i, img in enumerate(images_in_plot):
    subplot = fig.add_subplot(len(images_in_plot), 2, i*2+1)
    plt.imshow(img[0], cmap='gray')
    plt.axis('off')
    fig.add_subplot(len(images_in_plot), 2, i*2+2)
    plt.imshow(img[1], cmap='gray')
    plt.axis('off')
plt.show()
del(images_in_plot)
