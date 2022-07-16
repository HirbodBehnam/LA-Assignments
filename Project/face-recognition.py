import random
import torch
import torch.linalg
import numpy as np
import matplotlib.image as mpimg

def find_bases(images: torch.Tensor) -> torch.Tensor:
    """
    Finds the basis of the spaces of images

    Arguments
    images: The images to find basis of them. Note that each column must be an image

    Returns
    A tensor of nx10 which each column is the best space basis
    """

    psi = torch.mean(images, 1)
    a_matrix = torch.t(torch.t(images) - psi)
    eigenvectors = torch.linalg.svd(a_matrix)[0][:,0:10]
    return eigenvectors

def classify_image(basis: torch.Tensor, image: torch.Tensor) -> torch.Tensor:
    return torch.t(basis) @ image

images = torch.t(torch.tensor(np.array([mpimg.imread(f"data/{i}a.jpg").flatten() for i in range(1,201)]), dtype=torch.float32, device=torch.device('cuda')))
basis = find_bases(images)
del(images)
diffs = []
for i in range(1,200):
    mse = torch.nn.MSELoss(reduction='sum')
    first = classify_image(basis, torch.tensor(mpimg.imread(f"data/{i}a.jpg").flatten(), dtype=torch.float32, device=torch.device('cuda')))
    second = classify_image(basis, torch.tensor(mpimg.imread(f"data/{i}b.jpg").flatten(), dtype=torch.float32, device=torch.device('cuda')))
    diffs.append(int(mse(first, second).item()))
print(sorted(diffs)[-9:])
del(diffs)

THRESHOLD=4 * (10**7)
def same_person(image1: torch.Tensor, image2: torch.Tensor) -> bool:
    first = classify_image(basis, image1)
    second = classify_image(basis, image2)
    mse = torch.nn.MSELoss(reduction='sum')
    return mse(first, second).item() < THRESHOLD

# Test
true_positive = 0
true_negative = 0
false_positive = 0
false_negative = 0
# Tests positive
for i in range(1,200):
    first = torch.tensor(mpimg.imread(f"data/{i}a.jpg").flatten(), dtype=torch.float32, device=torch.device('cuda'))
    second = torch.tensor(mpimg.imread(f"data/{i}b.jpg").flatten(), dtype=torch.float32, device=torch.device('cuda'))
    if same_person(first, second):
        true_positive += 1
    else:
        false_negative += 1
for i in range(200):
    first_index = random.randrange(1, 200)
    second_index = random.randrange(1, 200)
    while second_index == first_index:
        second_index = random.randrange(1, 200)
    first = torch.tensor(mpimg.imread(f"data/{first_index}a.jpg").flatten(), dtype=torch.float32, device=torch.device('cuda'))
    second = torch.tensor(mpimg.imread(f"data/{second_index}a.jpg").flatten(), dtype=torch.float32, device=torch.device('cuda'))
    if same_person(first, second):
        false_positive += 1
    else:
        true_negative += 1

print("true negative:", true_negative)
print("true positive:", true_positive)
print("false negative:", false_negative)
print("false positive:", false_positive)