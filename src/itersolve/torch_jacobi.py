import torch
from torch.optim import Adam
import time

def torch_jacobi_solve(A, b, optimizer="Adam", hparams={"lr": 0.1}, max_iter=1000, residual_tol=1e-5, seed=None):
    """Solves the equation Ax=b via the Jacobi iterative method, using pytorch optimizers"""

    # set seed
    if seed is not None:
        torch.manual_seed(seed)
    x0 = torch.randn(dim, requires_grad=True)


    # Jacobi decomposition of A
    D = torch.diag(A)
    # R = A - torch.diagflat(D)

    # create optimizer
    if optimizer is not None:
        optimizer_constructor = getattr(torch.optim, optimizer)
        optimizer = optimizer_constructor([x0], **hparams)

    # Iterate until convergence or until max_iter
    x = x0
    for i in range(max_iter):
        x_prev = x.clone()

        if optimizer is not None:
            optimizer.zero_grad()
            grad = -((b - torch.mv(A, x_prev)) / D)
            x.grad = grad
            optimizer.step()
        else:
            x = x_prev + hparams["lr"] * ((b - torch.mv(A, x_prev)) / D)

        # print("Residual: {}".format(redidual))
        # if exact_solution is not None:
        #     error = torch.norm(x - exact_solution)
            # print("Error: {}".format(error))

        # if torch.norm(x - x_prev) < tol:
        #     print("Converged in {} iterations".format(i))
        #     break
        
        if torch.norm(A @ x - b) < residual_tol:
            print("Converged in {} iterations".format(i))
            break

    if i == max_iter - 1:
        print("Max iterations reached with torch.norm(x - x_prev) = {}".format(torch.norm(x - x_prev)))
    return x


def get_random_diagonally_dominant_matrix(dim, conditioning_factor=1, sparsity=0.9):
    """Returns a random diagonally dominant matrix"""
    A = torch.randn(dim, dim, requires_grad=False)**2

    # make symmetric
    A = A + A.T 

    # make sparse
    A = A * (torch.rand(dim, dim, requires_grad=False) < sparsity)

    # make diagonally dominant
    A += A.sum(1).diag(0) + 1

    # make ill conditioned
    conditioner = torch.diag(torch.arange(1, conditioning_factor + 1, conditioning_factor / dim))
    A = conditioner @ A 
    return A


if __name__ == "__main__":

    dim=3000
    conditioning_factor = 5000
    sparsity = 0.99
    lr = 0.5
    torch.manual_seed(1)
    A = get_random_diagonally_dominant_matrix(dim, conditioning_factor, sparsity=sparsity)
    b = torch.randn(dim, requires_grad=False)


    t1 = time.time()
    exact_solution = torch.linalg.solve(A, b)
    direct_time = time.time() - t1

    t1 = time.time()
    sol = torch_jacobi_solve(A, b, optimizer="SGD", hparams={"lr": lr, "momentum": 0.1}, seed=1)
    optimizer_jacobi_time = time.time() - t1

    # torch.manual_seed(1)
    # x0 = torch.randn(dim, requires_grad=True)
    t1 = time.time()
    x0 = torch.randn(dim, requires_grad=True)
    sol = torch_jacobi_solve(A, b, optimizer=None, hparams={"lr": lr}, seed=1)
    weighted_jacobi_time = time.time() - t1

    print("Time to solve Ax=b direct: {}".format(direct_time))
    print("Time to solve Ax=b via optimizer Jacobi: {}".format(optimizer_jacobi_time))
    print("Time to solve Ax=b via weighted Jacobi: {}".format(weighted_jacobi_time))