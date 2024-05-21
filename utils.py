import scipy.sparse

# import torch
from tqdm import tqdm
import numpy as np
import pandas as pd
import cvxpy as cp

import torch
import torch.nn as nn

import argparse
import utils
from scnn.metrics import Metrics
from scnn.models import ConvexGatedReLU, LinearModel
from scnn.activations import sample_gate_vectors
from scnn.solvers import RFISTA, CVXPYSolver
from scnn.optimize import optimize_model
from scnn.regularizers import NeuronGL1,L1

import time
from sklearn import linear_model
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

def one_hot(a, num_classes):
    return np.squeeze(np.eye(num_classes)[a.reshape(-1)])

def get_SJLT_matrix(m, n, s): 
    # This function returns SJLT sketching matrix in the form of a sparse matrix.
    # m: sketch size, n: number of data samples, s: sparsity
    nonzeros = 2*np.random.binomial(1, 0.5, size=(s*n)) - 1 # Rademacher random variables

    K = int(np.ceil(s*n / m)) # number of repetitions
    shuffled_row_indices = np.zeros((K*m), dtype=np.int32)
    all_row_indices = np.linspace(0, m-1, m, dtype=np.int32)
    
    for k in range(K):
        shuffled_row_indices[k*m:(k+1)*m] = np.random.permutation(all_row_indices)  

    I = shuffled_row_indices[0:s*n]
    J = np.repeat(np.linspace(0, n-1, n, dtype=np.int32), s)
    V = nonzeros

    S = scipy.sparse.coo_matrix((V,(I,J)), shape=(m,n), dtype=np.int8)
    S = S.tocsr()
    
    return S

def general_cross_product(X):
    # Input:
    #    X: (d-1)*d matrix
    # Output:
    #    v: d-dim vector
    d = X.shape[1]
    v = (-1*np.ones(d))**np.arange(d)
    base = np.arange(d)
    for i in range(d):
        X_i = X[:,np.delete(base,i,0)]
        v[i]*=np.linalg.det(X_i)
    return v

def general_cross_product_v2(X,max_trial=20):
    d = X.shape[1]
    iter_num = 0
    while iter_num<max_trial:
        v_aug = np.random.randn(d)
        X_aug = np.concatenate([X,v_aug.reshape([1,-1])],axis=0)
        if np.linalg.det(X_aug)!=0:
            break
        iter_num+=1
        # print(iter_num)
    if iter_num<max_trial:
        q = np.zeros(d)
        q[-1]=1
        v = np.linalg.solve(X_aug,q)
        return v, 0
    else:
        return None, -1

def general_cross_product_v3(X):
    # using svd
    U, s, Vt = np.linalg.svd(X.T)
    return U[:,-1]
    # s = np.sum(v)/np.linalg.det(X_aug)
    # v = s*v

def test_general_cross_product(d=3, gcp_ver='v3', max_trial=10):
    X = np.random.randn(d-1,d)
    if gcp_ver=='v2':
        v, flag = general_cross_product_v2(X, max_trial=max_trial)
    elif gcp_ver=='std':
        v = general_cross_product(X)
        flag = 0
    elif gcp_ver=='v3':
        v = general_cross_product_v3(X)
        flag = 0
    if flag==0:
        x = np.random.randn(1,d)
        print(X@v)
        print((x@v).item())
        print(np.linalg.det(np.concatenate([x,X],axis=0)).item())

def relu(x):
    return np.maximum(0,x)
def drelu(x):
    return x>=0

def cvx_sample_vectors(training_data_np, max_neurons, arr_select = 'Gaussian', with_bias = True, sketch=True, sdim=50, 
    gcp_ver='v3', max_trial=20, aug_sym = False):
    n, Embedding_Size = np.shape(training_data_np)

    if arr_select == 'Gaussian':
        if with_bias:
            G = np.random.randn(Embedding_Size+1,max_neurons)
        else:
            G = np.random.randn(Embedding_Size,max_neurons)
    elif arr_select == 'Geometric_Algebra':
        if sketch:
            S = get_SJLT_matrix(Embedding_Size,sdim,1)
            training_data_np = training_data_np@S
            d = sdim
        else:
            d = Embedding_Size
            S = np.identity(d)
        if with_bias:
            G = np.zeros([Embedding_Size+1,max_neurons])
            for i in range(max_neurons):
                # sample x_1,...x_{d}
                iter_num = 0 
                while iter_num<max_trial:
                    iter_num+=1
                    index = np.random.choice(n,d,replace=False)
                    x_d = training_data_np[index[-1],:].reshape([1,-1])
                    A = training_data_np[index[:-1],:]-x_d
                    # A = A/np.linalg.norm(A)*10
                    if gcp_ver=='v2':
                        v, flag = general_cross_product_v2(A, max_trial=max_trial)
                    elif gcp_ver=='std':
                        v = general_cross_product(A)
                        flag = 0
                    elif gcp_ver=='v3':
                        v = general_cross_product_v3(A)
                        flag = 0
                    if flag==0:
                        v = v.reshape([1,-1])
                        break
                # print(np.linalg.norm(v))
                if iter_num<max_trial:
                    G[:,i] = np.concatenate([v@S.T,-x_d@v.T],axis=1)/np.linalg.norm(v@S.T)
                else:
                    print('Warning: reach max trial')
                    v = np.random.randn(d).reshape([1,-1])
                    G[:,i] = np.concatenate([v@S.T,-x_d@v.T],axis=1)/np.linalg.norm(v@S.T)
        else:
            G = np.zeros([Embedding_Size,max_neurons])
            for i in range(max_neurons):
                index = np.random.choice(n,d-1,replace=False)
                if gcp_ver=='v2':
                    v = general_cross_product_v2(training_data_np[index,:], max_trial=max_trial)
                elif gcp_ver == 'std':
                    v = general_cross_product(training_data_np[index,:])
                elif gcp_ver == 'std':
                    v = general_cross_product(training_data_np[index,:])
                G[:,i] = v@S.T/np.linalg.norm(v@S.T)
        if aug_sym == True:
            G = np.concatenate([G,-G],axis=1)
    elif arr_select == 'Polished_Gaussian':
        if sketch:
            S = get_SJLT_matrix(Embedding_Size,sdim,1)
            training_data_np = training_data_np@S
            d = sdim
        else:
            d = Embedding_Size
            S = np.identity(d)
        G_aug = np.random.randn(d,max_neurons)
        if with_bias:
            G = np.zeros([Embedding_Size+1,max_neurons])
            for i in range(max_neurons):
                index_x_d = np.random.choice(n,1)
                x_d = training_data_np[index_x_d,:].reshape([1,-1])
                product = np.abs((training_data_np-x_d)@(G_aug[:,i].reshape(-1)-x_d.reshape(-1)))
                index = product.argsort()[1:d] # remove index_x_d from indexing
                # print(index)
                A = training_data_np[index,:]-x_d
#                 print(A.shape)
                if fast_comp:
                    v = general_cross_product_v2(A, max_trial=max_trial).reshape([1,-1])
                else:
                    v = general_cross_product(A).reshape([1,-1])
                G[:,i] = np.concatenate([v@S.T,-x_d@v.T],axis=1)/np.linalg.norm(v@S.T)
        else:
            G = np.zeros([Embedding_Size,max_neurons])
            for i in range(max_neurons):
                product = np.abs(training_data_np@G_aug[:,i])
                index = product.argsort()[1:d]
                if fast_comp:
                    v = general_cross_product_v2(training_data_np[index,:], max_trial=max_trial)
                else:
                    v = general_cross_product(training_data_np[index,:])
                G[:,i] = v@S.T/np.linalg.norm(v@S.T)

    return G

def cvx_solver_mosek(training_data_np,training_labels_np,beta=1e-3,Hidden=50,
                   arr_select = 'Gaussian', with_bias = True, activation='gReLU',
                     reg_p=2,sdim=100,gcp_ver='v3',verbose=False,add_eps=True,eps=1e-8,
                     aug_sym=False,solver='mosek', cp_verbose=False):
    
    n, Embedding_Size = np.shape(training_data_np)

    
    if arr_select == 'GA_enum':
        assert Embedding_Size==2, 'wrong input dimension'
        G = np.zeros([3,n*(n-1)])
        count = 0
        for i in range(n):
            for j in range(i+1,n):
                xi = training_data_np[i]
                xj = training_data_np[j]
                v = (xi-xj)/np.linalg.norm(xi-xj)
                G[:2,count] = v
                G[2,count] = -xj@v
                G[:2,count+1] = -v
                G[2,count+1] = xi@v
                count+=2
        
    else:
        if sdim<=0:
            sketch=False
        else:
            sketch=True
        G = utils.cvx_sample_vectors(training_data_np, Hidden, arr_select=arr_select, with_bias=True,sketch=sketch,
            sdim=sdim,gcp_ver=gcp_ver,max_trial=20,aug_sym=aug_sym)
        if arr_select=='Geometric_Algebra' and add_eps:
            G[-1,:]+=eps
    training_data_np = np.concatenate([training_data_np,np.ones([n,1])],axis=1)
    Embedding_Size += 1

    dmat= drelu(np.matmul(training_data_np,G))

    if activation=='gReLU':
        # Optimal CVX
        m1=dmat.shape[1]
        Uopt1=cp.Variable((Embedding_Size,m1))

        ## Below we use squared loss as a performance metric for binary classification
        yopt1=cp.sum(cp.multiply(dmat,(training_data_np@Uopt1)),axis=1)

        if with_bias:
            regularization = cp.mixed_norm(Uopt1[:-1,:].T,reg_p,1)
        else:
            regularization = cp.mixed_norm(Uopt1.T,reg_p,1)
        cost=cp.sum_squares(training_labels_np-yopt1)+beta*regularization
        constraints=[]
        prob=cp.Problem(cp.Minimize(cost))
        prob.solve(solver=cp.MOSEK,verbose=verbose)
        cvx_opt=prob.value
        if verbose:
            print("Convex program objective value: ",cvx_opt)

        return G, Uopt1._value
    elif activation=='ReLU':
        dmat=(np.unique(dmat,axis=1))
        m1=dmat.shape[1]
        Uopt1=cp.Variable((Embedding_Size,m1))
        Uopt2=cp.Variable((Embedding_Size,m1))

        yopt1=cp.sum(cp.multiply(dmat,(training_data_np@Uopt1)),axis=1)
        yopt2=cp.sum(cp.multiply(dmat,(training_data_np@Uopt2)),axis=1)
        if with_bias:
            regularization = cp.mixed_norm(Uopt1[:-1,:].T,reg_p,1)+cp.mixed_norm(Uopt2[:-1,:].T,reg_p,1)
        else:
            regularization = cp.mixed_norm(Uopt1.T,2,1)+cp.mixed_norm(Uopt1.T,2,1)
        cost=cp.sum_squares(yopt1-yopt2-training_labels_np)+beta*regularization
        constraints=[]
        constraints+=[cp.multiply((2*dmat-np.ones((n,m1))),(training_data_np@Uopt1))>=0]
        constraints+=[cp.multiply((2*dmat-np.ones((n,m1))),(training_data_np@Uopt2))>=0]

        prob=cp.Problem(cp.Minimize(cost),constraints)
        prob.solve(solver=cp.MOSEK,verbose=verbose)
        cvx_opt=prob.value
        
        if verbose:
            print("Convex program objective value: ",cvx_opt)
        U1, U2 = Uopt1._value, Uopt2._value
        return U1, U2
    if activation=='Lasso':
        G = G/np.linalg.norm(G[:-1,:],2,axis=0)
        m1=G.shape[1]
        z =cp.Variable(m1)
        t = cp.Variable(1)
        yopt = relu(training_data_np@G)@z+t
        regularization = cp.norm(z,1)
        cost=cp.sum_squares(yopt-training_labels_np)+beta*regularization
        prob=cp.Problem(cp.Minimize(cost))
        if solver == 'mosek':
            cvx_solver = cp.MOSEK
        elif solver == 'scs':
            cvx_solver = cp.SCS
        prob.solve(solver=cvx_solver,verbose=cp_verbose)
        cvx_opt=prob.value
        
        if verbose:
            print("Convex program objective value: ",cvx_opt)
        z_v, t_v = z._value, t._value
        return G, z_v, t_v

def cvx_solver_evaluate(test_data_np,test_labels_np,params,activation='gReLU', with_bias=True):

    if with_bias:
        n = np.shape(test_data_np)[0]
        test_data_np = np.concatenate([test_data_np,np.ones([n,1])],axis=1)
        
    if activation=='gReLU':
        G, U = params
        preds = np.sum(drelu(test_data_np@G)*(test_data_np@U),axis=1)
    elif activation=='ReLU':
        U1, U2 = params
        preds = np.sum(relu(test_data_np@U1)-relu(test_data_np@U2),axis=1)
    elif activation=='Lasso':
        G, z, t = params
        preds = relu(test_data_np@G)@z+t
    acc = accuracy(preds,test_labels_np)*100
#     print("Convex accuacy:{:.2f}%".format(acc))
    return acc

def set_seed(seed_value=42):
    """Set seed for reproducibility."""
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)

# Create the NNClassifier class
class NNClassifier(nn.Module):
    """2 layer NN Model for Classification Tasks.
    """
    def __init__(self, hidden, D_in = 768*2, add_skip=False):
        """
        @param    bert: a BertModel object
        @param    classifier: a torch.nn.Module classifier
        """
        super(NNClassifier, self).__init__()
        # Specify hidden size of BERT, hidden size of our classifier, and number of labels
        D_in, H, D_out = D_in, hidden, 1

        self.add_skip = add_skip

        # Instantiate a two-layer feed-forward classifier
        self.classifier = nn.Sequential(
            nn.Linear(D_in, H),
            nn.ReLU(),
            nn.Linear(H, D_out)
        )

        if add_skip:
            self.skip_layer = nn.Sequential(
            nn.Linear(D_in, 1),
            nn.Linear(1, D_out)
        )


    def forward(self, input_ids):
        # BY LOADING THE CSV FILE OF THE OUTPUT EMBEDDING FROM BERT
        logits = self.classifier(input_ids)
        if self.add_skip:
            logits+= self.skip_layer(input_ids)

        logits = logits.squeeze()
        return logits

def evaluate(model, val_dataloader, device):
    loss_fn = nn.MSELoss()
    """Measure model's performance on the validation set."""
    model.eval()  # Evaluation mode
    val_accuracy_list = []
    val_loss_list = []
    for batch in val_dataloader:
        b_input_ids, b_labels = tuple(t.to(device) for t in batch)

        with torch.no_grad():
            logits = model(b_input_ids).squeeze()
        loss = loss_fn(logits, b_labels)
        val_loss_list.append(loss.item())
        preds = torch.sign(logits)
#         preds = torch.argmax(logits, dim=1).flatten()
        accuracy = (preds == b_labels).cpu().numpy().mean() * 100
        val_accuracy_list.append(accuracy)
    val_loss = np.mean(val_loss_list)
    val_accuracy = np.mean(val_accuracy_list)
    return val_loss, val_accuracy

def transform_U(U, X):
    """
    For each row in U, find the (d-1) rows in X with the lowest inner-product magnitude,
    then replace the row in U with the normal of the hyperplane through the origin and those points.
    """
    d = U.shape[1]  # Dimensions
    transformed_U = np.zeros_like(U)
    #for i, u_row in enumerate(U):
    for i, u_row in tqdm(enumerate(U), total=U.shape[0], desc="Processing neurons"):
        u_row_normalized = u_row# / np.linalg.norm(u_row)
        inner_products = [np.abs(np.dot(u_row_normalized, x_row)) for x_row in X]#/ np.linalg.norm(x_row)
        #add debug point
        # Get indices of the (d-1) smallest inner products
        smallest_indices = np.argsort(inner_products)[:d-1]
        
        # Get the (d-1) rows from X
        selected_rows = X[smallest_indices]
        
        # Find normal of hyperplane passing through the origin and these rows
        normal = find_normal_of_hyperplane(selected_rows)
        
        # Replace the current row of U with this normal
        transformed_U[i] = normal
    return transformed_U

def find_normal_of_hyperplane(points):
    """
    Find the normal vector of the hyperplane passing through the origin and the given (d-1) points.
    Assumes points is a (d-1)xN matrix, where each row is a point in d-dimensional space.
    """
    method = 'eigh'
    # Use SVD to find the null space of the matrix formed by points
    if method == 'svd':
        u, s, vh = np.linalg.svd(points, full_matrices=True)
        #d = u.shape[1]
        # MP: change this to rank k svd not the full svd.
        # problem: smallest singular value is zero for MNIST.
        # The normal vector is the last column of vh, corresponding to the smallest singular value
        normal = vh[-1]
    else:
        G = np.dot(points.T, points)
        # Compute the eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eigh(G)
        # The smallest eigenvalue's corresponding eigenvector
        normal = eigenvectors[:, 0]
    normalized_normal = normal / np.linalg.norm(normal)
    return normalized_normal

def train(model, optimizer, scheduler, train_dataloader, val_dataloader=None, epochs=4, evaluation=False, freq_batch=5, 
    polish=False,polish_freq=1, sdim=100):
    """Train the BertClassifier model."""
    train_loss_set = []
    val_loss_set = []
    train_accuracy_set = []
    val_accuracy_set = []
    cumulative_time = []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    total_time_elapsed = 0
    print("Start training...\n")
    t0_start = time.time()
    
    loss_fn = nn.MSELoss()

    for epoch_i in range(epochs):

        total_loss, total_correct, total_preds, total_num = 0, 0, 0, 0
        model.train()
        data_list = []
        labels_list = []
        for step, batch in enumerate(train_dataloader):
            b_input_ids, b_labels = tuple(t.to(device) for t in batch)
            data_list.append(b_input_ids.cpu())
            labels_list.append(b_labels.cpu().numpy())


            model.zero_grad()
            logits = model(b_input_ids)
            loss = loss_fn(logits, b_labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()*int(b_labels.size(0))
            total_num += b_labels.size(0)

            preds = torch.sign(logits)
            total_correct += (preds == b_labels).sum().item()
            total_preds += b_labels.size(0)

            # Print training results for every freq_batch
            if (step % freq_batch == 0 and step != 0) or (step == len(train_dataloader) - 1):

                #print(f"Epoch: {epoch_i + 1}, Batch: {step}, Batch Loss: {batch_loss:.6f}, "
                      #f"Batch Accuracy: {batch_accuracy:.2f}%, Time: {time_elapsed:.2f}s")

                avg_train_loss = total_loss / total_num
                total_loss = 0
                total_num = 0
                train_accuracy = (total_correct / total_preds) * 100
                total_correct = 0
                total_preds = 0
                train_loss_set.append(avg_train_loss)
                train_accuracy_set.append(train_accuracy)

                time_elapsed = time.time() - t0_start
                cumulative_time.append(time_elapsed)
                '''
                print(f"Epoch: {epoch_i + 1}, Batch: {step}, Train Loss: {avg_train_loss:.6f}, "
                      f"Train Accuracy: {train_accuracy:.2f}%, Time: {time_elapsed:.2f}s")
                '''

                if evaluation:
                    val_loss, val_accuracy = evaluate(model, val_dataloader, device)
                    val_loss_set.append(val_loss)
                    val_accuracy_set.append(val_accuracy)
        if polish:
            if (epoch_i)%polish_freq==0:
                Uorg = model.classifier[0].weight.detach().numpy().T
                bias = model.classifier[0].bias.detach().numpy()
                Xdata_org = torch.cat(data_list, dim=0)
                if sdim>0:
                    onestrain = np.ones((Xdata_org.shape[0], 1))
                    S = get_SJLT_matrix(Xdata_org.shape[1],sdim,1)
                    Xdata_np = Xdata_org.detach().numpy()
                    Xdata_np_sketch = Xdata_np@S
                    Xdata = np.hstack((Xdata_np_sketch, onestrain))
                    ydata = np.concatenate(labels_list)
                    Uorg = np.vstack((S.T@Uorg, bias))
                    new_U = transform_U(np.copy(Uorg).T,Xdata).T
                else:
                    onestrain = np.ones((Xdata_org.shape[0], 1))
                    Uorg = np.vstack((Uorg, bias))
                    Xdata = np.hstack((Xdata_org.detach().numpy(), onestrain))
                    ydata = np.concatenate(labels_list)
                    new_U = transform_U(np.copy(Uorg).T,Xdata).T

                beta = 1e-3
                A = relu(Xdata@new_U)
                AtA = A.T@A
                w_ls = np.linalg.solve(AtA + beta*np.identity(AtA.shape[0]),A.T@ydata)

                for col in range(new_U.shape[1]):
                    if np.abs(w_ls[col]) > 1e-10:
                        scale = np.sqrt(np.linalg.norm(new_U[:,col])/np.abs(w_ls[col]))
                        new_U[:,col] = new_U[:,col]/scale
                        w_ls[col] = w_ls[col]*scale 
                    else:
                        new_U[:,col] = np.zeros_like(new_U[:,col])
                        w_ls[col] = 0
                        print('dropped neuron')
                      
                if sdim>0:
                    model.classifier[0].weight.data = torch.tensor(new_U[:-1,:].T@S.T,requires_grad=True)
                else:
                    model.classifier[0].weight.data = torch.tensor(new_U[:-1,:].T,requires_grad=True)
                model.classifier[0].bias.data = torch.tensor(new_U[-1,:],requires_grad=True)
                model.classifier[2].weight.data = torch.tensor(w_ls,requires_grad=True).float().reshape([1,-1])
                model.to(device)
        if evaluation:
            print('Epoch: {} Train_acc: {} Test_acc: {}'.format(epoch_i+1, train_accuracy, val_accuracy))

    return cumulative_time, train_loss_set, val_loss_set, train_accuracy_set, val_accuracy_set

def initialize_model(hidden, epochs=4, lr=1e-5, beta = 1e-2, add_skip=False, D_in=768,train_dataloader=None):
    """Initialize the Bert Classifier, the optimizer and the learning rate scheduler.
    """
    # Instantiate Bert Classifier
    bert_classifier = NNClassifier(hidden, D_in=D_in, add_skip = add_skip)

    # Tell PyTorch to run the model on GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #bert_classifier = bert_classifier.to(device)
    bert_classifier.to(device)


    # Create the optimizer
    optimizer = AdamW(bert_classifier.parameters(),
                      lr=lr,    # Default learning rate
                      eps=1e-8,    # Default epsilon value
                      weight_decay = beta #weight decay
                      )

    # Total number of training steps
    total_steps = len(train_dataloader) * epochs

    # Set up the learning rate scheduler
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0, # Default value
                                                num_training_steps=total_steps)
    return bert_classifier, optimizer, scheduler

def accuracy(logits, y):
        return np.sum((np.sign(logits) == y)) / len(y)

def scnn_inner(training_data_np,training_labels_np,test_data_np, test_labels_np,Hidden,method,c,beta=1e-3,
               verbose=True,tol=1e-7,add_skip=False,sdim=50,activation='grelu',gcp_ver='v3',solver='std',
               add_eps=False, eps=1e-8):
    if sdim<=0:
        sketch=False
    else:
        sketch=True
    G = utils.cvx_sample_vectors(training_data_np, Hidden, arr_select=method, with_bias=True,sketch=sketch,sdim=sdim,gcp_ver=gcp_ver,max_trial=20)

    G_bias = G[-1,:].reshape(-1)
    G = G[:-1,:]

    if add_eps:
        G_bias = G_bias + eps

    if add_skip:
        G_bias = np.concatenate([G_bias,np.array([1])],axis=0)
        d, m = G.shape
        G = np.concatenate([G,np.zeros(d).reshape([-1,1])],axis=1)
    
    if activation == 'grelu':
        model = ConvexGatedReLU(G, c=c, bias=True, G_bias=G_bias)
        if solver=='cvxpy':
            solver = CVXPYSolver(model,'mosek')
        else:
            solver = RFISTA(model,tol=tol)
    elif activation == 'relu':
        model = ConvexReLU(G, c=1, bias=True, G_bias=G_bias)
        solver = AL(model,tol=tol,max_primal_iters=100,constraint_tol=tol)
    beta = beta
    metrics = Metrics(train_accuracy=True, train_mse = True, test_accuracy = True, test_mse = True)
    cvx_model, metrics = optimize_model(model, solver, metrics, training_data_np, training_labels_np, test_data_np, test_labels_np, regularizer = NeuronGL1(beta), verbose=verbose)

    return cvx_model, metrics

def eval_model(training_data_np,training_labels_np,test_data_np, test_labels_np,cvx_model):
    preds = cvx_model(test_data_np)
    open_AI_test_accuracy =  accuracy(np.squeeze(preds), np.squeeze(test_labels_np))

    preds = cvx_model(training_data_np)
    open_AI_train_accuracy =  accuracy(np.squeeze(preds), np.squeeze(training_labels_np))

    return open_AI_train_accuracy, open_AI_test_accuracy