import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from model import *
import sys
sys.path.append('../../')
from lib.utils import LpLoss, Logger, Config

def main(**kwargs):
    opt = Config()
    for _k, _v in kwargs.items():
        setattr(opt, _k, _v)

    ntrain, nvalid, ntest = opt.ntrain, opt.nvalid, opt.ntest
    lambda_in, lambda_out = opt.lambda_in, opt.lambda_out
    nbasis_in = opt.nbasis_in
    nbasis_out = opt.nbasis_out
    basis_name = opt.basis_name.lower()
    batch_size = opt.batch_size
    learning_rate = opt.learning_rate
    epochs = opt.epochs
    device = opt.device
    activation = opt.activation
    base_in_hidden = opt.base_in_hidden
    base_out_hidden = opt.base_out_hidden
    middle_hidden = opt.middle_hidden
    model_name = opt.model_name.lower()
    if basis_name == 'sin':
        subpath = os.path.join(basis_name + '_' + str(opt.nbasis))
    elif basis_name == 'grf':
        subpath = os.path.join(basis_name + '_' + str(length_scale))
    else:
        print(basis_name + ' is not included now!')
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')    
    if lambda_in > 0.0 or lambda_out > 0.0:
        model_path = os.path.join('checkpoints', model_name + '_' + subpath + '.pth')
        logger = Logger(subpath=model_name + '_' + subpath)
    else:
        model_path = os.path.join('checkpoints', model_name + '_' + 'NOREG' + '_' + subpath + '.pth')
        logger = Logger(subpath=model_name + '_' + 'NOREG' + '_' + subpath)

    x = np.load(os.path.join('datasets', subpath, 'in_f.npy'))
    y = np.load(os.path.join('datasets', subpath, 'out_f.npy'))
    grid_in = np.load(os.path.join('datasets', subpath, 'grid_in.npy'))
    grid_out = np.load(os.path.join('datasets', subpath, 'grid_out.npy'))
    print('x_shape | y_shape | grid_in_shape | grid_out_shape: ', \
        x.shape, y.shape, grid_in.shape, grid_out.shape)
    
    x_train = x[:ntrain, ::opt.sub]
    x_valid = x[ntrain:ntrain+nvalid, ::opt.sub]
    x_test = x[ntrain+nvalid:ntrain+nvalid+ntest, ::opt.sub]
    y_train = y[:ntrain, ::opt.sub, ::opt.sub]
    y_valid = y[ntrain:ntrain+nvalid, ::opt.sub, ::opt.sub]
    y_test = y[ntrain+nvalid:ntrain+nvalid+ntest, ::opt.sub, ::opt.sub]
    grid_in = grid_in[::opt.sub]
    grid_out = grid_out[::opt.sub, ::opt.sub, :]
    J1_out, J2_out = y_train.shape[1], y_train.shape[2]
    print('x_train_shape | y_train_shape | grid_in_shape | grid_out_shape: ', \
        x_train.shape, y_train.shape, grid_in.shape, grid_out.shape)
    x_train = torch.from_numpy(x_train).float()
    x_valid = torch.from_numpy(x_valid).float()
    x_test = torch.from_numpy(x_test).float()
    y_train = torch.from_numpy(y_train).float()
    y_valid = torch.from_numpy(y_valid).float()
    y_test = torch.from_numpy(y_test).float()
    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_valid, y_valid), batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test), batch_size=batch_size, shuffle=False)

    model = BasisONet(n_base_in=nbasis_in, base_in_hidden=base_in_hidden, \
            middle_hidden=middle_hidden, n_base_out=nbasis_out, base_out_hidden=base_out_hidden, \
            grid_in=grid_in, grid_out=grid_out, device=device, activation=activation)
    logger.log_string('nbasis_in:{:.0f}\tnbasis_out:{:.0f}'.format(nbasis_in, nbasis_out))
    logger.log_string('lambda_in:{:.0f}\tlambda_out:{:.0f}'.format(lambda_in, lambda_out))

    model = model.to(device)
    mse_loss = nn.MSELoss()
    l2_rel_loss = LpLoss(size_average=False)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)
    valid_loss_max = 10000
    for ep in range(epochs):
        model.train()
        t1 = time.time()
        train_loss_total = 0
        train_l2_loss_operator = 0
        train_l2_loss_autoencoder_in = 0
        train_l2_loss_autoencoder_out = 0
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            out, aec_in, aec_out = model(x, y)
            loss_l2_operator = l2_rel_loss(out, y.reshape(-1, J1_out*J2_out))
            loss_l2_autoencoder_in = l2_rel_loss(aec_in, x)
            loss_l2_autoencoder_out = l2_rel_loss(aec_out, y.reshape(-1, J1_out*J2_out))
            loss_total = loss_l2_operator + lambda_in * loss_l2_autoencoder_in + lambda_out * loss_l2_autoencoder_out
            optimizer.zero_grad()
            loss_total.backward()
            optimizer.step()

            train_loss_total += loss_total.item()
            train_l2_loss_operator += loss_l2_operator.item()
            train_l2_loss_autoencoder_in += loss_l2_autoencoder_in.item()
            train_l2_loss_autoencoder_out += loss_l2_autoencoder_out.item()

        model.eval()
        valid_loss_total = 0
        valid_l2_loss_operator = 0
        valid_l2_loss_autoencoder_in = 0
        valid_l2_loss_autoencoder_out = 0
        with torch.no_grad():
            for x, y in valid_loader:
                x = x.to(device)
                y = y.to(device)
                out, aec_in, aec_out = model(x, y)
                loss_l2_operator = l2_rel_loss(out, y.reshape(-1, J1_out*J2_out))
                loss_l2_autoencoder_in = l2_rel_loss(aec_in, x)
                loss_l2_autoencoder_out = l2_rel_loss(aec_out, y.reshape(-1, J1_out*J2_out))
                loss_total = loss_l2_operator + lambda_in * loss_l2_autoencoder_in + lambda_out * loss_l2_autoencoder_out

                valid_loss_total += loss_total.item()
                valid_l2_loss_operator += loss_l2_operator.item()
                valid_l2_loss_autoencoder_in += loss_l2_autoencoder_in.item()
                valid_l2_loss_autoencoder_out += loss_l2_autoencoder_out.item()

        train_loss_total /= ntrain
        train_l2_loss_operator /= ntrain
        train_l2_loss_autoencoder_in /= ntrain
        train_l2_loss_autoencoder_out /= ntrain
        valid_loss_total /= nvalid
        valid_l2_loss_operator /= nvalid
        valid_l2_loss_autoencoder_in /= nvalid
        valid_l2_loss_autoencoder_out /= nvalid
        t2 = time.time()

        logger.log_string('ep:{:.0f} | lr:{:.5f} | time:{:.2f} | train_l2_total:{:.6f} | train_l2_op:{:.6f} | train_l2_aec_in:{:6f} | train_l2_aec_out:{:6f} | valid_l2_total:{:.6f} | valid_l2_op:{:.6f} | valid_l2_aec_in:{:.6f} | valid_l2_aec_out:{:.6f}'.format(ep,\
            optimizer.state_dict()['param_groups'][0]['lr'], t2-t1, train_loss_total, train_l2_loss_operator, train_l2_loss_autoencoder_in, train_l2_loss_autoencoder_out,\
                        valid_loss_total, valid_l2_loss_operator, valid_l2_loss_autoencoder_in, valid_l2_loss_autoencoder_out))
        if valid_l2_loss_operator < valid_loss_max:
            torch.save({'model_state': model.state_dict(), 'optimizer_state': optimizer.state_dict()}, model_path)
            valid_loss_max = valid_l2_loss_operator
            logger.log_string('find better model')

    model.load_params_from_file(model_path)
    model.eval()
    test_mse_loss_operator = 0
    test_l2_loss_operator = 0
    test_l2_loss_autoencoder_in = 0
    test_l2_loss_autoencoder_out = 0
    test_record = []
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            y = y.to(device)
            out, aec_in, aec_out = model(x, y)
            test_record.append(out)
            loss_mse_operator = mse_loss(out, y.reshape(-1, J1_out*J2_out)) * x.shape[0]
            loss_l2_operator = l2_rel_loss(out, y.reshape(-1, J1_out*J2_out))
            loss_l2_autoencoder_in = l2_rel_loss(aec_in, x)
            loss_l2_autoencoder_out = l2_rel_loss(aec_out, y.reshape(-1, J1_out*J2_out))

            test_l2_loss_operator += loss_l2_operator.item()
            test_mse_loss_operator += loss_mse_operator.item()
            test_l2_loss_autoencoder_in += loss_l2_autoencoder_in.item()
            test_l2_loss_autoencoder_out += loss_l2_autoencoder_out.item()

    test_mse_loss_operator /= ntest
    test_l2_loss_operator /= ntest
    test_l2_loss_autoencoder_in /= ntest
    test_l2_loss_autoencoder_out /= ntest

    logger.log_string('test_mse_op:{:.8f}\ttest_l2_op:{:.6f}\ttest_l2_aec_in:{:.6f}\ttest_l2_aec_out:{:.6f}'.format(test_mse_loss_operator, \
        test_l2_loss_operator, test_l2_loss_autoencoder_in, test_l2_loss_autoencoder_out))

if __name__ == "__main__":

    base_in_hidden = [512, 512, 512, 512, 512]
    base_out_hidden = [512, 512, 512, 512, 512]
    middle_hidden = [512, 512, 512]
    nbasis = None
    nbasis_in = 8
    nbasis_out = 35
    length_scale = 0.2
    main(model_name='BasisONet', sub = 1, \
            ntrain=1000, nvalid=1000, ntest=1000, \
            base_in_hidden=base_in_hidden, base_out_hidden=base_out_hidden, middle_hidden=middle_hidden, \
            nbasis=nbasis, nbasis_in=nbasis_in, nbasis_out=nbasis_out, \
            length_scale=length_scale, \
            lambda_in=1.0, lambda_out=1.0, \
            batch_size=100, epochs=20000, activation=F.gelu)