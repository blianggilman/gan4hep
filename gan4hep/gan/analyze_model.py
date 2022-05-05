import importlib
from operator import truth
import os
import py_compile
from tabnanny import filename_only
import time
import yaml

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import tensorflow as tf
import pandas as pd


def get_sec_particles(filename):
    Nsec = []
    with open(filename, 'r') as f:
        for line in f:
            items = line.split()
            n_particles = len(items[2:]) // 5
            Nsec.append(n_particles)
    return Nsec

def strip_newline(filename):
    file=open(filename,'r')
    content = file.readlines()
    arr = []
    for item in content:
        arr.append(float(item.split(", ")[0].split("\n")[0]))
    arr = np.array(arr)
    file.close()
    return arr

def inverse_transform(gen, orig_file):
    """
    Transform generated data (between -1, 1) back to original space.
    """
    y_orig = get_sec_particles(orig_file)
    y_max, y_min = np.max(y_orig), np.min(y_orig)
    output = ((gen + 1) / 2 * (y_max - y_min)) + y_min
    return output.astype(int)


def rescale_plots():
    """
    Turn generated plots from (-1,1) back to original scale of secondary particles
    """


def save_best_epoch_info(img_dir, best_epoch, epochs, KE, num_input, model, datasets):
    print(img_dir) #will print "log_training/img"
    if num_input == "1m":
        placeholder = img_dir
        img_dir = placeholder + '/1mevts' # os.path.join does not work here bc of forward slash
    file=open("{}/best_epoch_per_{}_epochs_{}.csv".format(img_dir, epochs, KE),'w')
    predictions = []
    for data in datasets:
        test_input, _ = data
        predictions.append(model(test_input, training=False))
    predictions = tf.concat(predictions, axis=0).numpy()

    num_variables = predictions.shape[1]
    print("ARE THESE TWO EQUAL?", num_variables-1, best_epoch)

    # file.write(predictions[:, best_epoch])
    np.savetxt(file, predictions[:, 0], delimiter=" ")
    file.close()


def plot_best_to_scale(img_dir, epochs, KE, num_input):

    # for KE_curr in ['01', '05', '1', '5', '10', '15', '20', '30']: # GeV
    for i in range(1):
        
        " --- make plot --- "

        # y_truth = np.loadtxt(orig_file, usecols=(6))
        filename_truth = '/global/homes/b/blianggi/290e/MCGenerators/G4/HadronicInteractions/build/kaon_minus_Cu_{}_{}evts.csv'.format(KE, num_input)
        y_truth = get_sec_particles(filename_truth)
        filename_pred = '{}best_epoch_per_{}_epochs_{}.csv'.format(img_dir, epochs, KE)
        predictions = np.loadtxt(filename_pred)
        print(predictions)


        plt.figure()
        bins = np.arange(0,100,1)

        plt.hist(y_truth, density=True, histtype='step', bins=bins, label='Truth', zorder=1)
        plt.hist(inverse_transform(predictions, filename_truth), density=True, histtype='step', bins=bins, label='Generated', zorder=1)

        plt.xlabel('Number of secondary particles')
        plt.title('Best Epoch, {}'.format(KE))
        plt.legend()

        plt.savefig(img_dir + 'bestepoch_{}.png'.format(KE), dpi=150)
        # print(f'Comparison plot saved to {FIG_DIR}')


def plot_best__randGeV_to_scale(img_dir, epochs, KE, num_input):        
        " --- make plot --- "

        # y_truth = np.loadtxt(orig_file, usecols=(6))
        filename_truth = '/global/homes/b/blianggi/290e/MCGenerators/G4/HadronicInteractions/build/kaon_minus_Cu_{}_{}evts.csv'.format(KE, num_input)
        y_truth = get_sec_particles(filename_truth)
        filename_pred = '{}best_epoch_per_{}_epochs_{}.csv'.format(img_dir, epochs, KE)
        predictions = np.loadtxt(filename_pred)
        print(predictions)


        plt.figure()
        bins = np.arange(0,100,1)

        plt.hist(y_truth, density=True, histtype='step', bins=bins, label='Truth (full)', zorder=1)
        plt.hist(y_truth[:int(0.2*len(y_truth))], density=True, histtype='step', bins=bins, label='Truth (20%)', zorder=1)
        plt.hist(inverse_transform(predictions, filename_truth), density=True, histtype='step', bins=bins, label='Generated', zorder=1)

        plt.xlabel('Number of secondary particles')
        plt.title('Best Epoch, {}'.format(KE))
        plt.legend()

        plt.savefig(img_dir + 'bestepoch_{}.png'.format(KE), dpi=150)
        # print(f'Comparison plot saved to {FIG_DIR}')



def plot_losses(img_dir, epochs, KE):
    
    #read information
    gen_loss_over_time = strip_newline("{}gen_loss_data_per_{}_epochs_{}.txt".format(img_dir, epochs, KE))
    dis_loss_over_time = strip_newline("{}disc_loss_data_per_{}_epochs_{}.txt".format(img_dir, epochs, KE))

    #assign epochs array
    epochs_arr = np.arange(1,len(gen_loss_over_time)+1)

    print("length of gen loss", len(gen_loss_over_time)) 
    print(len(epochs_arr))

    #plot
    plt.figure(0)
    plt.plot(epochs_arr, gen_loss_over_time, label='Generator', color='red')
    plt.plot(epochs_arr, dis_loss_over_time, label='Discriminator', color='blue')
    # plt.ylim(bottom=0)
    # plt.xlim([0,1000])
    plt.title("Loss per epoch, {}".format(KE))
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    
    outname_loss = os.path.join(img_dir, "loss_per_{}_epochs_{}".format(epochs, KE))
    plt.savefig(outname_loss)

def plot_losses_separate(img_dir, epochs, KE):
    
    #read information
    gen_loss_over_time = strip_newline("{}gen_loss_data_per_{}_epochs_{}.txt".format(img_dir, epochs, KE))
    dis_loss_over_time = strip_newline("{}disc_loss_data_per_{}_epochs_{}.txt".format(img_dir, epochs, KE))

    #assign epochs array
    epochs_arr = np.arange(1,len(gen_loss_over_time)+1)

    print("length of gen loss", len(gen_loss_over_time)) 
    print(len(epochs_arr))

    #plot
    plt.figure(0)
    plt.subplot(2,1,2)
    plt.plot(epochs_arr, gen_loss_over_time, label='Generator', color='red')
    plt.ylabel("Generator Loss")
    plt.xlabel("Epoch")
    plt.subplot(2,1,1)
    plt.plot(epochs_arr, dis_loss_over_time, label='Discriminator', color='blue')
    # plt.ylim(bottom=0)
    # plt.xlim([0,1000])
    plt.title("Loss per epoch, {}".format(KE))
    plt.ylabel("Discriminator Loss")
    # plt.legend()
    
    outname_loss = os.path.join(img_dir, "sep_loss_per_{}_epochs_{}".format(epochs, KE))
    plt.savefig(outname_loss)


def plot_w_dist(img_dir, epochs, KE):

    #read information
    wdist_over_time = strip_newline("{}wdist_data_per_{}_epochs_{}.txt".format(img_dir, epochs, KE))
    lowest_wdist_over_time = strip_newline("{}lowest_wdist_data_per_{}_epochs_{}.txt".format(img_dir, epochs, KE))
    
    # lowest_wdist_file=open("{}lowest_wdist_data_per_{}_epochs_{}.txt".format(img_dir, epochs, KE),'r')
    # content = lowest_wdist_file.readlines()
    # content = np.array(content)
    # """ Looks like 
    # ['0.016721254214644432, 0\n' '0.016721254214644432, 0\n'
    # '0.016721254214644432, 0\n' '0.016721254214644432, 0\n'
    # '0.016721254214644432, 0\n' '0.016721254214644432, 0\n'
    # '0.016721254214644432, 0\n' '0.016721254214644432, 0']
    # """# print(content.shape) #(has shape (n,))
    # lowest_epoch = []
    # for item in content:
    #     lowest_epoch.append(item.split(", ")[1].split("\n")[0])
    # lowest_epoch = np.array(lowest_epoch)
    # lowest_wdist_file.close()

    #assign epochs array
    epochs_arr = np.arange(1,len(lowest_wdist_over_time)+1)
    # print("epochs", epochs_arr)
    # print(wdist_over_time)
    # print(lowest_wdist_over_time)

    #plot
    _, ax = plt.subplots()
    ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%1.2f'))
    plt.plot(epochs_arr, wdist_over_time, label='Wasserstein distance', color='blue')
    plt.plot(epochs_arr, lowest_wdist_over_time, label='Lowest current W distance', color='red')
    # plt.plot(epochs_arr, wdist_over_time[:len(lowest_wdist_over_time)], label='Wasserstein distance', color='blue')
    # plt.plot(epochs_arr, lowest_wdist_over_time, label='Lowest current W distance', color='red')
    # plt.ylim(bottom=0)
    # plt.xlim([0,1000])
    plt.legend()
    plt.title("Wasserstein distance per epoch, {}".format(KE))
    plt.xlabel("Epoch")
    plt.ylabel("Wasserstein distance")
    outname_wdist = os.path.join(img_dir, "wdist_per_{}_epochs_{}".format(epochs, KE))
    plt.savefig(outname_wdist)
    plt.close('all')




def analyze_model(log_dir, epochs, KE, num_input):
    if num_input == "1m":
        img_dir = os.path.join(log_dir, 'img/', '1mevts/')
    else:
        img_dir = os.path.join(log_dir, 'img/')

    plot_losses(img_dir, epochs, KE)
    plot_losses_separate(img_dir, epochs, KE)
    plot_w_dist(img_dir, epochs, KE)
    if "randGeV" in KE and len(KE)>7:
        plot_best__randGeV_to_scale(img_dir, epochs, KE, num_input)
    else:
        plot_best_to_scale(img_dir, epochs, KE, num_input)



if __name__=='__main__':

    import argparse                                                                                                                                                                                       
    parser = argparse.ArgumentParser(description='Analyze model')                                                                                                                                         
    add_arg = parser.add_argument
    add_arg("KE", help='KE being used including MeV/GeV', default=None)
    add_arg("--num-input", help='Number of input events (in filename)', default='1m')
    add_arg("--epochs", help='Number of epochs (in filename)', default=1000)
    args = parser.parse_args()

    #test run
    log_dir = 'log_training/'
    # img_dir = 'log_training/img/'
    epochs = args.epochs #24414
    KE=args.KE
    num_input=args.num_input
    print(num_input)

    analyze_model(log_dir, epochs, KE, num_input)


    # add_arg("--log_dir", help='name of log folder', default=None)
    # add_arg("--ngen", default=100000, type=int, help='if not None, generate sample!')
    # add_arg("--elow", default='05', type=str)
    # add_arg("--ehigh", default='15', type=str)
    # add_arg("--filename", default=None, type=str)
    # add_arg("--epochs", default=100, type=int)

    # args = parser.parse_args()

    # LOG_DIR = args.log_dir
    # GEN_DIR = LOG_DIR + args.log_dir + '/generator'
    # FIG_DIR = LOG_DIR + args.log_dir
    # INPUT_DIR = '~/290e/MCGenerators/G4/HadronicInteractions/build/'


    #1. run cgan.py - do we import cgan?
    # python cgan.py filename --epochs

    #2. call invert method to invert data and graph it
        # can this be done in utils_plot.py?
    
    #3. save data every epoch so it can graphed?
        # losses, W distance, best W distance -> move into the img folders
