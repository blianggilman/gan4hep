

import pandas as pd
import numpy as np
# from gan4hep.preprocess import shuffle

def convert_g4_data(filename):

    data = None
    Nsec = []
    with open(filename, 'r') as f:
        for line in f: 
            # Each line has 1 col Iron info, 1 col copper info, 5 cols of primary particle, 5 cols for each secondary particle
            data = line                                                                                                                                                                      
            items = data.split()                                                                                                                                                      
            n_particles = (len(items[2:]) // 5) - 1 #disregarding the primary particle (first 5 columns)
            Nsec.append(n_particles)

        
        N = 400
        # df = pd.read_csv(filename, header=None, sep=' ', usecols=[1,2,3,4]) #no material information
        # df.insert(4, "5", Nsec, True)
        df = pd.read_csv(filename, header=None, sep=' ', usecols=[0,1,3,4,5,6])
        df.insert(6, "7", Nsec, True)
        print(df.shape)
        
        print(df)

        # df.to_csv('/global/homes/b/blianggi/290e/MCGenerators/G4/HadronicInteractions/build/abr_kaon_minus_Cu_30GeV.csv', sep=' ', index=False)

        data = df.to_numpy().astype(np.float32)
        # data = data[:,-1]/100
        print("data.shape", data.shape)
        print(data[:,1])
        training_size = int(data.shape[0]*0.8)

        train_4vec, test_4vec = data[:training_size, :-1], data[training_size:, :-1]
        train_nsec, test_nsec = data[:training_size, -1], data[training_size:, -1]

        train_nsec = train_nsec[..., None]
        test_nsec = test_nsec[..., None]

        # print("max", np.max(train_4vec))
        # print("max", np.max(train_nsec))
        # print("min", np.min(test_nsec))

        # Normalize data and scale it
        # train_4vec /=np.max(train_4vec, axis=0)
        # test_4vec /=np.max(test_4vec, axis=0)
        # print("helllooooooo", train_4vec[:,2:])
        train_just_4vec = train_4vec[:,2:]
        test_just_4vec = test_4vec[:,2:]
        train_just_4vec /= np.max(train_just_4vec, axis=0)
        test_just_4vec /= np.max(test_just_4vec, axis=0)
        train_4vec = np.concatenate((train_4vec[:,:2],train_just_4vec), axis=1)
        test_4vec = np.concatenate((test_4vec[:,:2],test_just_4vec), axis=1)
        train_nsec = ((train_nsec - np.min(train_nsec)) / (np.max(train_nsec) - np.min(train_nsec))) * 2 - 1
        test_nsec = ((test_nsec - np.min(test_nsec)) / (np.max(test_nsec) - np.min(test_nsec))) * 2 - 1

        print("here!!", test_nsec[:10])
        print(train_4vec[0:10])

        xlabels = ['num_secondary']
        print(train_4vec.shape, test_4vec.shape, train_nsec.shape, test_nsec.shape)

        return (train_4vec, test_4vec, train_nsec, test_nsec, xlabels)


if __name__ == '__main__':
    convert_g4_data('/global/homes/b/blianggi/290e/MCGenerators/G4/HadronicInteractions/build/kaon_minus_Cu_30GeV_millionevts.csv')
