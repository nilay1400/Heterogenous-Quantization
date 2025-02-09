import torch
import random
import math
import struct

class DMR():
    def __init__(self, x):
        self.weight = x
        # print(self.weight)
        dense_tensor = self.weight.to_dense()
        self.x_size, self.z_size, self.y_size, _ = dense_tensor.shape
        # Unfold the values for the specific key and create a list of tensors
        unfolded_list = list(dense_tensor)
        unfolded_tensor = torch.cat(unfolded_list)
        # Select a random index from the unfolded list
        self.flattened = unfolded_tensor.flatten()
        # flatten_numpy = self.flattened.detach().numpy()
        # param = flatten_numpy.size
        # Sufficient_number_of_faults = round((31 * param) / (1 + ((31 * param) - 1) * (0.00010412328)))
        # Sufficient_number_of_faults = math.floor(Sufficient_number_of_faults * 0.1)
        # Sufficient_number_of_faults = 1
        # fault_injection_accuracy = 0
        # start_time = time.time()
        # for i in range(Sufficient_number_of_faults):

    def set(self):
        flattened_copy = self.flattened.clone()
        for iindex in range(self.flattened.shape[0]):
            # print(flattened_copy[iindex])
            # set_weight_f = self.set(flattened_copy[iindex])
            
            int_bin = format(flattened_copy[iindex], '08b')
            # print("..........s..........")
            # print(flattened_copy[iindex], int_bin)
            int_bin_list = list(int_bin)
            # print(int_bin_list)
            int_bin_list[4] = int_bin_list[5]
            int_bin_list[3] = int_bin_list[5]       
            # print(int_bin_list)            
            new_int_bin = ''.join(int_bin_list)
            # print(new_int_bin)
            set_weight_f = int(new_int_bin, 2)
            set_weight = torch.tensor(set_weight_f)
            # print(set_weight_f, new_int_bin)
            # print("*********s**********")
            flattened_copy[iindex] = set_weight
        modified_weights = flattened_copy.reshape(self.x_size, self.z_size, self.y_size, self.y_size)
        self.updated_weights = modified_weights.to(self.weight.dtype)
        return self.updated_weights

    def correct(self, int_value):
        int_bin = format(int_value, '08b')
        # print(".................")
        # print(int_value, int_bin)
        int_bin_list = list(int_bin)
        # print(int_bin_list)
        if int_bin_list[0] == '1' or int_bin_list[1] == '1' or int_bin_list[2] == '1':
            int_bin_list[0] = '0'
            int_bin_list[1] = '0'
            int_bin_list[2] = '0'  
        if int_bin_list[4] != int_bin_list[5] or int_bin_list[3] != int_bin_list[5]:
            # print("no")
            n_zero = 0
            n_one = 0
            if int_bin_list[5] =='0': n_zero += 1
            else: n_one += 1
            if int_bin_list[4] =='0': n_zero += 1
            else: n_one += 1
            if int_bin_list[3] =='0': n_zero += 1
            else: n_one += 1
            if n_one > n_zero : int_bin_list[5] = '1'
            else: int_bin_list[5] = '0'
                        #int_bin_list[5] = '0'
            #option = ['0', '1']
            #int_bin_list[5] = random.choice(option)
            #print(int_bin_list[5])
        int_bin_list[4] = '0'  
        int_bin_list[3] = '0'            
        new_int_bin = ''.join(int_bin_list)
        new_value = abs(int(new_int_bin, 2))
        # print(new_value, new_int_bin)
        # print("*********************")
        return new_value


    def protect(self):

        flattened_copy = self.flattened.clone()
            # print(flattened_copy)
            #print("random in flattened", iindex)
            #print(flattened_copy[iindex])
            #range: self.flattened.shape[0]

        for iindex in range(self.flattened.shape[0]):
            # print(flattened_copy[iindex])
            protected_weight_f = self.correct(flattened_copy[iindex])
            # print(protected_weight_f)
            protected_weight = torch.tensor(protected_weight_f)
            flattened_copy[iindex] = protected_weight
            #print(flattened_copy[iindex])

        modified_weights = flattened_copy.reshape(self.x_size, self.z_size, self.y_size, self.y_size)
        self.updated_weights = modified_weights.to(self.weight.dtype)
        return self.updated_weights
