import os
from typing import Generator, OrderedDict
import numpy as np
import torch
import time
from collections import defaultdict

import argparse

class Timer():
    def __init__(self, running_avg=20, log_interval=10):
        self.history = defaultdict(list)
        self.running_avg = running_avg
        self.log_interval = log_interval

    def log(self, action, time):
        self.history[action].append(time)

        if len(self.history[action]) % self.log_interval == 0:
            interval = np.array(self.history[action][-self.running_avg:])
            print(f'Time for {action} took AVG={interval.mean()}, STD={interval.std()}')


class TimeIt():
    def __init__(self, timer, action, cuda_sync = False):
        self.timer = timer
        self.action = action
        self.cuda_sync = cuda_sync

    def __enter__(self):
        if self.cuda_sync == True:
            torch.cuda.synchronize()
        self.start = time.time()

    def __exit__(self, *args):
        if self.cuda_sync == True:
            torch.cuda.synchronize()

        self.end = time.time()
        self.timer.log(self.action, self.end - self.start)

def main():
    print('Running inference')

    print(f'Cuda available {torch.cuda.is_available()}')
    print(f'We use the GPU of id {torch.cuda.current_device()}')
    print(f'There are {torch.cuda.device_count()} visible')
    print(f'The GPU name is {torch.cuda.get_device_name(torch.cuda.current_device())}')

    import argparse

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--bs', type=int)
    parser.add_argument('--seq', type=int)
    parser.add_argument('--sseq', type=int)
    parser.add_argument('--gather', action='store_true')
    parser.add_argument('--repeat', action='store_true')
    args = parser.parse_args()

    timer = Timer(running_avg = 50, log_interval = 10)

    batch_size, shortened_seq_len, d_model = args.bs, args.sseq, 512
    seq_len = args.seq
    
    if args.repeat:
        for _ in range(100):
            with TimeIt(timer, 'Repeat interleave', cuda_sync=True) as _:
                data = torch.normal(0, 1, (batch_size, shortened_seq_len, d_model)).cuda()
                upsampling_mask = torch.randint(low=0, high=shortened_seq_len, size=(batch_size, seq_len)).cuda()
                with torch.no_grad():
                    x = data[torch.arange(batch_size).repeat_interleave(seq_len), upsampling_mask.flatten(), :]
    
    if args.gather:
        for _ in range(100):
            with TimeIt(timer, 'Gather', cuda_sync=True) as _:
                data = torch.normal(0, 1, (batch_size, shortened_seq_len, d_model)).cuda()
                upsampling_mask = torch.randint(low=0, high=shortened_seq_len, size=(batch_size, seq_len)).cuda()
                with torch.no_grad():
                    x = torch.gather(data, 1, upsampling_mask.long().unsqueeze(-1).repeat(1, 1, 512))

if __name__ == "__main__":
    torch.set_num_threads(1)
    main()
