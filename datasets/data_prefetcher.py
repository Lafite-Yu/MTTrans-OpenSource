# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

import torch
from util.misc import nested_tensor_from_tensor_list


def to_cuda(samples, targets, device):
    samples = samples.to(device, non_blocking=True)
    targets = [{k: v.to(device, non_blocking=True) for k, v in t.items()} for t in targets]
    return samples, targets

class data_prefetcher():
    def __init__(self, loader, device, prefetch=True):
        self.loader = iter(loader)
        self.prefetch = prefetch
        self.device = device
        if prefetch:
            self.stream = torch.cuda.Stream()
            self.preload()

    def preload(self):
        try:
            self.next_samples, self.next_targets = next(self.loader)
        except StopIteration:
            self.next_samples = None
            self.next_targets = None
            return
        # if record_stream() doesn't work, another option is to make sure device inputs are created
        # on the main stream.
        # self.next_input_gpu = torch.empty_like(self.next_input, device='cuda')
        # self.next_target_gpu = torch.empty_like(self.next_target, device='cuda')
        # Need to make sure the memory allocated for next_* is not still in use by the main stream
        # at the time we start copying to next_*:
        # self.stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self.stream):
            self.next_samples, self.next_targets = to_cuda(self.next_samples, self.next_targets, self.device)
            # more code for the alternative if record_stream() doesn't work:
            # copy_ will record the use of the pinned source tensor in this side stream.
            # self.next_input_gpu.copy_(self.next_input, non_blocking=True)
            # self.next_target_gpu.copy_(self.next_target, non_blocking=True)
            # self.next_input = self.next_input_gpu
            # self.next_target = self.next_target_gpu

            # With Amp, it isn't necessary to manually convert data to half.
            # if args.fp16:
            #     self.next_input = self.next_input.half()
            # else:

    def next(self):
        if self.prefetch:
            torch.cuda.current_stream().wait_stream(self.stream)
            samples = self.next_samples
            targets = self.next_targets
            if samples is not None:
                samples.record_stream(torch.cuda.current_stream())
            if targets is not None:
                for t in targets:
                    for k, v in t.items():
                        v.record_stream(torch.cuda.current_stream())
            self.preload()
        else:
            try:
                samples, targets = next(self.loader)
                samples, targets = to_cuda(samples, targets, self.device)
            except StopIteration:
                samples = None
                targets = None
        return samples, targets


class data_prefetcher_teacher_student():
    def __init__(self, loader, device, prefetch=True):
        self.loader = iter(loader)
        self.prefetch = prefetch
        self.device = device
        if prefetch:
            self.stream = torch.cuda.Stream()
            self.preload()

    def preload(self):
        try:
            self.t_samples, self.t_targets, self.s_samples, self.s_targets = next(self.loader)
            self.s_samples = nested_tensor_from_tensor_list(self.s_samples)
        except StopIteration:
            self.t_samples, self.t_targets = None, None
            self.s_samples, self.s_targets = None, None
            return
        # if record_stream() doesn't work, another option is to make sure device inputs are created
        # on the main stream.
        # self.next_input_gpu = torch.empty_like(self.next_input, device='cuda')
        # self.next_target_gpu = torch.empty_like(self.next_target, device='cuda')
        # Need to make sure the memory allocated for next_* is not still in use by the main stream
        # at the time we start copying to next_*:
        # self.stream.wait_stream(torch.cuda.current_stream())
        # print(11111111111, type(self.t_samples), type(self.t_targets), type(self.s_samples), type(self.s_targets))
        with torch.cuda.stream(self.stream):
            # print(2222222222222222, type(self.t_samples), type(self.t_targets), type(self.s_samples), type(self.s_targets))
            self.t_samples, self.t_targets = to_cuda(self.t_samples, self.t_targets, self.device)
            self.s_samples, self.s_targets = to_cuda(self.s_samples, self.s_targets, self.device)
            # more code for the alternative if record_stream() doesn't work:
            # copy_ will record the use of the pinned source tensor in this side stream.
            # self.next_input_gpu.copy_(self.next_input, non_blocking=True)
            # self.next_target_gpu.copy_(self.next_target, non_blocking=True)
            # self.next_input = self.next_input_gpu
            # self.next_target = self.next_target_gpu

            # With Amp, it isn't necessary to manually convert data to half.
            # if args.fp16:
            #     self.next_input = self.next_input.half()
            # else:

    def next(self):
        if self.prefetch:
            torch.cuda.current_stream().wait_stream(self.stream)
            t_samples, t_targets = self.t_samples, self.t_targets
            s_samples, s_targets = self.s_samples, self.s_targets
            if t_samples is not None:
                t_samples.record_stream(torch.cuda.current_stream())
            if t_targets is not None:
                for t in t_targets:
                    for k, v in t.items():
                        v.record_stream(torch.cuda.current_stream())

            if s_samples is not None:
                s_samples.record_stream(torch.cuda.current_stream())
            if s_targets is not None:
                for t in s_targets:
                    for k, v in t.items():
                        v.record_stream(torch.cuda.current_stream())
            self.preload()
        else:
            try:
                t_samples, t_targets, s_samples, s_targets = next(self.loader)
                s_samples = nested_tensor_from_tensor_list(s_samples)
                t_samples, t_targets = to_cuda(t_samples, t_targets, self.device)
                s_samples, s_targets = to_cuda(s_samples, s_targets, self.device)
            except StopIteration:
                t_samples, t_targets = None, None
                s_samples, s_targets = None, None
        return (t_samples, t_targets), (s_samples, s_targets)