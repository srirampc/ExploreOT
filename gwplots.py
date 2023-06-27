import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

import pickle
import pathlib
import itertools
import sys
from os import path
from abc import ABC, abstractmethod

def set_default_args(kargs):
    default_args = {
                    "batchsize": [1],
                    "best_T_not_needed": True,
                    "entropy": [0.1],
                    "entropy_SGWL": [0.1],
                    "figsize": (10, 5),
                    "fontsize_axis" : 17,
                    "fontsize_legend" : 17,
                    "fontsize_ticks" : 14,
                    "iteration": [1000],
                    "legend_SGW": False,
                    "linestyle": "dotted",
                    "log_scale": True,
                    "loss_func_name": "1_loss",
                    "markersize": 10,
                    "n_samples_s": [50, 100, 200],
                    "n_samples_t": [],
                    "name_dataset": "gaussian",
                    "names_algo": ["sampled_gromov"],
                    "names_algo_legend": ["sampled_gromov"],
                    "noise_graph": 0.1,
                    "paper_plot": False,
                    "pickle_name": "",
                    "pickle_path": "pickle_compare",
                    "save": False,
                    "show_legend": True,
                    }
    for kx, vx in default_args.items():
        if kx not in kargs:
            kargs[kx] = vx
    return kargs

def set_names_algo_args(kargs):
    if kargs["name_dataset"] in ["gaussian_graph", "same_graph"]:
        try:
            kargs["names_algo"].remove("sliced_gromov")
            kargs["names_algo_legend"].remove("SGW")
        except:
            pass
    if kargs["name_dataset"] in ["gaussian_graph", "gaussian_point_graph"]:
        try:
            kargs["names_algo"].remove("identity")
            kargs["names_algo_legend"].remove("Identity")
        except:
            pass
    return kargs

def set_paper_plot_args(kargs):
    if kargs["paper_plot"]:
        # I found manually the best param for each dataset.
        # Print the average best value over all the iteration, this give the best parameter.
        if kargs["name_dataset"] == "same_graph":
            kargs["entropy"] = [0.005]
        elif kargs["name_dataset"] == "gaussian_graph":
            kargs["entropy"] = [0.01]
        elif kargs["name_dataset"] == "gaussian_point_graph":
            kargs["entropy"] = [0.005]
        #
        if kargs["name_dataset"] == "same_graph":
            kargs["entropy_SGWL"] = [100]
        elif kargs["name_dataset"] == "gaussian_graph":
            kargs["entropy_SGWL"] = [1]
        elif kargs["name_dataset"] == "gaussian_point_graph":
            kargs["entropy_SGWL"] = [10]
        #
        if kargs["name_dataset"] == "same_graph":
            kargs["best_param_m"] = 500
            kargs["best_param_b"] = 10
        elif kargs["name_dataset"] == "gaussian_graph":
            kargs["best_param_m"] = 500
            kargs["best_param_b"] = 10
        elif kargs["name_dataset"] == "gaussian_point_graph":
            kargs["best_param_m"] = 100
            kargs["best_param_b"] = 10
    return kargs

def set_n_samples_args(kargs):
    kargs["n_samples_s"] = np.array(kargs["n_samples_s"])
    if not kargs["n_samples_t"]:
        kargs["n_samples_t"] = kargs["n_samples_s"]
    return kargs

def load_pickle(path_pickle):
    try:
        with open(path_pickle, "rb") as pickle_in:
            return pickle.load(pickle_in)
    except:
        return None

def load_and_verify_pickle(params, same_param,
                           acces_dict, path_pickle):
    if not path.exists(path_pickle):
        params[acces_dict] = None
        # print(path_pickle)
        # print("outer")
        print("Pickle Path doesn't exist", path_pickle, file=sys.stderr)
        return params, same_param
    params[acces_dict] = load_pickle(path_pickle)
    if same_param is None:
        same_param = params[acces_dict]
        return params, same_param
    if params[acces_dict] is None:
        print("Failed to Load Pickle", path_pickle)
        return params, same_param
    if params[acces_dict]["name_algo"] == "uniform":
        return params, same_param
    # Use this to check if every run use the same parameter
    for test in ["n_iter_algo", "rdm_seed", "same_space"]:
        if same_param[test] != params[acces_dict][test]:
            #print(params[acces_dict])
            print("Warning", acces_dict, test, same_param[test], 
                  params[acces_dict][test])
    # assert same_param[test] == params[acces_dict][test]
    return params, same_param
 

class OPTMethodRun(ABC):
    def __init__(self, kargs, method, color, marker):
        self.kargs = kargs
        self.method = method
        self.color = color
        self.marker = marker

    @abstractmethod
    def param_key(self, **pwx):
        pass

    @abstractmethod
    def pickle_file_name(self, **fwx):
        pass

    @abstractmethod
    def load_run_params(self, params, same_param, **dwx):
        pass

    @abstractmethod
    def plot_runtimes(self, params, same_param, method_legend):
        pass

    @abstractmethod
    def plot_distances(self, params, same_param, method_legend):
        pass

    def set_method(self, method):
        self.method = method

    def acc_runtimes(self, params, same_param, pkey_list):
        nss_dim = len(self.kargs["n_samples_s"])
        niters = same_param["n_iter_algo"]
        time_list_only_W = np.zeros((nss_dim, niters)) 
        time_list_only_T = np.zeros((nss_dim, niters))
        time_list_only_W_approx = np.zeros((nss_dim, niters))
        time_list = np.zeros((nss_dim, niters))
        mask = np.ones(len(self.kargs["n_samples_s"]), dtype=bool)
        for i in range(nss_dim):
            acces_dict = pkey_list[i]
            if params[acces_dict] is None:
                mask[i] = 0
                continue
            time_list_only_W_approx[i] = params[acces_dict]["time_list_only_W_approx"]
            time_list_only_T[i] = params[acces_dict]["time_list_only_T"]
            time_list_only_W[i] = params[acces_dict]["time_list_only_W"]
            time_list[i] = time_list_only_T[i] + time_list_only_W[i]
        return time_list, time_list_only_T, mask

    def plt_fig_runtimes(self, time_F, time_T, mask,
                         method_legend, param_legend, skipOne=False):
        linestyle = self.kargs["linestyle"]
        figsize = self.kargs["figsize"]
        markersize = self.kargs["markersize"]
        # Plot 0
        plt.figure(0, figsize=figsize)
        plt.plot(self.kargs["n_samples_s"][mask], 
                 np.mean(time_F, axis=1)[mask], 
                 linestyle=linestyle, 
                 label=method_legend + param_legend,
                 color=self.color, marker=self.marker,
                 markersize=markersize)
        plt.fill_between(self.kargs["n_samples_s"][mask],
             (np.mean(time_F, axis=1) - np.std(time_F, axis=1))[mask],
             (np.mean(time_F, axis=1) + np.std(time_F, axis=1))[mask],
             alpha=0.3, color=self.color)
        # Plot 1
        plt.figure(1, figsize=figsize)
        if skipOne:
            return
        plt.plot(self.kargs["n_samples_s"][mask], 
                 np.mean(time_T, axis=1)[mask],
                 linestyle=linestyle,
                 label=method_legend + param_legend, 
                 color=self.color, marker=self.marker,
                 markersize=markersize)
        plt.fill_between(self.kargs["n_samples_s"][mask],
             (np.mean(time_T, axis=1) - np.std(time_T, axis=1))[mask],
             (np.mean(time_T, axis=1) + np.std(time_T, axis=1))[mask],
             alpha=0.3, color=self.color)

    def acc_distances(self, params, same_param, pkey_list):
        nss_dim = len(self.kargs["n_samples_s"])
        niters = same_param["n_iter_algo"]
        W_distance_list = np.ones((nss_dim, niters)) * 10e10
        W_distance_list_approx = np.zeros((nss_dim, niters))
        mask = np.ones(nss_dim, dtype=bool)
        for i in range(nss_dim):
            acces_dict = pkey_list[i]
            if params[acces_dict] is None:
                mask[i] = False
                continue
            if self.kargs["n_samples_s"][i] > 1000:
                mask[i] = False
            W_distance_list[i, :] = params[acces_dict]["W_distance_list"]
            W_distance_list_approx[i, :] = params[acces_dict]["W_distance_list_approx"]
        return W_distance_list, W_distance_list_approx, mask

    def plt_fig_distances(self, dist_F, mask,
                          method_legend, param_legend):
        linestyle = self.kargs["linestyle"]
        figsize = self.kargs["figsize"]
        markersize = self.kargs["markersize"]
        # Plot 2
        plt.figure(2, figsize=figsize)
        plt.plot(self.kargs["n_samples_s"][mask],
             np.mean(dist_F[mask], axis=1), 
             linestyle=linestyle, 
             label=method_legend + param_legend, 
             color=self.color, marker=self.marker, markersize=markersize)
        plt.fill_between(self.kargs["n_samples_s"][mask],
             (np.mean(dist_F, axis=1) - np.std(dist_F, axis=1))[mask],
             (np.mean(dist_F, axis=1) + np.std(dist_F, axis=1))[mask],
             alpha=0.3,
             color=self.color)


class EGromovRun(OPTMethodRun):
    def __init__(self, kargs, method = "e_gromov",
                 color = (0, 0.5, 0.8), marker = "o"):
        super().__init__(kargs, method, color, marker)

    def param_key(self, **pwx):
       return "{method}{s}_{t}{e}".format(method=self.method, s=pwx["s"],
                                          t=pwx["t"], e=pwx["e"])

    def pickle_file_name(self, **fwx):
        if self.kargs["name_dataset"] == "same_graph":
            return "{s}_{t}_{loss}_{noise}_{e}__{name}.pickle".format(
                    s=fwx["s"], t=fwx["t"], loss=self.kargs["loss_func_name"],
                    noise=self.kargs["noise_graph"], e=fwx["e"],
                    name=self.kargs["pickle_name"])
        else:
            return "{s}_{t}_{loss}_{e}_{name}.pickle".format(
                    s=fwx["s"], t=fwx["t"], loss=self.kargs["loss_func_name"],
                    e=fwx["e"], name=self.kargs["pickle_name"])

    def load_run_params(self, params, same_param, **dwx):
        for e in self.kargs["entropy"]:
            acces_dict = self.param_key(s=dwx["s"], t=dwx["t"], e=e)
            pickle_fname = self.pickle_file_name(s=dwx["s"], t=dwx["t"], e=e)
            pickle_path = pathlib.PurePath("./", self.kargs["pickle_path"], 
                            self.kargs["name_dataset"], self.method, pickle_fname)
            params, same_param = load_and_verify_pickle(params, same_param, 
                                                        acces_dict, pickle_path)
        return params, same_param

    def get_runtimes(self, params, same_param, **rwx):
        pkey_list = [self.param_key(s=nss, t=nst, e=rwx["e"]) for nss,nst in zip(
            self.kargs["n_samples_s"], self.kargs["n_samples_t"])]
        return self.acc_runtimes(params, same_param, pkey_list)
 
    def plot_runtimes(self, params, same_param, method_legend):
        for e in self.kargs["entropy"]:
            time_F, time_T, mask = self.get_runtimes(params, same_param, e=e)
            if self.kargs["paper_plot"]:
                legend_param = ""
            else:
                legend_param = "$_{" + str(e) + "}$"
            self.plt_fig_runtimes(time_F, time_T, mask,
                                  method_legend, legend_param)

    def get_distances(self, params, same_param, **dwx):
        pkey_list = [self.param_key(s=nss, t=nst, e=dwx["e"]) for nss,nst in zip(
            self.kargs["n_samples_s"], self.kargs["n_samples_t"])]
        return self.acc_distances(params, same_param, pkey_list)

    def plot_distances(self, params, same_param, method_legend):
        for e in self.kargs["entropy"]:
            distance_W, _, mask = self.get_distances(params, same_param, e=e)
            if self.kargs["paper_plot"]:
                legend_param = ""
            else:
                legend_param = "$_{" + str(e) + "}$"
            self.plt_fig_distances(distance_W, mask,
                                  method_legend, legend_param)


class EGromovKLRun(EGromovRun):
    def __init__(self, kargs, method = "e_gromov_KL",
                 color = (0, 0, 1), marker = "P"):
        super().__init__(kargs, method, color, marker)


class SGWLRun(OPTMethodRun):
    def __init__(self, kargs, method = "S_GWL",
                color = (0, 0, 0), marker = "*"):
        super().__init__(kargs, method, color, marker)

    def param_key(self, **pwx):
       return "{method}{s}_{t}{e}".format(method=self.method, s=pwx["s"],
                                          t=pwx["t"], e=pwx["e"])

    def pickle_file_name(self, **fwx):
        if self.kargs["name_dataset"] == "same_graph":
            return "{s}_{t}_{loss}_{noise}_{e}__{name}.pickle".format(
                    s=fwx["s"], t=fwx["t"], loss=self.kargs["loss_func_name"],
                    noise=self.kargs["noise_graph"], e=fwx["e"],
                    name=self.kargs["pickle_name"])
        else:
            return "{s}_{t}_{loss}_{e}_{name}.pickle".format(
                    s=fwx["s"], t=fwx["t"], loss=self.kargs["loss_func_name"],
                    e=fwx["e"], name=self.kargs["pickle_name"])

    def load_run_params(self, params, same_param, **dwx):
        for e in self.kargs["entropy_SGWL"]:
            acces_dict = self.param_key(s=dwx["s"], t=dwx["t"], e=e)
            pickle_fname = self.pickle_file_name(s=dwx["s"], t=dwx["t"], e=e)
            pickle_path = pathlib.PurePath("./", self.kargs["pickle_path"],
                                           self.kargs["name_dataset"], 
                                           self.method, pickle_fname)
            params, same_param = load_and_verify_pickle(params, same_param, 
                                                        acces_dict, pickle_path)
        return params, same_param

    def get_runtimes(self, params, same_param, **rwx):
        pkey_list = [self.param_key(s=nss, t=nst, e=rwx["e"]) for nss,nst in zip(
            self.kargs["n_samples_s"], self.kargs["n_samples_t"])]
        return self.acc_runtimes(params, same_param, pkey_list)

    def plot_runtimes(self, params, same_param, method_legend):
        for e in self.kargs["entropy_SGWL"]:
            time_F, time_T, mask = self.get_runtimes(params, same_param, e=e)
            if self.kargs["paper_plot"]:
                legend_param = ""
            else:
                legend_param = "$_{" + str(e) + "}$"
            self.plt_fig_runtimes(time_F, time_T, mask,
                                   method_legend, legend_param)

    def get_distances(self, params, same_param, **dwx):
        pkey_list = [self.param_key(s=nss, t=nst, e=dwx["e"]) for nss,nst in zip(
            self.kargs["n_samples_s"], self.kargs["n_samples_t"])]
        return self.acc_distances(params, same_param, pkey_list)

    def plot_distances(self, params, same_param, method_legend):
        for e in self.kargs["entropy_SGWL"]:
            distance_W, _, mask = self.get_distances(params, same_param, e=e)
            if self.kargs["paper_plot"]:
                legend_param = ""
            else:
                legend_param = "$_{" + str(e) + "}$"
            self.plt_fig_distances(distance_W, mask,
                                   method_legend, legend_param)


class SampledGWRun(OPTMethodRun):
    def __init__(self, kargs, method = "sampled_gromov",
                 color = (1, 0, 0), marker = "s"):
        super().__init__(kargs, method, color, marker)
        if self.kargs["paper_plot"]:
            color = (1, 0, 0)
            marker = "s"

    def param_key(self, **pwx):
       return "{method}{s}_{t}{b}{m}".format(method=self.method, s=pwx["s"],
                                             t=pwx["t"], b=pwx["b"], m=pwx["m"])

    def pickle_file_name(self, **fwx):
        if self.kargs["name_dataset"] == "same_graph":
            return "{s}_{t}_{loss}__{noise}_{b}_{m}_{name}.pickle".format(
                    s=fwx["s"], t=fwx["t"], loss=self.kargs["loss_func_name"],
                    noise=self.kargs["noise_graph"], b=fwx["b"], m = fwx["m"],
                    name=self.kargs["pickle_name"])
        else:
            return "{s}_{t}_{loss}_{b}_{m}{name}.pickle".format(
                    s=fwx["s"], t=fwx["t"], loss=self.kargs["loss_func_name"],
                    b=fwx["b"], m = fwx["m"], name=self.kargs["pickle_name"])

    def load_run_params(self, params, same_param, **dwx):
        for b, m in itertools.product(self.kargs["batchsize"],
                                      self.kargs["iteration"]):
            acces_dict = self.param_key(s=dwx["s"], t=dwx["t"], b=b, m=m)
            pickle_fname = self.pickle_file_name(s=dwx["s"], t=dwx["t"], b=b, m=m)
            pickle_path = pathlib.PurePath("./", self.kargs["pickle_path"], 
                                           self.kargs["name_dataset"],
                                           self.method, pickle_fname)
            params, same_param = load_and_verify_pickle(params, same_param, 
                                                        acces_dict, pickle_path)
        return params, same_param

    def get_runtimes(self, params, same_param, **rwx):
        if self.kargs["paper_plot"]:
            if self.method == "sampled_gromov_sliced":
                if rwx["b"] != 1 or rwx["m"] != 100:
                    return None, None, None
            else:
                if rwx["b"] != self.kargs["best_param_b"] or rwx["m"] != self.kargs["best_param_m"]:
                    return None, None, None
        pkey_list = [self.param_key(s=nss, t=nst,
                                    b=rwx["b"], m=rwx["m"])  for nss,nst in zip(
            self.kargs["n_samples_s"], self.kargs["n_samples_t"])]
        return self.acc_runtimes(params, same_param, pkey_list)

    def plot_runtimes(self, params, same_param, method_legend):
        batchsize = self.kargs["batchsize"]
        iteration = self.kargs["iteration"]
        for (b_index, b), m in itertools.product(enumerate(batchsize),
                                                 iteration):
            time_F, time_T, mask = self.get_runtimes(params, same_param,
                                                     b=b, m=m)
            if time_F is None or time_T is None:
                continue
            # print(mask)
            # linestyle = "-"
            if not self.kargs["paper_plot"]:
                self.color = (1, b_index / len(batchsize), 0) 
            self.plt_fig_runtimes(time_F, time_T, mask,
                                  method_legend, "")

    def get_distances(self, params, same_param, **dwx):
        if self.kargs["paper_plot"]:
            if self.method == "sampled_gromov_sliced":
                if dwx["b"] != 1 or dwx["m"] != 100:
                    return None, None, None
            else:
                if dwx["b"] != self.kargs["best_param_b"] or dwx["m"] != self.kargs["best_param_m"]:
                    return None, None, None
        pkey_list = [self.param_key(s=nss, t=nst,
                                    b=dwx["b"], m=dwx["m"])  for nss,nst in zip(
            self.kargs["n_samples_s"], self.kargs["n_samples_t"])]
        return self.acc_distances(params, same_param, pkey_list)

    def plot_distances(self, params, same_param, method_legend):
        batchsize = self.kargs["batchsize"]
        iteration = self.kargs["iteration"]
        for (b_index, b), m  in itertools.product(enumerate(batchsize),
                                                  iteration):
            distance_W, _,  mask = self.get_distances(params, same_param, b=b, m=m)
            if distance_W is None:
                continue
            if not self.kargs["paper_plot"]:
                self.color = (1, b_index / len(batchsize), 0)
            self.plt_fig_distances(distance_W, mask,
                                  method_legend, "")


class SampledGWEConstRun(SampledGWRun):
    def __init__(self, kargs, method = "sampled_gromov_e_constant"):
        super().__init__(kargs, method)


class SampledGWSlicedRun(SampledGWRun):
    def __init__(self, kargs, method = "sampled_gromov_sliced"):
        super().__init__(kargs, method)
        if self.kargs["paper_plot"]:
            self.color = (1, 0.5, 0)
            self.marker = "X"
 

class SampledGWNoKLEConstRun(SampledGWRun):
    def __init__(self, kargs, method = "sampled_gromov_no_KL_e_constant"):
        super().__init__(kargs, method)

    def pickle_file_name(self, **fwx):
        if self.kargs["name_dataset"] == "same_graph":
             return "{s}_{t}_{loss}__{noise}_{b}_{m}_{name}.pickle".format(
                s=fwx["s"], t=fwx["t"], loss=self.kargs["loss_func_name"],
                noise=self.kargs["noise_graph"], b=fwx["b"], m = fwx["m"],
                name=self.kargs["pickle_name"])
        else:
             return "{s}_{t}_{loss}_{b}_{m}_3.pickle".format(
                s=fwx["s"], t=fwx["t"], loss=self.kargs["loss_func_name"],
                b=fwx["b"], m = fwx["m"])


class DefaultRun(OPTMethodRun):
    def __init__(self, kargs, method = "default", color = (0, 0.8, 0.5), #(0, 0, 1)
                 marker = "+", skipOne = False):
        super().__init__(kargs, method, color, marker)
        self.legend_SGW  = self.kargs["legend_SGW"]
        self.skipOne = skipOne

    def param_key(self, **pwx):
        return "{method}{s}_{t}".format(method=self.method, s=pwx["s"],
                                        t=pwx["t"])

    def pickle_file_name(self, **fwx):
        if self.kargs["name_dataset"] == "same_graph":
            return "{s}_{t}_{loss}__{noise}_{name}.pickle".format(
                s=fwx["s"], t=fwx["t"], loss = self.kargs["loss_func_name"],
                noise=self.kargs["noise_graph"], name=self.kargs["pickle_name"])
        else:
            return "{s}_{t}_{loss}{name}.pickle".format(
                s=fwx["s"], t=fwx["t"], loss = self.kargs["loss_func_name"],
                name=self.kargs["pickle_name"])

    def load_run_params(self, params, same_param, **dwx):
        acces_dict = self.param_key(s=dwx["s"], t=dwx["t"])
        pickle_fname = self.pickle_file_name(s=dwx["s"], t=dwx["t"])
        pickle_path = pathlib.PurePath("./", self.kargs["pickle_path"], 
                                       self.kargs["name_dataset"],
                                       self.method, pickle_fname)
        params, same_param = load_and_verify_pickle(params, same_param, 
                                                    acces_dict, pickle_path)
        return params, same_param

    def get_runtimes(self, params, same_param, **rwx):
        pkey_list = [self.param_key(s=nss, t=nst) for nss,nst in zip(
            self.kargs["n_samples_s"], self.kargs["n_samples_t"])]
        return self.acc_runtimes(params, same_param, pkey_list)

    def plot_runtimes(self, params, same_param, method_legend):
        time_F, time_T, mask = self.get_runtimes(params, same_param)
        self.plt_fig_runtimes(time_F, time_T, mask,
                              method_legend, "", self.skipOne)

    def get_distances(self, params, same_param, **dwx):
        pkey_list = [self.param_key(s=nss, t=nst) for nss,nst in zip(
            self.kargs["n_samples_s"], self.kargs["n_samples_t"])]
        return self.acc_distances(params, same_param, pkey_list)

    def plot_distances(self, params, same_param, method_legend):
        linestyle = self.kargs["linestyle"]
        distance_W, _,  mask = self.get_distances(params, same_param)
        self.plt_fig_distances(distance_W, mask,
                              method_legend, "")
        # here just to add Sliced GW in the legend. Kind of ugly...
        if "gaussian_graph" == self.kargs["name_dataset"] and self.kargs["paper_plot"] and self.legend_SGW:
            # linestyle = "dotted"
            color = (0.40, 0.2, 0)
            plt.plot([None], [None], linestyle=linestyle,
                     label="SGW", color=color)
            plt.plot([None], [None], linestyle=linestyle,
                     label="15 SGW", color=color)
            self.legend_SGW = False

class UniformRun(DefaultRun):
    def __init__(self, kargs, method = "uniform",
                 color = (0.1, 0.5, 0.1), marker = "d"):
        super().__init__(kargs, method, color, marker, True)


class IdentityRun(DefaultRun):
    def __init__(self, kargs, method = "identity",
                color = (0, 0.8, 0.5), #(0, 0, 1)
                marker = "+"):
        super().__init__(kargs, method, color, marker, True)


class SlicedGWRun(DefaultRun):
    def __init__(self, kargs, method = "sliced_gromov",
                color = (0.40, 0.2, 0), marker = "^"):
        super().__init__(kargs, method, color, marker)

    def plot_distances(self, params, same_param, method_legend):
        _, distance_Wx, mask = self.get_distances(params, same_param)
        self.plt_fig_distances(distance_Wx, mask, method_legend, "")
        self.plt_fig_distances(distance_Wx*25, mask, "25"+method_legend, "")


def analyse_data_hyper(**kargs):
    kargs = set_default_args(kargs)
    kargs = set_names_algo_args(kargs)
    kargs = set_n_samples_args(kargs)
    kargs = set_paper_plot_args(kargs)

    method_obj_lookup = {
        "e_gromov" : EGromovRun(kargs),
        "e_gromov_KL" : EGromovKLRun(kargs),
        "S_GWL" : SGWLRun(kargs),
        "sampled_gromov_e_constant": SampledGWEConstRun(kargs),
        "sampled_gromov_no_KL_e_constant": SampledGWNoKLEConstRun(kargs),
        "sampled_gromov_sliced": SampledGWSlicedRun(kargs),
        "uniform": UniformRun(kargs),
        "identity": IdentityRun(kargs),
        "sliced_gromov": SlicedGWRun(kargs),
        "default" : DefaultRun(kargs),
    }
    params = {}
    same_param = None

    # Load all Pickle files
    for name_algo, (i, nss) in itertools.product(kargs["names_algo"], 
                                                 enumerate(kargs["n_samples_s"])):
        if name_algo in method_obj_lookup:
            gw_run = method_obj_lookup[name_algo]
        else:
            gw_run = method_obj_lookup["default"]
            gw_run.set_method(name_algo)
        nst = kargs["n_samples_t"][i]
        params, same_param = gw_run.load_run_params(params, same_param,
                                                    s=nss, t=nst)
    #
    assert same_param is not None  #  Error in the pickle file name ?
    # Plot the runtimes
    for name_algo, name_algo_legend in zip(kargs["names_algo"], 
                                           kargs["names_algo_legend"]):
        if name_algo in method_obj_lookup:
            gw_run = method_obj_lookup[name_algo]
        else:
            gw_run = method_obj_lookup["default"]
            gw_run.set_method(name_algo)
        gw_run.plot_runtimes(params, same_param, name_algo_legend)
    #
    # Update plots
    plt.figure(0)
    if kargs["log_scale"]:
        plt.yscale("log")
        plt.xscale("log")
    plt.xlabel('Number of points: $N$', fontsize=kargs["fontsize_axis"])
    plt.ylabel('Computational time (s)', fontsize=kargs["fontsize_axis"])
    plt.xticks(fontsize=kargs["fontsize_ticks"])
    plt.yticks(fontsize=kargs["fontsize_ticks"])
    if kargs["show_legend"]:
        plt.legend(loc=(1.01, 0.15), fontsize=kargs["fontsize_legend"])
    # plt.title("Time needed to compute T and Wasserstein")
    #
    plt.figure(1)
    if kargs["log_scale"]:
        plt.yscale("log")
        plt.xscale("log")
    plt.xlabel('Number of points: $N$', fontsize=kargs["fontsize_axis"])
    plt.ylabel('Computational time (s)', fontsize=kargs["fontsize_axis"])
    plt.xticks(fontsize=kargs["fontsize_ticks"])
    plt.yticks(fontsize=kargs["fontsize_ticks"])
    if kargs["show_legend"]:
        plt.legend(loc=(1.01, 0.15), fontsize=kargs["fontsize_legend"])
    # plt.title("Time needed to compute T")
    #
    for name_algo, name_algo_legend in zip(kargs["names_algo"],
                                           kargs["names_algo_legend"]):
        if name_algo in method_obj_lookup:
            gw_run = method_obj_lookup[name_algo]
        else:
            gw_run = method_obj_lookup["default"]
            gw_run.set_method(name_algo)
        gw_run.plot_distances(params, same_param, name_algo_legend)
    #
    if kargs["log_scale"]:
        plt.xscale("log")
    plt.xlabel('Number of points: $N$', fontsize=kargs["fontsize_axis"])
    plt.ylabel('GW value: $\mathcal{E}$ (T)', fontsize=kargs["fontsize_axis"])
    plt.xticks(fontsize=kargs["fontsize_ticks"])
    plt.yticks(fontsize=kargs["fontsize_ticks"])
    # if kargs["name_dataset"] == "gaussian_point_graph":
    #     loc = (1, 0)
    # else:
    #     loc = (1, 0)
    # if show_legend:
    #     plt.legend(loc=(-0.8, 0.05), fontsize=fontsize_legend)
    if kargs["show_legend"]:
        if kargs["name_dataset"] == "gaussian_graph":
            plt.legend(loc=(1.01, 0.15), fontsize=kargs["fontsize_legend"])
        else:
            plt.legend(loc=(1.01, 0.05), fontsize=kargs["fontsize_legend"])
    #
    if kargs["save"]:
        plt.figure(0)
        # plt.gca().set_axis_off()
        # plt.subplots_adjust(top=1, bottom=0, right=1, left=0,
        #                     hspace=0, wspace=0)
        # plt.margins(0, 0)
        # plt.gca().xaxis.set_major_locator(plt.NullLocator())
        # plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.savefig( "./figure/hyper/hyper_T_and_W_{name}_{loss}.pdf".format(
            name=kargs["name_dataset"], loss=kargs["loss_func_name"]),
                    bbox_inches="tight")
        plt.figure(1)
        plt.savefig("./figure/hyper/hyper_T_{name}_{loss}.pdf".format(
            name=kargs["name_dataset"], loss=kargs["loss_func_name"]),
                    bbox_inches="tight")
        plt.figure(2)
        plt.savefig("./figure/hyper/hyper_distance_{name}_{loss}.pdf".format(
            name = kargs["name_dataset"],  loss=kargs["loss_func_name"]),
                    bbox_inches="tight")
    plt.show()
