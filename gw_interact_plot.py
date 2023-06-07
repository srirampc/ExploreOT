import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import pickle
from os import path


def hyperparameter_analysis(n_samples_s=500,
                            name_dataset="gaussian",
                            pickle_path="pickle_compare",
                            names_algo=["sampled_gromov"],
                            names_algo_legend=["sampled_gromov"],
                            pickle_name="",
                            noise_graph=0.1,
                            loss_func_name="1_loss",
                            figsize=(10, 5),
                            save=False,
                            entropy=[0.1],
                            batchsize=[1],
                            iteration=[1000],
                            log_scale=False,
                            min_plot=False,
                            only_PGW=True,
                            final_plot=True):
    plt.figure(0, figsize=figsize)
    already_plot_SGW_legend = False
    if name_dataset in ["gaussian_graph", "same_graph", "gaussian_point_graph"]:
        # Can't compare sliced with other.
        try:
            names_algo.remove("sliced_gromov")
            names_algo_legend.remove("Sliced GW")
        except:
            pass
    params = {}
    same_param = None
    markersize = 10
    for name_algo, name_algo_legend in zip(names_algo, names_algo_legend):
        if name_algo in ["e_gromov", "e_gromov_KL"]:  # "e_gromov_KL"
            m_save_time = []
            m_save_distance = []
            for e_index, e in enumerate(entropy):
                acces_dict = name_algo + str(n_samples_s) + "_" + str(n_samples_s) + str(e)

                if name_dataset == "same_graph":
                    path_pickle = "./" + pickle_path + "/" + name_dataset + "/" + name_algo + "/" + str(
                        n_samples_s) + "_" + str(n_samples_s) + "_" + loss_func_name + "__" + str(noise_graph) + "_" + \
                                  str(e) + "__" + pickle_name + ".pickle"
                else:
                    path_pickle = "./" + pickle_path + "/" + name_dataset + "/" + name_algo + "/" + str(
                        n_samples_s) + "_" + str(n_samples_s) + "_" + loss_func_name + "_" + str(
                        e) + "_" + pickle_name + ".pickle"
                if path.exists(path_pickle):
                    with open(path_pickle, "rb") as pickle_in:
                        params[acces_dict] = pickle.load(pickle_in)
                        if same_param is None:
                            same_param = params[acces_dict]
                        elif params[acces_dict]["name_algo"] != "uniform":
                            # Use this to check if every run use the same parameter
                            for test in ["n_iter_algo", "rdm_seed", "same_space"]:
                                if same_param[test] != params[acces_dict][test]:
                                    print(params[acces_dict])
                                    pass
                                    print("Warning", test, same_param[test], params[acces_dict][test])
                    time_list_only_W_approx = params[acces_dict]["time_list_only_W_approx"]
                    time_list_only_T = params[acces_dict]["time_list_only_T"]
                    time_list_only_W = params[acces_dict]["time_list_only_W"]
                    time_list = time_list_only_T + time_list_only_W
                    W_distance_list_approx = params[acces_dict]["W_distance_list_approx"]
                    W_distance_list = params[acces_dict]["W_distance_list"]
                    color = (0, 1 - e_index / (len(entropy) - 1), e_index / (len(entropy) - 1))
                    label = None
                    if name_algo == "e_gromov":
                        marker = 'o'
                    else:
                        marker = 'P'
                    if e_index == 0:
                        if name_algo == "e_gromov":
                            label = "$EGW$"
                            plt.plot([], [],
                                     marker=marker,
                                     color=(0, 0, 0),
                                     label=label,
                                     markersize=markersize,
                                     linestyle="None")
                        else:
                            label = "$EGW^{KL}$"
                            plt.plot([], [],
                                     marker=marker,
                                     color=(0, 0, 0),
                                     label=label,
                                     markersize=markersize,
                                     linestyle="None")
                    markersize = 10
                    label = None
                    if min_plot:
                        W_distance_list_ = np.min(W_distance_list)
                        time_list_only_T_ = time_list_only_T[np.argmin(W_distance_list)]
                    else:
                        W_distance_list_ = np.mean(W_distance_list)
                        time_list_only_T_ = np.mean(time_list_only_T)
                    m_save_time.append(time_list_only_T_)
                    m_save_distance.append(W_distance_list_)

                    plt.plot(time_list_only_T_, W_distance_list_,
                             marker=marker,
                             color=color,
                             label=label,
                             markersize=markersize,
                             linestyle="None")
                else:
                    params[acces_dict] = None
            # acces_dict = "gromov" + str(n_samples_s) + "_" + str(n_samples_s)
            # if params[acces_dict] is not None:
            #     m_save_time = [np.mean(params[acces_dict]["time_list_only_T"])] + m_save_time
            #     m_save_distance = [np.mean(params[acces_dict]["W_distance_list"])] + m_save_distance
            plt.plot(m_save_time,
                     m_save_distance,
                     "-",
                     color=(0, 0.5, 0.5),
                     markersize=3,
                     linewidth=0.5)

        elif name_algo in ["S_GWL"]:
            m_save_time = []
            m_save_distance = []
            for e_index, e in enumerate(entropy):
                acces_dict = name_algo + str(n_samples_s) + "_" + str(n_samples_s) + str(e)

                if name_dataset == "same_graph":
                    path_pickle = "./" + pickle_path + "/" + name_dataset + "/" + name_algo + "/" + str(
                        n_samples_s) + "_" + str(n_samples_s) + "_" + loss_func_name + "__" + str(noise_graph) + "_" + \
                                  str(e) + "__" + pickle_name + ".pickle"
                else:
                    path_pickle = "./" + pickle_path + "/" + name_dataset + "/" + name_algo + "/" + str(
                        n_samples_s) + "_" + str(n_samples_s) + "_" + loss_func_name + "_" + str(
                        e) + "_" + pickle_name + ".pickle"
                if path.exists(path_pickle):
                    with open(path_pickle, "rb") as pickle_in:
                        params[acces_dict] = pickle.load(pickle_in)
                        if same_param is None:
                            same_param = params[acces_dict]
                        elif params[acces_dict]["name_algo"] != "uniform":
                            # Use this to check if every run use the same parameter
                            for test in ["n_iter_algo", "rdm_seed", "same_space"]:
                                if same_param[test] != params[acces_dict][test]:
                                    print(params[acces_dict])
                                    pass
                                    print("Warning", test, same_param[test], params[acces_dict][test])
                    time_list_only_W_approx = params[acces_dict]["time_list_only_W_approx"]
                    time_list_only_T = params[acces_dict]["time_list_only_T"]
                    time_list_only_W = params[acces_dict]["time_list_only_W"]
                    time_list = time_list_only_T + time_list_only_W
                    W_distance_list_approx = params[acces_dict]["W_distance_list_approx"]
                    W_distance_list = params[acces_dict]["W_distance_list"]
                    color = (0, 1 - e_index / (len(entropy) - 1), e_index / (len(entropy) - 1))

                    marker = "*"
                    # print(e_index, e)
                    if not already_plot_SGW_legend and final_plot:
                        already_plot_SGW_legend = True
                        plt.plot([], [],
                                 marker=marker,
                                 color=(0, 0, 0),
                                 label=name_algo_legend,
                                 markersize=markersize,
                                 linestyle="None")
                    # markersize = 10
                    label = None
                    # if not final_plot:
                    #     label = name_algo_legend
                    color = (0, 1 - e_index / (len(entropy) - 1), e_index / (len(entropy) - 1))
                    if min_plot:
                        W_distance_list_ = np.min(W_distance_list)
                        time_list_only_T_ = time_list_only_T[np.argmin(W_distance_list)]
                    else:
                        W_distance_list_ = np.mean(W_distance_list)
                        time_list_only_T_ = np.mean(time_list_only_T)
                    m_save_time.append(time_list_only_T_)
                    m_save_distance.append(W_distance_list_)

                    plt.plot(time_list_only_T_, W_distance_list_,
                             marker=marker,
                             color=color,
                             label=label,
                             markersize=markersize,
                             linestyle="None")
                else:
                    params[acces_dict] = None
            # acces_dict = "gromov" + str(n_samples_s) + "_" + str(n_samples_s)
            # if params[acces_dict] is not None:
            #     m_save_time = [np.mean(params[acces_dict]["time_list_only_T"])] + m_save_time
            #     m_save_distance = [np.mean(params[acces_dict]["W_distance_list"])] + m_save_distance
            plt.plot(m_save_time,
                     m_save_distance,
                     "-",
                     color=(0, 0.5, 0.5),
                     markersize=3,
                     linewidth=0.5)

        elif name_algo in ["sampled_gromov_e_constant", "sampled_gromov_sliced", "sampled_gromov_no_KL_e_constant"]:
            if name_algo == "sampled_gromov_e_constant" or name_algo == "sampled_gromov_no_KL_e_constant":
                plt.plot([], [],
                         marker="",
                         color=(0, 0, 0),
                         label="$SaGroW^{KL}$" if name_algo == "sampled_gromov_e_constant" else "$SaGroW$",
                         markersize=2,
                         linestyle="None")
                plt.plot([], [],
                         marker="X",
                         color=(0, 0, 0),
                         label="PoGroW",
                         markersize=10,
                         linestyle="None")
                plt.plot([], [],
                         marker="x",
                         color=(0, 0, 0),
                         label="Minimum\nover\n10 runs",
                         markersize=10,
                         linestyle="None")
                if final_plot:
                    plt.plot([], [], label=" ", linestyle="None", marker="", color=(0, 0, 0), )
                    if iteration[-1] == 1000:  # and loss_func_name == "square_loss":
                        plt.plot([], [], label=" ", linestyle="None", marker="", color=(0, 0, 0), )

            for b_index, b in enumerate(batchsize):
                m_save_time = []
                m_save_distance = []
                for m_index, m in enumerate(iteration):

                    acces_dict = name_algo + str(n_samples_s) + "_" + str(n_samples_s) + str(b) + str(m)

                    if name_dataset == "same_graph":
                        path_pickle = "./" + pickle_path + "/" + name_dataset + "/" + name_algo + "/" + str(
                            n_samples_s) + \
                                      "_" + str(n_samples_s) + "_" + loss_func_name + "__" + str(
                            noise_graph) + "_" + \
                                      str(b) + "_" + str(m) + "_" + pickle_name + ".pickle"
                    elif name_algo == "sampled_gromov_no_KL_e_constant":
                        path_pickle = "./" + pickle_path + "/" + name_dataset + "/" + name_algo + "/" + str(
                            n_samples_s) + \
                                      "_" + str(n_samples_s) + "_" + loss_func_name + "_" + str(b) + "_" + str(
                            m) + "_3" + ".pickle"
                    else:
                        path_pickle = "./" + pickle_path + "/" + name_dataset + "/" + name_algo + "/" + str(
                            n_samples_s) + \
                                      "_" + str(n_samples_s) + "_" + loss_func_name + "_" + str(b) + "_" + str(
                            m) + pickle_name + ".pickle"

                    if path.exists(path_pickle):
                        with open(path_pickle, "rb") as pickle_in:
                            params[acces_dict] = pickle.load(pickle_in)
                            if same_param is None:
                                same_param = params[acces_dict]
                            elif params[acces_dict]["name_algo"] != "uniform":
                                # Use this to check if every run use the same parameter
                                for test in ["n_iter_algo", "rdm_seed", "same_space"]:
                                    if same_param[test] != params[acces_dict][test]:
                                        print(params[acces_dict])
                                        pass
                                        print("Warning", test, same_param[test], params[acces_dict][test])
                                    # assert same_param[test] == params[acces_dict][test]
                        time_list_only_W_approx = params[acces_dict]["time_list_only_W_approx"]
                        time_list_only_T = params[acces_dict]["time_list_only_T"]
                        time_list_only_W = params[acces_dict]["time_list_only_W"]
                        time_list = time_list_only_T + time_list_only_W
                        W_distance_list_approx = params[acces_dict]["W_distance_list_approx"]
                        W_distance_list = params[acces_dict]["W_distance_list"]

                        label = None
                        markersize = 10

                        if b == 1 and name_algo == "sampled_gromov_sliced" and only_PGW:
                            markersize = 10
                            marker = "X"
                            # print(1)
                            # print((1 - (m_index / (len(iteration) - 1))) ** 0.45 / 1.3)
                            color = (1, (1 - (m_index / (len(iteration) - 1))) ** 0.45 / 1.3, 0)
                            # color = (1, (1 - m_index / (len(iteration) - 1)) / 1.2, 0)
                            if min_plot:
                                W_distance_list_ = np.min(W_distance_list)
                                time_list_only_T_ = time_list_only_T[np.argmin(W_distance_list)]
                            else:
                                W_distance_list_ = np.mean(W_distance_list)
                                time_list_only_T_ = np.mean(time_list_only_T)

                            plt.plot(time_list_only_T_, W_distance_list_,
                                     marker=marker,
                                     color=color,
                                     label=None,
                                     markersize=markersize,
                                     linestyle="None")
                            plt.plot([np.mean(time_list_only_T), np.sum(time_list_only_T)],
                                     [np.mean(W_distance_list), np.min(W_distance_list)],
                                     color=color,
                                     marker=None,
                                     linestyle="dotted")
                            plt.plot(np.sum(time_list_only_T), np.min(W_distance_list),
                                     color=color,
                                     marker="x",
                                     linestyle=None)

                        else:
                            markersize = 10
                            marker = (b_index + 2, 0, 0)
                            color = (1, (1 - (m_index / (len(iteration) - 1))) ** 0.45 / 1.3, 0)

                            # color = 1, ((1 - m_index / (len(iteration) - 1))) / 1.5, 0)
                            if name_algo == "sampled_gromov_e_constant" or name_algo == "sampled_gromov_no_KL_e_constant":
                                # if plot:
                                if m_index == 0:
                                    plt.plot([None], [None],
                                             marker=marker,
                                             color=(0, 0, 0),
                                             label=str(b) + " sample" + ((b != 1) * "s"),
                                             markersize=markersize,
                                             linestyle="None")
                                if b_index == (len(batchsize) - 1):
                                    if iteration[-1] == 1000 and m_index == 0:  # and loss_func_name == "loss_square":
                                        plt.plot([], [], label=" ", linestyle="None", marker="", color=(0, 0, 0), )
                                    plt.plot([None], [None],
                                             marker="s",  # (3, 0, 0),
                                             color=color,
                                             label=str(m) + " iterations",
                                             markersize=8, )
                                    # linestyle="None")

                                # if b_index == 2 and m_index == 2:
                                #     label = "$SGW^{KL}$"
                            else:
                                if only_PGW:
                                    continue

                                    # marker = (b_index + 2, 0, 0)
                                    # color = (1, ((1 - m_index / (len(iteration) - 1))) / 1.5, 0)
                                    #
                                    # if m_index == 0:
                                    #     plt.plot([None], [None],
                                    #              marker=marker,
                                    #              color=(0, 0, 0),
                                    #              label=str(b) + " sample" + ((b != 1) * "s"),
                                    #              markersize=markersize,
                                    #              linestyle="None")
                                    # if b_index == (len(batchsize) - 1):
                                    #     plt.plot([None], [None],
                                    #              marker=(3, 0, 0),
                                    #              color=color,
                                    #              label=str(m) + " iterations",
                                    #              markersize=markersize,
                                    #              linestyle="None")
                            #     marker = (b_index + 2, 0, 0)
                            #     color = (0, 0, 1 - m_index / (len(iteration) - 1))
                            #     if b_index == 2 and m_index == 2:
                            #         label = "$SGW$"
                            if min_plot:
                                W_distance_list_ = np.min(W_distance_list)
                                time_list_only_T_ = time_list_only_T[np.argmin(W_distance_list)]
                            else:
                                W_distance_list_ = np.mean(W_distance_list)
                                time_list_only_T_ = np.mean(time_list_only_T)
                            m_save_time.append(time_list_only_T_)
                            m_save_distance.append(W_distance_list_)
                            plt.plot(time_list_only_T_, W_distance_list_,
                                     marker=marker,
                                     color=color,
                                     label=label,
                                     markersize=markersize,
                                     linestyle="None")
                    else:
                        params[acces_dict] = None
                plt.plot(m_save_time, m_save_distance,
                         "-",
                         color=(1, 0.5, 0),
                         markersize=3,
                         linewidth=0.5)

        else:
            acces_dict = name_algo + str(n_samples_s) + "_" + str(n_samples_s)

            if name_dataset == "same_graph":
                path_pickle = "./" + pickle_path + "/" + name_dataset + "/" + name_algo + "/" + str(
                    n_samples_s) + \
                              "_" + str(n_samples_s) + "_" + loss_func_name + "__" + str(noise_graph) + "_" + \
                              pickle_name + ".pickle"
            else:
                path_pickle = "./" + pickle_path + "/" + name_dataset + "/" + name_algo + "/" + str(
                    n_samples_s) + \
                              "_" + str(n_samples_s) + "_" + loss_func_name + pickle_name + ".pickle"

            if path.exists(path_pickle):
                with open(path_pickle, "rb") as pickle_in:

                    params[acces_dict] = pickle.load(pickle_in)
                    if same_param is None:
                        same_param = params[acces_dict]
                    elif params[acces_dict]["name_algo"] != "uniform":
                        # Use this to check if every run use the same parameter
                        for test in ["n_iter_algo", "rdm_seed", "same_space"]:
                            if same_param[test] != params[acces_dict][test]:
                                print(params[acces_dict])
                                pass
                                print("Warning", test, same_param[test], params[acces_dict][test])
                            # assert same_param[test] == params[acces_dict][test]
                time_list_only_W_approx = params[acces_dict]["time_list_only_W_approx"]
                time_list_only_T = params[acces_dict]["time_list_only_T"]
                time_list_only_W = params[acces_dict]["time_list_only_W"]
                time_list = time_list_only_T + time_list_only_W
                W_distance_list_approx = params[acces_dict]["W_distance_list_approx"]
                W_distance_list = params[acces_dict]["W_distance_list"]
                label = name_algo_legend
                # if name_algo_legend[0] == "$":
                #     marker = name_algo_legend
                # else:
                #     marker = "$" + name_algo_legend + "$"
                # markersize = 50
                # color = (0.5,1,0.5)

                if name_algo == "S_GWL":
                    color = (0, 0, 0)
                    marker = "+"
                elif name_algo == "e_gromov_KL":
                    color = (0, 0, 1)
                elif name_algo == "gromov":
                    color = (0, 1, 0)
                    marker = "+"
                elif name_algo == "sampled_gromov":
                    color = (0, 1, 0.5)
                    marker = "+"

                if not name_algo in ["uniform", "identity"]:
                    if min_plot:
                        W_distance_list_ = np.min(W_distance_list)
                        time_list_only_T_ = time_list_only_T[np.argmin(W_distance_list)]
                    else:
                        W_distance_list_ = np.mean(W_distance_list)
                        time_list_only_T_ = np.mean(time_list_only_T)
                    plt.plot(np.mean(time_list_only_T_), np.mean(W_distance_list_),
                             marker=marker,
                             label=label,
                             markersize=markersize,
                             linestyle="None",
                             color=color)
                else:
                    label = name_algo_legend
                    if name_algo == "uniform":
                        # color = (1, 1/1.4, 0)
                        color = (219. / 255., 112. / 255., 147. / 255.)
                        entropy_string = ["Low $\epsilon$", "Medium $\epsilon$", "High $\epsilon$"]
                        entropy_color = [(0, 1, 0), (0, 0.5, 0.5), (0, 0, 1)]
                        for ent in range(3):
                            plt.plot([], [],
                                     marker="s",
                                     label=entropy_string[ent],
                                     color=entropy_color[ent],
                                     markersize=8)
                    else:
                        color = (1, 0, 0)
                    if min_plot:
                        W_distance_list_ = np.min(W_distance_list)
                        time_list_only_T_ = time_list_only_T[np.argmin(W_distance_list)]
                    else:
                        W_distance_list_ = np.mean(W_distance_list)
                        time_list_only_T_ = np.mean(time_list_only_T)
                    plt.axhline(y=W_distance_list_, color=color, linestyle="-", label=label)
                    # plt.plot([1e-2, 1e4], [W_distance_list_] * 2,
                    #          label=label,
                    #          markersize=markersize,
                    #          color=color)
            else:
                # print("outer")
                params[acces_dict] = None

    plt.xscale("log")

    if final_plot:
        plt.xlabel("Computational time (s)", fontsize=16)
        plt.ylabel("Gromov Wasserstein value: $\mathcal{E}$ (T)", fontsize=16)
        plt.xticks(fontsize=13)
        plt.yticks(fontsize=13)
        from matplotlib.legend_handler import HandlerLine2D
        import matplotlib.lines

        class SymHandler(HandlerLine2D):
            def create_artists(self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans):
                xx = 0.6 * height
                return super(SymHandler, self).create_artists(legend, orig_handle, xdescent, xx, width, height,
                                                              fontsize, trans)

        plt.legend(handler_map={matplotlib.lines.Line2D: SymHandler()},
                   bbox_to_anchor=(-0.12, 0.9, 1.2, 0.3), ncol=5, handleheight=2, labelspacing=0,
                   # bbox_to_anchor=(-0.12, 0.95, 1.25, 0.3), ncol=5, handleheight=2, labelspacing=0,
                   prop={'size': 13.5},
                   frameon=False)
    else:
        plt.xlabel("Computational time (s)")
        plt.ylabel("Gromov Wasserstein value: $\mathcal{E}$ (T)")
        # plt.xticks(fontsize=16)
        plt.legend()
    if save:
        if "sampled_gromov_no_KL_e_constant" in names_algo:
            plt.savefig(
                "./figure/Analysis_hyper/noKL_" + str(min_plot) + "_" + loss_func_name + "_" + str(n_samples_s) + "_" +
                name_dataset + ".pdf", bbox_inches="tight")
        else:
            plt.savefig(
                "./figure/Analysis_hyper/" + str(min_plot) + "_" + loss_func_name + "_" + str(n_samples_s) + "_" +
                name_dataset + ".pdf", bbox_inches="tight")
    plt.show()


def hyperparameter_analysis_3(n_samples_s=500,
                              name_dataset="gaussian",
                              pickle_path="pickle_compare",
                              names_algo=["sampled_gromov"],
                              names_algo_legend=["sampled_gromov"],
                              pickle_name="",
                              noise_graph=0.1,
                              loss_func_name="1_loss",
                              figsize=(10, 5),
                              save=False,
                              entropy=[0.1],
                              batchsize=[1],
                              iteration=[1000],
                              log_scale=False,
                              min_plot=False,
                              only_PGW=True,
                              final_plot=True,
                              which_plot=0):
    grey_color = (0.9, 0.9, 0.9)
    fig = plt.figure(0, figsize=figsize)
    already_plot_SGW_legend = False
    if name_dataset in ["gaussian_graph", "same_graph", "gaussian_point_graph"]:
        # Can't compare sliced with other.
        try:
            names_algo.remove("sliced_gromov")
            names_algo_legend.remove("Sliced GW")
        except:
            pass
    params = {}
    same_param = None
    markersize = 10
    for name_algo, name_algo_legend in zip(names_algo, names_algo_legend):
        if name_algo in ["e_gromov", "e_gromov_KL"]:  # "e_gromov_KL"
            m_save_time = []
            m_save_distance = []
            for e_index, e in enumerate(entropy):
                acces_dict = name_algo + str(n_samples_s) + "_" + str(n_samples_s) + str(e)

                if name_dataset == "same_graph":
                    path_pickle = "./" + pickle_path + "/" + name_dataset + "/" + name_algo + "/" + str(
                        n_samples_s) + "_" + str(n_samples_s) + "_" + loss_func_name + "__" + str(noise_graph) + "_" + \
                                  str(e) + "__" + pickle_name + ".pickle"
                else:
                    path_pickle = "./" + pickle_path + "/" + name_dataset + "/" + name_algo + "/" + str(
                        n_samples_s) + "_" + str(n_samples_s) + "_" + loss_func_name + "_" + str(
                        e) + "_" + pickle_name + ".pickle"
                if path.exists(path_pickle):
                    with open(path_pickle, "rb") as pickle_in:
                        params[acces_dict] = pickle.load(pickle_in)
                        if same_param is None:
                            same_param = params[acces_dict]
                        elif params[acces_dict]["name_algo"] != "uniform":
                            # Use this to check if every run use the same parameter
                            for test in ["n_iter_algo", "rdm_seed", "same_space"]:
                                if same_param[test] != params[acces_dict][test]:
                                    print(params[acces_dict])
                                    pass
                                    print("Warning", test, same_param[test], params[acces_dict][test])
                    time_list_only_W_approx = params[acces_dict]["time_list_only_W_approx"]
                    time_list_only_T = params[acces_dict]["time_list_only_T"]
                    time_list_only_W = params[acces_dict]["time_list_only_W"]
                    time_list = time_list_only_T + time_list_only_W
                    W_distance_list_approx = params[acces_dict]["W_distance_list_approx"]
                    W_distance_list = params[acces_dict]["W_distance_list"]
                    color = (0, 1 - e_index / (len(entropy) - 1), e_index / (len(entropy) - 1))
                    # print(name_algo, color, entropy[e_index])
                    label = None
                    if name_algo == "e_gromov":
                        marker = 'o'
                    else:
                        marker = 'P'
                    if e_index == 0:
                        if name_algo == "e_gromov":
                            label = "$EGW$"
                            if which_plot == 2:
                                plt.plot([], [],
                                         marker=marker,
                                         color=(0, 0, 0),
                                         label=label,
                                         markersize=markersize,
                                         linestyle="None")
                        else:
                            label = "$EGW^{KL}$"
                            if which_plot == 2:
                                plt.plot([], [],
                                         marker=marker,
                                         color=(0, 0, 0),
                                         label=label,
                                         markersize=markersize,
                                         linestyle="None")
                    markersize = 10
                    label = None
                    if min_plot:
                        W_distance_list_ = np.min(W_distance_list)
                        time_list_only_T_ = time_list_only_T[np.argmin(W_distance_list)]
                    else:
                        W_distance_list_ = np.mean(W_distance_list)
                        # print("std", np.std(W_distance_list))
                        time_list_only_T_ = np.mean(time_list_only_T)
                    m_save_time.append(time_list_only_T_)
                    m_save_distance.append(W_distance_list_)
                    if which_plot != 2:
                        color = grey_color
                    plt.plot(time_list_only_T_, W_distance_list_,
                             marker=marker,
                             color=color,
                             label=label,
                             markersize=markersize,
                             linestyle="None")
                else:
                    params[acces_dict] = None
            if which_plot != 2:
                color = grey_color
            else:
                color = (0, 0.5, 0.5)
            plt.plot(m_save_time,
                     m_save_distance,
                     "-",
                     color=color,
                     markersize=3,
                     linewidth=0.5)

        elif name_algo in ["S_GWL"]:
            m_save_time = []
            m_save_distance = []
            for e_index, e in enumerate(entropy):
                acces_dict = name_algo + str(n_samples_s) + "_" + str(n_samples_s) + str(e)

                if name_dataset == "same_graph":
                    path_pickle = "./" + pickle_path + "/" + name_dataset + "/" + name_algo + "/" + str(
                        n_samples_s) + "_" + str(n_samples_s) + "_" + loss_func_name + "__" + str(noise_graph) + "_" + \
                                  str(e) + "__" + pickle_name + ".pickle"
                else:
                    path_pickle = "./" + pickle_path + "/" + name_dataset + "/" + name_algo + "/" + str(
                        n_samples_s) + "_" + str(n_samples_s) + "_" + loss_func_name + "_" + str(
                        e) + "_" + pickle_name + ".pickle"
                if path.exists(path_pickle):
                    with open(path_pickle, "rb") as pickle_in:
                        params[acces_dict] = pickle.load(pickle_in)
                        if same_param is None:
                            same_param = params[acces_dict]
                        elif params[acces_dict]["name_algo"] != "uniform":
                            # Use this to check if every run use the same parameter
                            for test in ["n_iter_algo", "rdm_seed", "same_space"]:
                                if same_param[test] != params[acces_dict][test]:
                                    print(params[acces_dict])
                                    pass
                                    print("Warning", test, same_param[test], params[acces_dict][test])
                    time_list_only_W_approx = params[acces_dict]["time_list_only_W_approx"]
                    time_list_only_T = params[acces_dict]["time_list_only_T"]
                    time_list_only_W = params[acces_dict]["time_list_only_W"]
                    time_list = time_list_only_T + time_list_only_W
                    W_distance_list_approx = params[acces_dict]["W_distance_list_approx"]
                    W_distance_list = params[acces_dict]["W_distance_list"]
                    color = (0, 1 - e_index / (len(entropy) - 1), e_index / (len(entropy) - 1))

                    marker = "*"
                    # print(e_index, e)
                    if not already_plot_SGW_legend and final_plot:
                        already_plot_SGW_legend = True
                        if which_plot == 2:
                            plt.plot([], [],
                                     marker=marker,
                                     color=(0, 0, 0),
                                     label=name_algo_legend,
                                     markersize=markersize,
                                     linestyle="None")
                    # markersize = 10
                    label = None
                    # if not final_plot:
                    #     label = name_algo_legend
                    color = (0, 1 - e_index / (len(entropy) - 1), e_index / (len(entropy) - 1))
                    # print("SGWL", color, entropy[e_index])

                    if min_plot:
                        W_distance_list_ = np.min(W_distance_list)
                        time_list_only_T_ = time_list_only_T[np.argmin(W_distance_list)]
                    else:
                        # print(np.std(W_distance_list))
                        W_distance_list_ = np.mean(W_distance_list)
                        time_list_only_T_ = np.mean(time_list_only_T)
                    m_save_time.append(time_list_only_T_)
                    m_save_distance.append(W_distance_list_)
                    if which_plot != 2:
                        color = grey_color
                    plt.plot(time_list_only_T_, W_distance_list_,
                             marker=marker,
                             color=color,
                             label=label,
                             markersize=markersize,
                             linestyle="None")
                else:
                    params[acces_dict] = None
            # acces_dict = "gromov" + str(n_samples_s) + "_" + str(n_samples_s)
            # if params[acces_dict] is not None:
            #     m_save_time = [np.mean(params[acces_dict]["time_list_only_T"])] + m_save_time
            #     m_save_distance = [np.mean(params[acces_dict]["W_distance_list"])] + m_save_distance
            if which_plot != 2:
                color = grey_color
            else:
                color = (0, 0.5, 0.5)
            plt.plot(m_save_time,
                     m_save_distance,
                     "-",
                     color=color,
                     markersize=3,
                     linewidth=0.5)

        elif name_algo in ["sampled_gromov_e_constant", "sampled_gromov_sliced", "sampled_gromov_no_KL_e_constant"]:
            if (name_algo == "sampled_gromov_e_constant" or name_algo == "sampled_gromov_no_KL_e_constant"):
                if which_plot == 1:
                    plt.plot([], [],
                             marker="",
                             color=(0, 0, 0),
                             label="$SaGroW^{KL}$" if name_algo == "sampled_gromov_e_constant" else "$SaGroW$",
                             markersize=2,
                             linestyle="None")
                if which_plot == 0:
                    plt.plot([], [],
                             marker="X",
                             color=(0, 0, 0),
                             label="PoGroW",
                             markersize=10,
                             linestyle="None")
                    plt.plot([], [],
                             marker="x",
                             color=(0, 0, 0),
                             label="Minimum over 10 runs",
                             markersize=10,
                             linestyle="None")
                # if final_plot:
                #     plt.plot([], [], label=" ", linestyle="None", marker="", color=(0, 0, 0), )
                #     if iteration[-1] == 1000:  # and loss_func_name == "square_loss":
                #         plt.plot([], [], label=" ", linestyle="None", marker="", color=(0, 0, 0), )

            for b_index, b in enumerate(batchsize):
                m_save_time = []
                m_save_distance = []
                for m_index, m in enumerate(iteration):

                    acces_dict = name_algo + str(n_samples_s) + "_" + str(n_samples_s) + str(b) + str(m)

                    if name_dataset == "same_graph":
                        path_pickle = "./" + pickle_path + "/" + name_dataset + "/" + name_algo + "/" + str(
                            n_samples_s) + \
                                      "_" + str(n_samples_s) + "_" + loss_func_name + "__" + str(
                            noise_graph) + "_" + \
                                      str(b) + "_" + str(m) + "_" + pickle_name + ".pickle"
                    elif name_algo == "sampled_gromov_no_KL_e_constant" and which_plot == 1:
                        path_pickle = "./" + pickle_path + "/" + name_dataset + "/" + name_algo + "/" + str(
                            n_samples_s) + \
                                      "_" + str(n_samples_s) + "_" + loss_func_name + "_" + str(b) + "_" + str(
                            m) + "_3" + ".pickle"
                    else:
                        path_pickle = "./" + pickle_path + "/" + name_dataset + "/" + name_algo + "/" + str(
                            n_samples_s) + \
                                      "_" + str(n_samples_s) + "_" + loss_func_name + "_" + str(b) + "_" + str(
                            m) + pickle_name + ".pickle"

                    if path.exists(path_pickle):
                        with open(path_pickle, "rb") as pickle_in:
                            params[acces_dict] = pickle.load(pickle_in)
                            if same_param is None:
                                same_param = params[acces_dict]
                            elif params[acces_dict]["name_algo"] != "uniform":
                                # Use this to check if every run use the same parameter
                                for test in ["n_iter_algo", "rdm_seed", "same_space"]:
                                    if same_param[test] != params[acces_dict][test]:
                                        print(params[acces_dict])
                                        pass
                                        print("Warning", test, same_param[test], params[acces_dict][test])
                                    # assert same_param[test] == params[acces_dict][test]
                        time_list_only_W_approx = params[acces_dict]["time_list_only_W_approx"]
                        time_list_only_T = params[acces_dict]["time_list_only_T"]
                        time_list_only_W = params[acces_dict]["time_list_only_W"]
                        time_list = time_list_only_T + time_list_only_W
                        W_distance_list_approx = params[acces_dict]["W_distance_list_approx"]
                        W_distance_list = params[acces_dict]["W_distance_list"]

                        label = None
                        markersize = 10

                        if b == 1 and name_algo == "sampled_gromov_sliced" and only_PGW:
                            markersize = 10
                            marker = "X"
                            # print(1)
                            # print((1 - (m_index / (len(iteration) - 1))) ** 0.45 / 1.3)
                            if which_plot == 0:
                                color = (1, (1 - (m_index / (len(iteration) - 1))) ** 0.45 / 1.3, 0)
                            else:
                                color = grey_color
                            # color = (1, (1 - m_index / (len(iteration) - 1)) / 1.2, 0)
                            if min_plot:
                                W_distance_list_ = np.min(W_distance_list)
                                time_list_only_T_ = time_list_only_T[np.argmin(W_distance_list)]
                            else:
                                W_distance_list_ = np.mean(W_distance_list)
                                time_list_only_T_ = np.mean(time_list_only_T)

                            if which_plot == 0:
                                alpha = 0.15
                                plt.vlines(time_list_only_T_,
                                                 W_distance_list_ - np.std(W_distance_list),
                                                 W_distance_list_ + np.std(W_distance_list),
                                                 alpha=alpha,
                                                 # marker="_",
                                                 color=color,)
                                plt.plot([time_list_only_T_, time_list_only_T_],
                                         [W_distance_list_ + np.std(W_distance_list),
                                          W_distance_list_ - np.std(W_distance_list)],
                                         alpha=alpha,
                                         color=color,
                                         marker="_")
                            plt.plot(time_list_only_T_, W_distance_list_,
                                     marker=marker,
                                     color=color,
                                     label=None,
                                     markersize=markersize,
                                     linestyle="None")
                            plt.plot([np.mean(time_list_only_T), np.sum(time_list_only_T)],
                                     [np.mean(W_distance_list), np.min(W_distance_list)],
                                     color=color,
                                     marker=None,
                                     linestyle="dotted")
                            plt.plot(np.sum(time_list_only_T), np.min(W_distance_list),
                                     color=color,
                                     marker="x",
                                     linestyle=None)


                        else:
                            markersize = 10
                            marker = (b_index + 3, 0, 0)
                            # if which_plot == 1:
                            color = (1, (1 - (m_index / (len(iteration) - 1))) ** 0.45 / 1.3, 0)
                            # else:
                            #     color = grey_color
                            # color = (1, ((1 - m_index / (len(iteration) - 1))) / 1.5, 0)
                            if (
                                    name_algo == "sampled_gromov_e_constant" or name_algo == "sampled_gromov_no_KL_e_constant"):
                                if m_index == 0 and which_plot == 1:
                                    plt.plot([None], [None],
                                             marker=marker,
                                             color=(0, 0, 0),
                                             label=str(b) + " sample" + ((b != 1) * "s"),
                                             markersize=markersize,
                                             linestyle="None")
                                if b_index == (len(batchsize) - 1) and which_plot < 2:
                                    if iteration[-1] == 1000 and m_index == 0:  # and loss_func_name == "loss_square":
                                        plt.plot([], [], label=" ", linestyle="None", marker="", color=(0, 0, 0), )
                                    plt.plot([None], [None],
                                             marker="s",  # (3, 0, 0),
                                             color=color,
                                             label=str(m) + " iterations",
                                             markersize=8)

                            else:
                                if only_PGW:
                                    continue

                                    # marker = (b_index + 2, 0, 0)
                                    # color = (1, ((1 - m_index / (len(iteration) - 1))) / 1.5, 0)
                                    #
                                    # if m_index == 0:
                                    #     plt.plot([None], [None],
                                    #              marker=marker,
                                    #              color=(0, 0, 0),
                                    #              label=str(b) + " sample" + ((b != 1) * "s"),
                                    #              markersize=markersize,
                                    #              linestyle="None")
                                    # if b_index == (len(batchsize) - 1):
                                    #     plt.plot([None], [None],
                                    #              marker=(3, 0, 0),
                                    #              color=color,
                                    #              label=str(m) + " iterations",
                                    #              markersize=markersize,
                                    #              linestyle="None")
                            #     marker = (b_index + 2, 0, 0)
                            #     color = (0, 0, 1 - m_index / (len(iteration) - 1))
                            #     if b_index == 2 and m_index == 2:
                            #         label = "$SGW$"
                            if min_plot:
                                W_distance_list_ = np.min(W_distance_list)
                                time_list_only_T_ = time_list_only_T[np.argmin(W_distance_list)]
                            else:
                                W_distance_list_ = np.mean(W_distance_list)
                                time_list_only_T_ = np.mean(time_list_only_T)
                            m_save_time.append(time_list_only_T_)
                            m_save_distance.append(W_distance_list_)
                            if which_plot == 1:
                                color = (1, (1 - (m_index / (len(iteration) - 1))) ** 0.45 / 1.3, 0)
                            else:
                                color = grey_color
                            if which_plot == 1:
                                alpha = 0.15
                                plt.vlines(time_list_only_T_,
                                                 W_distance_list_ - np.std(W_distance_list),
                                                 W_distance_list_ + np.std(W_distance_list),
                                                 alpha=alpha,
                                                 # marker="_",
                                                 color=color,)
                                plt.plot([time_list_only_T_, time_list_only_T_],
                                         [W_distance_list_ + np.std(W_distance_list),
                                          W_distance_list_ - np.std(W_distance_list)],
                                         alpha=alpha,
                                         color=color,
                                         marker="_")
                            plt.plot(time_list_only_T_, W_distance_list_,
                                     marker=marker,
                                     color=color,
                                     label=label,
                                     markersize=markersize,
                                     linestyle="None")
                    else:
                        params[acces_dict] = None
                if which_plot == 1:
                    color = (1, 0.5, 0)
                else:
                    color = grey_color

                plt.plot(m_save_time, m_save_distance,
                         "-",
                         color=color,
                         markersize=3,
                         linewidth=0.5)

        elif name_algo in ["e_gromov_KL", "e_gromov", "gromov", "S_GWL"]:
            acces_dict = name_algo + str(n_samples_s) + "_" + str(n_samples_s)

            if name_dataset == "same_graph":
                path_pickle = "./" + pickle_path + "/" + name_dataset + "/" + name_algo + "/" + str(
                    n_samples_s) + \
                              "_" + str(n_samples_s) + "_" + loss_func_name + "__" + str(noise_graph) + "_" + \
                              pickle_name + ".pickle"
            else:
                path_pickle = "./" + pickle_path + "/" + name_dataset + "/" + name_algo + "/" + str(
                    n_samples_s) + \
                              "_" + str(n_samples_s) + "_" + loss_func_name + pickle_name + ".pickle"

            if path.exists(path_pickle):
                with open(path_pickle, "rb") as pickle_in:

                    params[acces_dict] = pickle.load(pickle_in)
                    if same_param is None:
                        same_param = params[acces_dict]
                    elif params[acces_dict]["name_algo"] != "uniform":
                        # Use this to check if every run use the same parameter
                        for test in ["n_iter_algo", "rdm_seed", "same_space"]:
                            if same_param[test] != params[acces_dict][test]:
                                print(params[acces_dict])
                                pass
                                print("Warning", test, same_param[test], params[acces_dict][test])
                            # assert same_param[test] == params[acces_dict][test]
                time_list_only_W_approx = params[acces_dict]["time_list_only_W_approx"]
                time_list_only_T = params[acces_dict]["time_list_only_T"]
                time_list_only_W = params[acces_dict]["time_list_only_W"]
                time_list = time_list_only_T + time_list_only_W
                W_distance_list_approx = params[acces_dict]["W_distance_list_approx"]
                W_distance_list = params[acces_dict]["W_distance_list"]
                label = name_algo_legend
                # if name_algo_legend[0] == "$":
                #     marker = name_algo_legend
                # else:
                #     marker = "$" + name_algo_legend + "$"
                # markersize = 50
                # color = (0.5,1,0.5)

                if name_algo == "S_GWL":
                    color = (0, 0, 0)
                    marker = "+"
                elif name_algo == "e_gromov_KL":
                    color = (0, 0, 1)
                elif name_algo == "gromov":
                    if which_plot == 2:
                        color = (0, 1, 0)
                    else:
                        color = grey_color
                        label = ""
                    marker = "+"
                elif name_algo == "sampled_gromov":
                    color = (0, 1, 0.5)
                    marker = "+"
                # elif name_algo == "uniform":
                #     marker = ""

                if not name_algo in ["uniform", "identity"]:
                    if min_plot:
                        W_distance_list_ = np.min(W_distance_list)
                        time_list_only_T_ = time_list_only_T[np.argmin(W_distance_list)]
                    else:
                        W_distance_list_ = np.mean(W_distance_list)
                        time_list_only_T_ = np.mean(time_list_only_T)
                    plt.plot(np.mean(time_list_only_T_), np.mean(W_distance_list_),
                             marker=marker,
                             label=label,
                             markersize=markersize,
                             linestyle="None",
                             color=color)
                # if name_algo == "gromov":
                #     # color = (1, 1/1.4, 0)
                #     color = (219. / 255., 112. / 255., 147. / 255.)
                #     entropy_string = ["Low $\epsilon$", "Medium $\epsilon$", "High $\epsilon$"]
                #     entropy_color = [(0, 1, 0), (0, 0.5, 0.5), (0, 0, 1)]
                #     for ent in range(3):
                #         plt.plot([], [],
                #                  marker="s",
                #                  label=entropy_string[ent],
                #                  color=entropy_color[ent],
                #                  markersize=8)
                # else:
                #     label = name_algo_legend
                #
                #     else:
                #         color = (1, 0, 0)
                #     if min_plot:
                #         W_distance_list_ = np.min(W_distance_list)
                #         time_list_only_T_ = time_list_only_T[np.argmin(W_distance_list)]
                #     else:
                #         W_distance_list_ = np.mean(W_distance_list)
                #         time_list_only_T_ = np.mean(time_list_only_T)
                #     plt.axhline(y=W_distance_list_, color=color, linestyle="-", label=label)
                # plt.plot([1e-2, 1e4], [W_distance_list_] * 2,
                #          label=label,
                #          markersize=markersize,
                #          color=color)
            else:
                # print("outer")
                params[acces_dict] = None

        elif name_algo in ["uniform"]:
            color = (0.9, 0.5, 0.9)
            acces_dict = name_algo + str(n_samples_s) + "_" + str(n_samples_s)

            if name_dataset == "same_graph":
                path_pickle = "./" + pickle_path + "/" + name_dataset + "/" + name_algo + "/" + str(
                    n_samples_s) + \
                              "_" + str(n_samples_s) + "_" + loss_func_name + "__" + str(noise_graph) + "_" + \
                              pickle_name + ".pickle"
            else:
                path_pickle = "./" + pickle_path + "/" + name_dataset + "/" + name_algo + "/" + str(
                    n_samples_s) + \
                              "_" + str(n_samples_s) + "_" + loss_func_name + pickle_name + ".pickle"

            if path.exists(path_pickle):
                with open(path_pickle, "rb") as pickle_in:
                    params[acces_dict] = pickle.load(pickle_in)
                    if same_param is None:
                        same_param = params[acces_dict]
                    elif params[acces_dict]["name_algo"] != "uniform":
                        # Use this to check if every run use the same parameter
                        for test in ["n_iter_algo", "rdm_seed", "same_space"]:
                            if same_param[test] != params[acces_dict][test]:
                                print(params[acces_dict])
                                pass
                                print("Warning", test, same_param[test], params[acces_dict][test])
                            # assert same_param[test] == params[acces_dict][test]
                time_list_only_T = params[acces_dict]["time_list_only_T"]
                time_list_only_W = params[acces_dict]["time_list_only_W"]
                W_distance_list = params[acces_dict]["W_distance_list"]
                label = name_algo_legend
                W_distance_list_ = np.mean(W_distance_list)
                time_list_only_T_ = np.mean(time_list_only_T)
                plt.axhline(y=W_distance_list_, color=color, linestyle="-", label=label)

    plt.xscale("log")

    if final_plot:
        plt.xlabel("Computational time (s)", fontsize=16)
        plt.ylabel("Gromov Wasserstein value: $\mathcal{E}$ (T)", fontsize=16)
        plt.xticks(fontsize=13)
        plt.yticks(fontsize=13)
        from matplotlib.legend_handler import HandlerLine2D
        import matplotlib.lines

        class SymHandler(HandlerLine2D):
            def create_artists(self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans):
                xx = 0.6 * height
                return super(SymHandler, self).create_artists(legend, orig_handle, xdescent, xx, width, height,
                                                              fontsize, trans)

        # if which_plot == 0:

        handles, labels = plt.gca().get_legend_handles_labels()
        if which_plot == 0:
            bbox_to_anchor = (-0.3, 0.9, 1.3, 0.3)  # (-0.12, 0.9, 1.2, 0.3)
            plt.plot([], [], label=" ", linestyle="None", marker="", color=(0, 0, 0))
            handles, labels = plt.gca().get_legend_handles_labels()
            order = [0, 2, 1, 3, 6, 4, 7, 5]
            plt.legend([handles[idx] for idx in order], [labels[idx] for idx in order],
                       handler_map={matplotlib.lines.Line2D: SymHandler()},
                       bbox_to_anchor=bbox_to_anchor, ncol=4, handleheight=2, labelspacing=0,
                       prop={'size': 12.5},
                       frameon=False)
        elif which_plot == 1:
            bbox_to_anchor = (-0.19, 0.9, 1.2, 0.3)  # (-0.12, 0.9, 1.2, 0.3)
            plt.plot([], [], label=" ", linestyle="None", marker="", color=(0, 0, 0))
            handles, labels = plt.gca().get_legend_handles_labels()
            order = [0, 9, 1, 4, 2, 5, 3, 6, 8, 7] #np.arange(len(labels))#
            plt.legend([handles[idx] for idx in order], [labels[idx] for idx in order],
                       handler_map={matplotlib.lines.Line2D: SymHandler()},
                       bbox_to_anchor=bbox_to_anchor, ncol=5, handleheight=2, labelspacing=0,
                       columnspacing=1.5,
                       prop={'size': 12.5},
                       frameon=False)
        elif which_plot == 2:
            bbox_to_anchor = (-0.40, 0.9, 1.2, 0.3)  # (-0.12, 0.9, 1.2, 0.3)
            bbox_to_anchor = (21, 1.1, 1.4, 0.2)  # (-0.12, 0.9, 1.2, 0.3)
            order = [5, 1, 6, 2, 4, 3, 0, 7]
            order = [2, 3, 1, 0, 4]
            import matplotlib.colors as mcolors
            import matplotlib as mpl

            def make_colormap(seq):
                """Return a LinearSegmentedColormap
                seq: a sequence of floats and RGB-tuples. The floats should be increasing
                and in the interval (0,1).
                """
                seq = [(None,) * 3, 0.0] + list(seq) + [1.0, (None,) * 3]
                # print(seq)
                cdict = {'red': [], 'green': [], 'blue': []}
                for i, item in enumerate(seq):
                    if isinstance(item, float):
                        r1, g1, b1 = seq[i - 1]
                        r2, g2, b2 = seq[i + 1]
                        cdict['red'].append([item, r1, r2])
                        cdict['green'].append([item, g1, g2])
                        cdict['blue'].append([item, b1, b2])
                return mcolors.LinearSegmentedColormap('CustomMap', cdict)

            # c = mcolors.ColorConverter().to_rgb
            rvb = make_colormap(
                # [(0, 1, 0), (0, 0.5, 0.5), float(10**(-3)), (0, 0.5, 0.5), (0, 0, 1), float(10**2), (0, 0, 1), 10**2])
            [(0, 1, 0), (0, 0.5, 0.5), 0.33, (0, 0.5, 0.5), (0, 0, 1), 0.66, (0, 0, 1)])
            norm = mpl.colors.LogNorm(vmin=0.001, vmax=100)
            # norm = plt.Normalize(vmin=0, vmax=1)
            sm = plt.cm.ScalarMappable(cmap=rvb, norm=norm)

            sm._A = []
            # cbaxes = plt.add_axes
            cbaxes = fig.add_axes([0.15, 0.2, 0.03, 0.6])
            # , shrink = 0.8, anchor = (0, 0.5)
            cbar = plt.colorbar(sm, cax=cbaxes)#, pad=-0.5)#, norm=mpl.colors.Normalize(vmin=-0.5, vmax=1.5))
            cbar.set_label('Epsilon : $\epsilon$', rotation=90, labelpad=-70)
            # cbar.set_clim(-2.0, 2.0)
            # plt.clim(-4, 4)
            plt.legend([handles[idx] for idx in order], [labels[idx] for idx in order],
                       handler_map={matplotlib.lines.Line2D: SymHandler()},
                       bbox_to_anchor=bbox_to_anchor, ncol=5, handleheight=2, labelspacing=0,
                       prop={'size': 12.5},
                       frameon=False)
        # else:
        #     order = np.arange(len(labels))




    else:
        plt.xlabel("Computational time (s)")
        plt.ylabel("Gromov Wasserstein value: $\mathcal{E}$ (T)")
        # plt.xticks(fontsize=16)
        plt.legend()
    if save:
        if "sampled_gromov_no_KL_e_constant" in names_algo:
            plt.savefig(
                "./figure/Analysis_hyper/noKL_" + str(min_plot) + "_" + loss_func_name + "_" + str(n_samples_s) + "_" +
                name_dataset + "_" + str(which_plot) + ".pdf", bbox_inches="tight")
        else:
            plt.savefig(
                "./figure/Analysis_hyper/" + str(min_plot) + "_" + loss_func_name + "_" + str(n_samples_s) + "_" +
                name_dataset + "_" + str(which_plot) + ".pdf", bbox_inches="tight")
    plt.show()


def analyse_data_hyper(name_dataset="gaussian",
                       pickle_path="pickle_compare",
                       names_algo=["sampled_gromov"],
                       names_algo_legend=["sampled_gromov"],
                       pickle_name="",
                       n_samples_s=[50, 100, 200],
                       n_samples_t=[],
                       noise_graph=0.1,
                       loss_func_name="1_loss",
                       best_T_not_needed=True,
                       log_scale=True,
                       figsize=(10, 5),
                       save=False,
                       entropy=[0.1],
                       entropy_SGWL=[0.1],
                       batchsize=[1],
                       iteration=[1000],
                       paper_plot=False,
                       show_legend=True,
                       legend_SGW=False,
                       markersize=10,
                       linestyle="dotted"
                       ):
    n_samples_s = np.array(n_samples_s)
    if name_dataset in ["gaussian_graph", "same_graph"]:
        try:
            names_algo.remove("sliced_gromov")
            names_algo_legend.remove("SGW")
        except:
            pass

    if name_dataset in ["gaussian_graph", "gaussian_point_graph"]:
        try:
            names_algo.remove("identity")
            names_algo_legend.remove("Identity")
        except:
            pass

    fontsize_legend = 17
    fontsize_axis = 17
    fontsize_ticks = 14

    if not n_samples_t:
        n_samples_t = n_samples_s

    if paper_plot:
        # I found manually the best param for each dataset.
        # Print the average best value over all the iteration, this give the best parameter.
        if name_dataset == "same_graph":
            entropy = [0.005]
        elif name_dataset == "gaussian_graph":
            entropy = [0.01]
        elif name_dataset == "gaussian_point_graph":
            entropy = [0.005]

        if name_dataset == "same_graph":
            entropy_SGWL = [100]
        elif name_dataset == "gaussian_graph":
            entropy_SGWL = [1]
        elif name_dataset == "gaussian_point_graph":
            entropy_SGWL = [10]

        if name_dataset == "same_graph":
            best_param_m = 500
            best_param_b = 10
        elif name_dataset == "gaussian_graph":
            best_param_m = 500
            best_param_b = 10
        elif name_dataset == "gaussian_point_graph":
            best_param_m = 100
            best_param_b = 10

    params = {}

    same_param = None

    for name_algo in names_algo:
        for i in range(len(n_samples_s)):
            if name_algo in ["e_gromov", "e_gromov_KL"]:
                for e in entropy:
                    acces_dict = name_algo + str(n_samples_s[i]) + "_" + str(n_samples_t[i]) + str(e)
                    if name_dataset == "same_graph":
                        path_pickle = "./" + pickle_path + "/" + name_dataset + "/" + name_algo + "/" + str(
                            n_samples_s[i]) + \
                                      "_" + str(n_samples_t[i]) + "_" + loss_func_name + "__" + str(noise_graph) + "_" + \
                                      str(e) + "__" + pickle_name + ".pickle"
                    else:
                        path_pickle = "./" + pickle_path + "/" + name_dataset + "/" + name_algo + "/" + str(
                            n_samples_s[i]) + \
                                      "_" + str(n_samples_t[i]) + "_" + loss_func_name + "_" + str(
                            e) + "_" + pickle_name + ".pickle"
                    if path.exists(path_pickle):
                        with open(path_pickle, "rb") as pickle_in:

                            params[acces_dict] = pickle.load(pickle_in)
                            if same_param is None:
                                same_param = params[acces_dict]
                            elif params[acces_dict]["name_algo"] != "uniform":
                                # Use this to check if every run use the same parameter
                                for test in ["n_iter_algo", "rdm_seed", "same_space"]:
                                    if same_param[test] != params[acces_dict][test]:
                                        print(params[acces_dict])
                                        pass
                                        print("Warning", test, same_param[test], params[acces_dict][test])
                                    # assert same_param[test] == params[acces_dict][test]
                    else:
                        # print(path_pickle)
                        # print("outer")
                        params[acces_dict] = None
            elif name_algo in ["S_GWL"]:
                for e in entropy_SGWL:
                    acces_dict = name_algo + str(n_samples_s[i]) + "_" + str(n_samples_t[i]) + str(e)
                    if name_dataset == "same_graph":
                        path_pickle = "./" + pickle_path + "/" + name_dataset + "/" + name_algo + "/" + str(
                            n_samples_s[i]) + \
                                      "_" + str(n_samples_t[i]) + "_" + loss_func_name + "__" + str(noise_graph) + "_" + \
                                      str(e) + "__" + pickle_name + ".pickle"
                    else:
                        path_pickle = "./" + pickle_path + "/" + name_dataset + "/" + name_algo + "/" + str(
                            n_samples_s[i]) + \
                                      "_" + str(n_samples_t[i]) + "_" + loss_func_name + "_" + str(
                            e) + "_" + pickle_name + ".pickle"
                    if path.exists(path_pickle):
                        with open(path_pickle, "rb") as pickle_in:

                            params[acces_dict] = pickle.load(pickle_in)
                            if same_param is None:
                                same_param = params[acces_dict]
                            elif params[acces_dict]["name_algo"] != "uniform":
                                # Use this to check if every run use the same parameter
                                for test in ["n_iter_algo", "rdm_seed", "same_space"]:
                                    if same_param[test] != params[acces_dict][test]:
                                        print(params[acces_dict])
                                        pass
                                        print("Warning", test, same_param[test], params[acces_dict][test])
                                    # assert same_param[test] == params[acces_dict][test]
                    else:
                        # print(path_pickle)
                        # print("outer")
                        params[acces_dict] = None
            elif name_algo in ["sampled_gromov_e_constant", "sampled_gromov_sliced", "sampled_gromov_no_KL_e_constant"]:
                for b in batchsize:
                    for m in iteration:
                        acces_dict = name_algo + str(n_samples_s[i]) + "_" + str(n_samples_t[i]) + str(b) + str(m)

                        if name_dataset == "same_graph":
                            path_pickle = "./" + pickle_path + "/" + name_dataset + "/" + name_algo + "/" + str(
                                n_samples_s[i]) + \
                                          "_" + str(n_samples_t[i]) + "_" + loss_func_name + "__" + str(
                                noise_graph) + "_" + \
                                          str(b) + "_" + str(m) + "_" + pickle_name + ".pickle"
                        elif name_algo == "sampled_gromov_no_KL_e_constant":
                            path_pickle = "./" + pickle_path + "/" + name_dataset + "/" + name_algo + "/" + str(
                                n_samples_s[i]) + \
                                          "_" + str(n_samples_t[i]) + "_" + loss_func_name + "_" + str(b) + "_" + str(
                                m) + "_3" + ".pickle"
                        else:
                            path_pickle = "./" + pickle_path + "/" + name_dataset + "/" + name_algo + "/" + str(
                                n_samples_s[i]) + \
                                          "_" + str(n_samples_t[i]) + "_" + loss_func_name + "_" + str(b) + "_" + str(
                                m) + pickle_name + ".pickle"
                        if path.exists(path_pickle):
                            with open(path_pickle, "rb") as pickle_in:

                                params[acces_dict] = pickle.load(pickle_in)
                                if same_param is None:
                                    same_param = params[acces_dict]
                                elif params[acces_dict]["name_algo"] != "uniform":
                                    # Use this to check if every run use the same parameter
                                    for test in ["n_iter_algo", "rdm_seed", "same_space"]:
                                        if same_param[test] != params[acces_dict][test]:
                                            print(params[acces_dict])
                                            pass
                                            print("Warning", test, same_param[test], params[acces_dict][test])
                                        # assert same_param[test] == params[acces_dict][test]
                        else:
                            # print("outer")
                            params[acces_dict] = None
            else:
                acces_dict = name_algo + str(n_samples_s[i]) + "_" + str(n_samples_t[i])

                if name_dataset == "same_graph":
                    path_pickle = "./" + pickle_path + "/" + name_dataset + "/" + name_algo + "/" + str(
                        n_samples_s[i]) + \
                                  "_" + str(n_samples_t[i]) + "_" + loss_func_name + "__" + str(noise_graph) + "_" + \
                                  pickle_name + ".pickle"
                else:
                    path_pickle = "./" + pickle_path + "/" + name_dataset + "/" + name_algo + "/" + str(
                        n_samples_s[i]) + \
                                  "_" + str(n_samples_t[i]) + "_" + loss_func_name + pickle_name + ".pickle"

                if path.exists(path_pickle):
                    with open(path_pickle, "rb") as pickle_in:

                        params[acces_dict] = pickle.load(pickle_in)
                        if same_param is None:
                            same_param = params[acces_dict]
                        elif params[acces_dict]["name_algo"] != "uniform":
                            # Use this to check if every run use the same parameter
                            for test in ["n_iter_algo", "rdm_seed", "same_space"]:
                                if same_param[test] != params[acces_dict][test]:
                                    print(params[acces_dict])
                                    pass
                                    print("Warning", test, same_param[test], params[acces_dict][test])
                                # assert same_param[test] == params[acces_dict][test]
                else:
                    # print("outer")
                    params[acces_dict] = None

    assert same_param is not None  # Maybe an error in the pickle file name

    for name_algo, name_algo_legend in zip(names_algo, names_algo_legend):
        if name_algo in ["e_gromov", "e_gromov_KL"]:
            for e in entropy:
                time_list_only_W = np.zeros((len(n_samples_s), same_param["n_iter_algo"]))
                time_list_only_T = np.zeros((len(n_samples_s), same_param["n_iter_algo"]))
                time_list_only_W_approx = np.zeros((len(n_samples_s), same_param["n_iter_algo"]))
                time_list = np.zeros((len(n_samples_s), same_param["n_iter_algo"]))
                n_samples_s_temp = n_samples_s.copy()
                mask = np.ones(len(n_samples_s), dtype=bool)
                # n_samples_s_temp = n_samples_s_temp.astype(float)
                for i in range(len(n_samples_s)):
                    acces_dict = name_algo + str(n_samples_s[i]) + "_" + str(n_samples_t[i]) + str(e)
                    # print(params[acces_dict])
                    if params[acces_dict] is None:
                        # n_samples_s_temp[i] = np.NaN
                        mask[i] = 0
                        continue
                    time_list_only_W_approx[i] = params[acces_dict]["time_list_only_W_approx"]
                    time_list_only_T[i] = params[acces_dict]["time_list_only_T"]
                    time_list_only_W[i] = params[acces_dict]["time_list_only_W"]
                    time_list[i] = time_list_only_T[i] + time_list_only_W[i]
                # linestyle = "--"
                plt.figure(0, figsize=figsize)
                if name_algo == "e_gromov":
                    color = (0, 0.5, 0.8)
                    marker = "o"
                if name_algo == "e_gromov_KL":
                    color = (0, 0, 1)#(0, 0.8, 0.5)
                    marker = "P"
                if paper_plot:
                    legend_param = ""
                else:
                    legend_param = "$_{" + str(e) + "}$"
                plt.plot(n_samples_s[mask], np.mean(time_list, axis=1)[mask], linestyle=linestyle,
                         label=name_algo_legend + legend_param, color=color, marker=marker, markersize=markersize)
                plt.fill_between(n_samples_s[mask],
                                 (np.mean(time_list, axis=1) - np.std(time_list, axis=1))[mask],
                                 (np.mean(time_list, axis=1) + np.std(time_list, axis=1))[mask],
                                 alpha=0.3, color=color)

                plt.figure(1, figsize=figsize)

                plt.plot(n_samples_s[mask], np.mean(time_list_only_T, axis=1)[mask], linestyle=linestyle,
                         label=name_algo_legend + legend_param, color=color, marker=marker, markersize=10)
                plt.fill_between(n_samples_s[mask],
                                 (np.mean(time_list_only_T, axis=1) - np.std(time_list_only_T, axis=1))[mask],
                                 (np.mean(time_list_only_T, axis=1) + np.std(time_list_only_T, axis=1))[mask],
                                 alpha=0.3, color=color)
        elif name_algo in ["S_GWL"]:
            for e in entropy_SGWL:
                time_list_only_W = np.zeros((len(n_samples_s), same_param["n_iter_algo"]))
                time_list_only_T = np.zeros((len(n_samples_s), same_param["n_iter_algo"]))
                time_list_only_W_approx = np.zeros((len(n_samples_s), same_param["n_iter_algo"]))
                time_list = np.zeros((len(n_samples_s), same_param["n_iter_algo"]))
                n_samples_s_temp = n_samples_s.copy()
                mask = np.ones(len(n_samples_s), dtype=bool)
                # n_samples_s_temp = n_samples_s_temp.astype(float)
                for i in range(len(n_samples_s)):
                    acces_dict = name_algo + str(n_samples_s[i]) + "_" + str(n_samples_t[i]) + str(e)
                    # print(params[acces_dict])
                    if params[acces_dict] is None:
                        # n_samples_s_temp[i] = np.NaN
                        mask[i] = 0
                        continue
                    time_list_only_W_approx[i] = params[acces_dict]["time_list_only_W_approx"]
                    time_list_only_T[i] = params[acces_dict]["time_list_only_T"]
                    time_list_only_W[i] = params[acces_dict]["time_list_only_W"]
                    time_list[i] = time_list_only_T[i] + time_list_only_W[i]
                # linestyle = "--"
                plt.figure(0, figsize=figsize)
                color = (0, 0, 0)
                marker = "*"
                if paper_plot:
                    legend_param = ""
                else:
                    legend_param = "$_{" + str(e) + "}$"
                plt.plot(n_samples_s[mask], np.mean(time_list, axis=1)[mask], linestyle=linestyle,
                         label=name_algo_legend + legend_param, color=color, marker=marker, markersize=10)
                plt.fill_between(n_samples_s[mask],
                                 (np.mean(time_list, axis=1) - np.std(time_list, axis=1))[mask],
                                 (np.mean(time_list, axis=1) + np.std(time_list, axis=1))[mask],
                                 alpha=0.3, color=color)

                plt.figure(1, figsize=figsize)
                plt.plot(n_samples_s[mask], np.mean(time_list_only_T, axis=1)[mask], linestyle=linestyle,
                         label=name_algo_legend + legend_param, color=color, marker=marker, markersize=10)
                plt.fill_between(n_samples_s[mask],
                                 (np.mean(time_list_only_T, axis=1) - np.std(time_list_only_T, axis=1))[mask],
                                 (np.mean(time_list_only_T, axis=1) + np.std(time_list_only_T, axis=1))[mask],
                                 alpha=0.3, color=color)
        elif name_algo in ["sampled_gromov_e_constant", "sampled_gromov_sliced", "sampled_gromov_no_KL_e_constant"]:
            for b_index, b in enumerate(batchsize):
                for m_index, m in enumerate(iteration):
                    if paper_plot:
                        if name_algo == "sampled_gromov_sliced":
                            if b != 1 or m != 100:
                                continue
                        else:
                            if b != best_param_b or m != best_param_m:
                                continue
                    # print(name_algo, b, m)
                    time_list_only_W = np.zeros((len(n_samples_s), same_param["n_iter_algo"]))
                    time_list_only_T = np.zeros((len(n_samples_s), same_param["n_iter_algo"]))
                    time_list_only_W_approx = np.zeros((len(n_samples_s), same_param["n_iter_algo"]))
                    time_list = np.zeros((len(n_samples_s), same_param["n_iter_algo"]))
                    n_samples_s_temp = n_samples_s.copy()
                    mask = np.ones(len(n_samples_s), dtype=bool)
                    # n_samples_s_temp = n_samples_s_temp.astype(float)
                    for i in range(len(n_samples_s)):
                        acces_dict = name_algo + str(n_samples_s[i]) + "_" + str(n_samples_t[i]) + str(b) + str(m)
                        # print(params[acces_dict])
                        # print(acces_dict)
                        if params[acces_dict] is None:
                            # n_samples_s_temp[i] = np.NaN
                            mask[i] = 0
                            continue
                        time_list_only_W_approx[i] = params[acces_dict]["time_list_only_W_approx"]
                        time_list_only_T[i] = params[acces_dict]["time_list_only_T"]
                        time_list_only_W[i] = params[acces_dict]["time_list_only_W"]
                        time_list[i] = time_list_only_T[i] + time_list_only_W[i]
                    # print(mask)
                    # linestyle = "-"
                    plt.figure(0, figsize=figsize)
                    if paper_plot:
                        if name_algo == "sampled_gromov_sliced":
                            color = (1, 0.5, 0)
                            marker = "X"
                        else:
                            color = (1, 0, 0)
                            marker = "s"
                    else:
                        color = (1, b_index / len(batchsize), 0)

                    plt.plot(n_samples_s[mask], np.mean(time_list, axis=1)[mask], linestyle=linestyle,
                             label=name_algo_legend, color=color, marker=marker, markersize=markersize)
                    plt.fill_between(n_samples_s[mask],
                                     (np.mean(time_list, axis=1) - np.std(time_list, axis=1))[mask],
                                     (np.mean(time_list, axis=1) + np.std(time_list, axis=1))[mask],
                                     alpha=0.3, color=color)

                    plt.figure(1, figsize=figsize)

                    plt.plot(n_samples_s[mask], np.mean(time_list_only_T, axis=1)[mask], linestyle=linestyle,
                             label=name_algo_legend, color=color, marker=marker, markersize=markersize)
                    plt.fill_between(n_samples_s[mask],
                                     (np.mean(time_list_only_T, axis=1) - np.std(time_list_only_T, axis=1))[mask],
                                     (np.mean(time_list_only_T, axis=1) + np.std(time_list_only_T, axis=1))[mask],
                                     alpha=0.3, color=color)
        else:
            # print(name_algo, name_algo_legend)
            time_list_only_W = np.zeros((len(n_samples_s), same_param["n_iter_algo"]))
            time_list_only_T = np.zeros((len(n_samples_s), same_param["n_iter_algo"]))
            time_list_only_W_approx = np.zeros((len(n_samples_s), same_param["n_iter_algo"]))
            time_list = np.zeros((len(n_samples_s), same_param["n_iter_algo"]))
            n_samples_s_temp = n_samples_s.copy()
            mask = np.ones(len(n_samples_s), dtype=bool)
            # n_samples_s_temp = n_samples_s_temp.astype(float)
            for i in range(len(n_samples_s)):
                acces_dict = name_algo + str(n_samples_s[i]) + "_" + str(n_samples_t[i])
                # print(params[acces_dict])
                if params[acces_dict] is None:
                    # n_samples_s_temp[i] = np.NaN
                    mask[i] = 0
                    continue
                time_list_only_W_approx[i] = params[acces_dict]["time_list_only_W_approx"]
                time_list_only_T[i] = params[acces_dict]["time_list_only_T"]
                time_list_only_W[i] = params[acces_dict]["time_list_only_W"]
                time_list[i] = time_list_only_T[i] + time_list_only_W[i]
            if "S_GWL" in name_algo:
                # linestyle = "--"
                color = (0, 0, 0)
                marker = "*"
            elif "uniform" in name_algo:
                # linestyle = "dotted"
                color = (0.1, 0.5, 0.1)
                marker = "d"
            elif "sliced_gromov" == name_algo:
                color = (0.40, 0.2, 0)
                # linestyle = "dotted"
                marker = "^"
            else:
                # linestyle = "--"
                color = (0, 0.8, 0.5)#(0, 0, 1)
                marker = "+"

            plt.figure(0, figsize=figsize)
            plt.plot(n_samples_s[mask], np.mean(time_list, axis=1)[mask], linestyle=linestyle, label=name_algo_legend,
                     color=color, marker=marker, markersize=markersize)
            plt.fill_between(n_samples_s[mask],
                             (np.mean(time_list, axis=1) - np.std(time_list, axis=1))[mask],
                             (np.mean(time_list, axis=1) + np.std(time_list, axis=1))[mask],
                             alpha=0.3,
                             color=color)

            plt.figure(1, figsize=figsize)
            if name_algo == "uniform" or name_algo == "identity":
                continue
            # print(name_algo)
            # for i in range(len(n_samples_s)):
            #     print(n_samples_s[i])
            #     print(np.mean(time_list_only_T, axis=1)[i], np.std(time_list_only_T, axis=1)[i])
            plt.plot(n_samples_s[mask], np.mean(time_list_only_T, axis=1)[mask], linestyle=linestyle,
                     label=name_algo_legend, color=color, marker=marker, markersize=markersize)
            plt.fill_between(n_samples_s[mask],
                             (np.mean(time_list_only_T, axis=1) - np.std(time_list_only_T, axis=1))[mask],
                             (np.mean(time_list_only_T, axis=1) + np.std(time_list_only_T, axis=1))[mask],
                             alpha=0.3, color=color)

    plt.figure(0)
    if log_scale:
        plt.yscale("log")
        plt.xscale("log")
    plt.xlabel('Number of points: $N$', fontsize=fontsize_axis)
    plt.ylabel('Computational time (s)', fontsize=fontsize_axis)
    plt.xticks(fontsize=fontsize_ticks)
    plt.yticks(fontsize=fontsize_ticks)
    if show_legend:
        plt.legend(loc=(1.01, 0.15), fontsize=fontsize_legend)
    # plt.title("Time needed to compute T and Wasserstein")

    plt.figure(1)
    plt.xlabel('Number of points: $N$', fontsize=fontsize_axis)
    plt.ylabel('Computational time (s)', fontsize=fontsize_axis)
    plt.xticks(fontsize=fontsize_ticks)
    plt.yticks(fontsize=fontsize_ticks)

    if show_legend:
        plt.legend(loc=(1.01, 0.15), fontsize=fontsize_legend)
    # plt.title("Time needed to compute T")

    if log_scale:
        plt.yscale("log")
        plt.xscale("log")
    n_samples_s_temp = [0] * len(names_algo)

    for a, name_algo in enumerate(names_algo):
        if name_algo in ["e_gromov", "e_gromov_KL"]:
            for e in entropy:
                W_distance_list = np.ones((len(n_samples_s), same_param["n_iter_algo"])) * 10e10
                W_distance_list_approx = np.zeros((len(n_samples_s), same_param["n_iter_algo"]))
                plt.figure(2, figsize=figsize)
                # print("start with algo", a, name_algo)
                # n_samples_s_temp[a] = n_samples_s.copy()
                mask = np.ones(len(n_samples_s), dtype=bool)
                for i in range(len(n_samples_s)):
                    acces_dict = name_algo + str(n_samples_s[i]) + "_" + str(n_samples_t[i]) + str(e)
                    if params[acces_dict] is None:
                        mask[i] = False
                        continue
                    if n_samples_s[i] > 1000:
                        mask[i] = False
                    # if name_algo == "sliced_gromov":
                    #     W_distance_list_approx[i, :] = params[acces_dict]["W_distance_list_approx"] * 20
                    W_distance_list[i, :] = params[acces_dict]["W_distance_list"]

                # linestyle = "--"
                if name_algo == "e_gromov":
                    color = (0, 0.5, 0.8)
                    marker = "o"
                if name_algo == "e_gromov_KL":
                    color = (0,0,1)#(0, 0.8, 0.5)
                    marker = "P"
                # print(np.mean(W_distance_list[mask], axis=1), name_algo, e)
                if paper_plot:
                    legend_param = ""
                else:
                    legend_param = "$_{" + str(e) + "}$"
                plt.plot(n_samples_s[mask], np.mean(W_distance_list[mask], axis=1), linestyle=linestyle,
                         label=names_algo_legend[a] + legend_param, color=color, marker=marker, markersize=markersize)
                plt.fill_between(n_samples_s[mask],
                                 (np.mean(W_distance_list, axis=1) - np.std(W_distance_list, axis=1))[mask],
                                 (np.mean(W_distance_list, axis=1) + np.std(W_distance_list, axis=1))[mask],
                                 alpha=0.3,
                                 color=color)
        elif name_algo in ["S_GWL"]:
            for e in entropy_SGWL:
                W_distance_list = np.ones((len(n_samples_s), same_param["n_iter_algo"])) * 10e10
                W_distance_list_approx = np.zeros((len(n_samples_s), same_param["n_iter_algo"]))
                plt.figure(2, figsize=figsize)
                # print("start with algo", a, name_algo)
                # n_samples_s_temp[a] = n_samples_s.copy()
                mask = np.ones(len(n_samples_s), dtype=bool)
                for i in range(len(n_samples_s)):
                    acces_dict = name_algo + str(n_samples_s[i]) + "_" + str(n_samples_t[i]) + str(e)
                    if params[acces_dict] is None:
                        mask[i] = False
                        continue
                    if n_samples_s[i] > 1000:
                        mask[i] = False
                    # if name_algo == "sliced_gromov":
                    #     W_distance_list_approx[i, :] = params[acces_dict]["W_distance_list_approx"] * 20
                    W_distance_list[i, :] = params[acces_dict]["W_distance_list"]

                # linestyle = "--"
                color = (0, 0, 0)
                marker = "*"
                # print(np.mean(W_distance_list[mask], axis=1), np.mean(np.mean(W_distance_list[mask], axis=1)), name_algo, e)
                if paper_plot:
                    legend_param = ""
                else:
                    legend_param = "$_{" + str(e) + "}$"
                plt.plot(n_samples_s[mask], np.mean(W_distance_list[mask], axis=1), linestyle=linestyle,
                         label=names_algo_legend[a] + legend_param, color=color, marker=marker, markersize=markersize)
                plt.fill_between(n_samples_s[mask],
                                 (np.mean(W_distance_list, axis=1) - np.std(W_distance_list, axis=1))[mask],
                                 (np.mean(W_distance_list, axis=1) + np.std(W_distance_list, axis=1))[mask],
                                 alpha=0.3,
                                 color=color)
        elif name_algo in ["sampled_gromov_e_constant", "sampled_gromov_sliced", "sampled_gromov_no_KL_e_constant"]:
            for b_index, b in enumerate(batchsize):
                for m_index, m in enumerate(iteration):
                    if paper_plot:
                        if name_algo == "sampled_gromov_sliced":
                            if b != 1 or m != 100:
                                continue
                        else:
                            if b != best_param_b or m != best_param_m:
                                continue
                    W_distance_list = np.ones((len(n_samples_s), same_param["n_iter_algo"])) * 10e10
                    W_distance_list_approx = np.zeros((len(n_samples_s), same_param["n_iter_algo"]))

                    plt.figure(2, figsize=figsize)

                    # print("start with algo", a, name_algo)
                    # n_samples_s_temp[a] = n_samples_s.copy()
                    mask = np.ones(len(n_samples_s), dtype=bool)
                    for i in range(len(n_samples_s)):
                        acces_dict = name_algo + str(n_samples_s[i]) + "_" + str(n_samples_t[i]) + str(b) + str(m)
                        if params[acces_dict] is None:
                            # n_samples_s_temp[a][i] = np.NaN
                            # print("continue", name_algo, i)
                            mask[i] = False
                            continue
                        if n_samples_s[i] > 1000:
                            mask[i] = False
                        # if name_algo == "sliced_gromov":
                        #     W_distance_list_approx[i, :] = params[acces_dict]["W_distance_list_approx"] * 20
                        W_distance_list[i, :] = params[acces_dict]["W_distance_list"]

                    # linestyle = "-"
                    if paper_plot:
                        if name_algo == "sampled_gromov_sliced":
                            color = (1, 0.5, 0)
                            marker = "X"
                        else:
                            color = (1, 0, 0)
                            marker = "s"
                    else:
                        color = (1, b_index / len(batchsize), 0)

                    plt.plot(n_samples_s[mask], np.mean(W_distance_list[mask], axis=1), linestyle=linestyle,
                             label=names_algo_legend[a],
                             # label=None,
                             color=color, marker=marker, markersize=markersize)
                    plt.fill_between(n_samples_s[mask],
                                     (np.mean(W_distance_list, axis=1) - np.std(W_distance_list, axis=1))[mask],
                                     (np.mean(W_distance_list, axis=1) + np.std(W_distance_list, axis=1))[mask],
                                     alpha=0.3, color=color)
        else:
            W_distance_list = np.ones((len(n_samples_s), same_param["n_iter_algo"])) * 10e10
            W_distance_list_approx = np.zeros((len(n_samples_s), same_param["n_iter_algo"]))
            plt.figure(2, figsize=figsize)
            # print("start with algo", a, name_algo)
            # n_samples_s_temp[a] = n_samples_s.copy()
            mask = np.ones(len(n_samples_s), dtype=bool)
            for i in range(len(n_samples_s)):
                acces_dict = name_algo + str(n_samples_s[i]) + "_" + str(n_samples_t[i])
                if params[acces_dict] is None:
                    # n_samples_s_temp[a][i] = np.NaN
                    # print("continue", name_algo, i)
                    mask[i] = False
                    continue
                if n_samples_s[i] > 1000:
                    mask[i] = False
                if name_algo == "sliced_gromov":
                    W_distance_list_approx[i, :] = params[acces_dict]["W_distance_list_approx"]
                W_distance_list[i, :] = params[acces_dict]["W_distance_list"]

            if "S_GWL" in name_algo:
                # linestyle = "--"
                marker = ""
                color = (0, 0, 0)
            elif "uniform" in name_algo:
                # linestyle = "-."
                marker = "d"
                color = (0.9, 0.5, 0.9)
            elif "identity" in name_algo:
                # linestyle = "-."
                marker = ""
                color = (0.40, 0.2, 0)

            elif "sliced_gromov" in name_algo:
                # linestyle = "dotted"
                color = (0.40, 0.2, 0)
                marker = "^"

            else:
                marker = "+"
                # linestyle = "--"
                color = (0, 0.8, 0.5)#(0, 0, 1)

            if name_algo == "sliced_gromov":
                plt.plot(n_samples_s[mask], np.mean(W_distance_list_approx, axis=1)[mask], linestyle=linestyle,
                         label=names_algo_legend[a], color=color, marker=marker, markersize=markersize)
                plt.fill_between(n_samples_s[mask],
                                 (np.mean(W_distance_list_approx, axis=1) - np.std(W_distance_list_approx, axis=1))[
                                     mask],
                                 (np.mean(W_distance_list_approx, axis=1) + np.std(W_distance_list_approx, axis=1))[
                                     mask],
                                 alpha=0.3, color=color)
                W_distance_list_approx = W_distance_list_approx * 25
                plt.plot(n_samples_s[mask], np.mean(W_distance_list_approx, axis=1)[mask], linestyle=linestyle,
                         label="25" + names_algo_legend[a], color=color, marker=marker, markersize=markersize)
                plt.fill_between(n_samples_s[mask],
                                 (np.mean(W_distance_list_approx, axis=1) - np.std(W_distance_list_approx, axis=1))[
                                     mask],
                                 (np.mean(W_distance_list_approx, axis=1) + np.std(W_distance_list_approx, axis=1))[
                                     mask],
                                 alpha=0.3, color=color)
            else:
                plt.plot(n_samples_s[mask], np.mean(W_distance_list[mask], axis=1), linestyle=linestyle,
                         label=names_algo_legend[a], color=color, marker=marker, markersize=markersize)
                plt.fill_between(n_samples_s[mask],
                                 (np.mean(W_distance_list, axis=1) - np.std(W_distance_list, axis=1))[mask],
                                 (np.mean(W_distance_list, axis=1) + np.std(W_distance_list, axis=1))[mask],
                                 alpha=0.3, color=color)

                # here just to add Sliced GW in the legend. Kind of ugly...

                if "gaussian_graph" == name_dataset and paper_plot and legend_SGW:
                    # linestyle = "dotted"
                    color = (0.40, 0.2, 0)
                    plt.plot([None], [None], linestyle=linestyle,
                             label="SGW", color=color)

                    plt.plot([None], [None], linestyle=linestyle,
                             label="15 SGW", color=color)
                    legend_SGW = False
    if log_scale:
        plt.xscale("log")
    plt.xlabel('Number of points: $N$', fontsize=fontsize_axis)
    plt.ylabel('GW value: $\mathcal{E}$ (T)', fontsize=fontsize_axis)
    plt.xticks(fontsize=fontsize_ticks)
    plt.yticks(fontsize=fontsize_ticks)
    # if name_dataset == "gaussian_point_graph":
    #     loc = (1, 0)
    # else:
    #     loc = (1, 0)
    # if show_legend:
    #     plt.legend(loc=(-0.8, 0.05), fontsize=fontsize_legend)
    if show_legend:
        if name_dataset == "gaussian_graph":
            plt.legend(loc=(1.01, 0.15), fontsize=fontsize_legend)
        else:
            plt.legend(loc=(1.01, 0.05), fontsize=fontsize_legend)

    # plt.title("Wasserstein distance")

    # if name_dataset == "same_graph":
    #     # n_samples_s_temp = [0] * len(names_algo)
    #     diff_to_identity_list = np.ones((len(n_samples_s), same_param["n_iter_algo"])) * 10000
    #     plt.figure(3, figsize=figsize)
    #     for a, name_algo in enumerate(names_algo):
    #         if name_algo in ["e_gromov", "e_gromov_KL"]:
    #             for e_index, e in enumerate(entropy):
    #                 # n_samples_s_temp[a] = n_samples_s.copy()
    #                 mask = np.ones(len(n_samples_s), dtype=bool)
    #                 for i in range(len(n_samples_s)):
    #                     acces_dict = name_algo + str(n_samples_s[i]) + "_" + str(n_samples_t[i]) + str(e)
    #                     if params[acces_dict] is None:
    #                         # n_samples_s_temp[a][i] = np.NaN
    #                         mask[i] = False
    #                         continue
    #                     diff_to_identity_list[i, :] = params[acces_dict]["diff_to_identity"]
    #                 linestyle = "--"
    #                 color = (0, e_index / (len(entropy)), 0)
    #                 if e_index == 0:
    #                     plt.plot(n_samples_s[mask], np.mean(diff_to_identity_list, axis=1)[mask], linestyle=linestyle,
    #                              label=names_algo_legend[a], color=color)
    #                     plt.fill_between(n_samples_s[mask],
    #                                      (np.mean(diff_to_identity_list, axis=1) - np.std(diff_to_identity_list,
    #                                                                                       axis=1))[
    #                                          mask],
    #                                      (np.mean(diff_to_identity_list, axis=1) + np.std(diff_to_identity_list,
    #                                                                                       axis=1))[
    #                                          mask],
    #                                      alpha=0.3, color=color)
    #                 else:
    #                     plt.plot(n_samples_s[mask], np.mean(diff_to_identity_list, axis=1)[mask], linestyle=linestyle,
    #                              color=color)
    #                     plt.fill_between(n_samples_s[mask],
    #                                      (np.mean(diff_to_identity_list, axis=1) - np.std(diff_to_identity_list,
    #                                                                                       axis=1))[
    #                                          mask],
    #                                      (np.mean(diff_to_identity_list, axis=1) + np.std(diff_to_identity_list,
    #                                                                                       axis=1))[
    #                                          mask],
    #                                      alpha=0.3, color=color)
    #
    #         elif name_algo in ["sampled_gromov_e_constant", "sampled_gromov_sliced"]:
    #             for b_index, b in enumerate(batchsize):
    #                 for m_index, m in enumerate(iteration):
    #                     if paper_plot:
    #                         if name_algo == "sampled_gromov_sliced":
    #                             if b != 1 or m != 100:
    #                                 continue
    #                         else:
    #                             if b != best_param_b or m != best_param_m:
    #                                 continue
    #                     # n_samples_s_temp[a] = n_samples_s.copy()
    #                     mask = np.ones(len(n_samples_s), dtype=bool)
    #                     for i in range(len(n_samples_s)):
    #                         acces_dict = name_algo + str(n_samples_s[i]) + "_" + str(n_samples_t[i]) + str(b) + str(m)
    #                         if params[acces_dict] is None:
    #                             # n_samples_s_temp[a][i] = np.NaN
    #                             mask[i] = False
    #                             continue
    #                         diff_to_identity_list[i, :] = params[acces_dict]["diff_to_identity"]
    #                     # if "sampled_gromov" in name_algo:
    #                     linestyle = "-"
    #                     # elif "S_GWL" in name_algo:
    #                     #     linestyle = "--"
    #                     # elif "sliced_gromov" in name_algo:
    #                     #     linestyle = "--"
    #                     # elif "uniform" in name_algo:
    #                     #     linestyle = "dotted"
    #                     # else:
    #                     #     linestyle = "--"
    #                     color = (m_index / len(iteration), 0, 0)
    #                     if b_index == 0 and m_index == 0:
    #                         plt.plot(n_samples_s[mask], np.mean(diff_to_identity_list, axis=1)[mask],
    #                                  linestyle=linestyle,
    #                                  label=names_algo_legend[a], color=color)
    #                         plt.fill_between(n_samples_s[mask],
    #                                          (np.mean(diff_to_identity_list, axis=1) - np.std(diff_to_identity_list,
    #                                                                                           axis=1))[mask],
    #                                          (np.mean(diff_to_identity_list, axis=1) + np.std(diff_to_identity_list,
    #                                                                                           axis=1))[mask],
    #                                          alpha=0.3, color=color)
    #                     else:
    #                         plt.plot(n_samples_s[mask], np.mean(diff_to_identity_list, axis=1)[mask],
    #                                  linestyle=linestyle,
    #                                  color=color)
    #                         plt.fill_between(n_samples_s[mask],
    #                                          (np.mean(diff_to_identity_list, axis=1) - np.std(diff_to_identity_list,
    #                                                                                           axis=1))[mask],
    #                                          (np.mean(diff_to_identity_list, axis=1) + np.std(diff_to_identity_list,
    #                                                                                           axis=1))[mask],
    #                                          alpha=0.3, color=color)
    #
    #         else:
    #             # n_samples_s_temp[a] = n_samples_s.copy()
    #             mask = np.ones(len(n_samples_s), dtype=bool)
    #             for i in range(len(n_samples_s)):
    #                 acces_dict = name_algo + str(n_samples_s[i]) + "_" + str(n_samples_t[i])
    #                 print(acces_dict)
    #                 if params[acces_dict] is None:
    #                     # n_samples_s_temp[a][i] = np.NaN
    #                     mask[i] = False
    #                     continue
    #                 diff_to_identity_list[i, :] = params[acces_dict]["diff_to_identity"]
    #             if "sampled_gromov" in name_algo:
    #                 linestyle = "-"
    #             elif "S_GWL" in name_algo:
    #                 linestyle = "--"
    #             elif "sliced_gromov" in name_algo:
    #                 linestyle = "--"
    #             elif "uniform" in name_algo:
    #                 linestyle = "dotted"
    #             else:
    #                 linestyle = "--"
    #
    #             plt.plot(n_samples_s[mask], np.mean(diff_to_identity_list, axis=1)[mask], linestyle=linestyle,
    #                      label=names_algo_legend[a])
    #             plt.fill_between(n_samples_s[mask],
    #                              (np.mean(diff_to_identity_list, axis=1) - np.std(diff_to_identity_list, axis=1))[mask],
    #                              (np.mean(diff_to_identity_list, axis=1) + np.std(diff_to_identity_list, axis=1))[mask],
    #                              alpha=0.3)
    #
    #     if log_scale:
    #         plt.xscale("log")
    #     plt.legend()
    # plt.title("Wasserstein distance")

    if save:
        plt.figure(0)
        # plt.gca().set_axis_off()
        # plt.subplots_adjust(top=1, bottom=0, right=1, left=0,
        #                     hspace=0, wspace=0)
        # plt.margins(0, 0)
        # plt.gca().xaxis.set_major_locator(plt.NullLocator())
        # plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.savefig("./figure/hyper/hyper_T_and_W_" + name_dataset + "_" + loss_func_name + ".pdf",
                    bbox_inches="tight")
        plt.figure(1)
        plt.savefig("./figure/hyper/hyper_T_" + name_dataset + "_" + loss_func_name + ".pdf",
                    bbox_inches="tight")
        plt.figure(2)
        plt.savefig("./figure/hyper/hyper_distance_" + name_dataset + "_" + loss_func_name + ".pdf",
                    bbox_inches="tight")
        # if name_dataset == "same_graph":
        #     plt.figure(3)
        #     plt.savefig("./figure/hyper/hyper_diff_to_identity_" + name_dataset + "_" + loss_func_name + ".pdf",
        #                 bbox_inches="tight")
    plt.show()


def analyse_data(name_dataset="gaussian",
                 pickle_path="pickle_compare",
                 names_algo=["sampled_gromov"],
                 names_algo_legend=["sampled_gromov"],
                 pickle_name="",
                 n_samples_s=[50, 100, 200],
                 n_samples_t=[],
                 noise_graph=0.1,
                 loss_func_name="1_loss",
                 best_T_not_needed=True,
                 log_scale=True,
                 figsize=(10, 5),
                 save=False):
    n_samples_s = np.array(n_samples_s)
    if name_dataset in ["gaussian_graph", "same_graph"]:
        try:
            names_algo.remove("sliced_gromov")
            names_algo_legend.remove("Sliced GW")
        except:
            pass

    if name_dataset in ["gaussian_graph", "gaussian_point_graph"]:
        try:
            names_algo.remove("identity")
            names_algo_legend.remove("Identity")
        except:
            pass

    if not n_samples_t:
        n_samples_t = n_samples_s

    params = {}

    same_param = None

    for name_algo in names_algo:
        for i in range(len(n_samples_s)):
            acces_dict = name_algo + str(n_samples_s[i]) + "_" + str(n_samples_t[i])

            if name_dataset == "same_graph":
                path_pickle = "./" + pickle_path + "/" + name_dataset + "/" + name_algo + "/" + str(n_samples_s[i]) + \
                              "_" + str(n_samples_t[i]) + "_" + loss_func_name + "__" + str(noise_graph) + "_" + \
                              pickle_name + ".pickle"
            else:
                path_pickle = "./" + pickle_path + "/" + name_dataset + "/" + name_algo + "/" + str(n_samples_s[i]) + \
                              "_" + str(n_samples_t[i]) + "_" + loss_func_name + pickle_name + ".pickle"

            if path.exists(path_pickle):
                with open(path_pickle, "rb") as pickle_in:

                    params[acces_dict] = pickle.load(pickle_in)
                    if same_param is None:
                        same_param = params[acces_dict]
                    elif params[acces_dict]["name_algo"] != "uniform":
                        # Use this to check if every run use the same parameter
                        for test in ["n_iter_algo", "rdm_seed", "same_space"]:
                            if same_param[test] != params[acces_dict][test]:
                                print(params[acces_dict])
                                pass
                                print("Warning", test, same_param[test], params[acces_dict][test])
                            # assert same_param[test] == params[acces_dict][test]
            else:
                # print("outer")
                params[acces_dict] = None

    assert same_param is not None  # Maybe an error in the pickle file name
    for name_algo, name_algo_legend in zip(names_algo, names_algo_legend):
        # print(name_algo, name_algo_legend)
        time_list_only_W = np.zeros((len(n_samples_s), same_param["n_iter_algo"]))
        time_list_only_T = np.zeros((len(n_samples_s), same_param["n_iter_algo"]))
        time_list_only_W_approx = np.zeros((len(n_samples_s), same_param["n_iter_algo"]))
        time_list = np.zeros((len(n_samples_s), same_param["n_iter_algo"]))
        n_samples_s_temp = n_samples_s.copy()
        mask = np.ones(len(n_samples_s), dtype=bool)
        # n_samples_s_temp = n_samples_s_temp.astype(float)
        for i in range(len(n_samples_s)):
            acces_dict = name_algo + str(n_samples_s[i]) + "_" + str(n_samples_t[i])
            # print(params[acces_dict])
            if params[acces_dict] is None:
                # n_samples_s_temp[i] = np.NaN
                mask[i] = 0
                continue
            time_list_only_W_approx[i] = params[acces_dict]["time_list_only_W_approx"]
            time_list_only_T[i] = params[acces_dict]["time_list_only_T"]
            time_list_only_W[i] = params[acces_dict]["time_list_only_W"]
            time_list[i] = time_list_only_T[i] + time_list_only_W[i]
        if "sampled_gromov" in name_algo:
            linestyle = "-"
        elif "S_GWL" in name_algo:
            linestyle = "--"
        elif "sliced_gromov" in name_algo:
            linestyle = "--"
        elif "uniform" in name_algo:
            linestyle = "dotted"
        else:
            linestyle = "--"
        plt.figure(0, figsize=figsize)
        plt.plot(n_samples_s[mask], np.mean(time_list, axis=1)[mask], linestyle=linestyle, label=name_algo_legend)
        plt.fill_between(n_samples_s[mask],
                         (np.mean(time_list, axis=1) - np.std(time_list, axis=1))[mask],
                         (np.mean(time_list, axis=1) + np.std(time_list, axis=1))[mask],
                         alpha=0.3)

        plt.figure(1, figsize=figsize)
        if name_algo == "uniform" or name_algo == "identity":
            continue
        # print(name_algo)
        # for i in range(len(n_samples_s)):
        #     print(n_samples_s[i])
        #     print(np.mean(time_list_only_T, axis=1)[i], np.std(time_list_only_T, axis=1)[i])
        plt.plot(n_samples_s[mask], np.mean(time_list_only_T, axis=1)[mask], linestyle=linestyle,
                 label=name_algo_legend)
        plt.fill_between(n_samples_s[mask],
                         (np.mean(time_list_only_T, axis=1) - np.std(time_list_only_T, axis=1))[mask],
                         (np.mean(time_list_only_T, axis=1) + np.std(time_list_only_T, axis=1))[mask],
                         alpha=0.3)

    plt.figure(0)
    if log_scale:
        plt.yscale("log")
        plt.xscale("log")

    plt.legend()
    # plt.title("Time needed to compute T and Wasserstein")

    plt.figure(1)
    plt.legend()
    # plt.title("Time needed to compute T")

    if log_scale:
        plt.yscale("log")
        plt.xscale("log")
    n_samples_s_temp = [0] * len(names_algo)

    for a, name_algo in enumerate(names_algo):
        W_distance_list = np.ones((len(n_samples_s), same_param["n_iter_algo"])) * 10e10
        W_distance_list_approx = np.zeros((len(n_samples_s), same_param["n_iter_algo"]))
        plt.figure(2, figsize=figsize)
        # print("start with algo", a, name_algo)
        # n_samples_s_temp[a] = n_samples_s.copy()
        mask = np.ones(len(n_samples_s), dtype=bool)
        for i in range(len(n_samples_s)):
            acces_dict = name_algo + str(n_samples_s[i]) + "_" + str(n_samples_t[i])
            if params[acces_dict] is None:
                # n_samples_s_temp[a][i] = np.NaN
                # print("continue", name_algo, i)
                mask[i] = False
                continue
            if n_samples_s[i] > 1000:
                mask[i] = False
            if name_algo == "sliced_gromov":
                W_distance_list_approx[i, :] = params[acces_dict]["W_distance_list_approx"] * 25
            W_distance_list[i, :] = params[acces_dict]["W_distance_list"]

        if "sampled_gromov" in name_algo:
            linestyle = "-"
        elif "S_GWL" in name_algo:
            linestyle = "--"
        elif "sliced_gromov" in name_algo:
            linestyle = "--"
        elif "uniform" in name_algo:
            linestyle = "dotted"
        else:
            linestyle = "--"

        if name_algo == "sliced_gromov":
            plt.plot(n_samples_s[mask], np.mean(W_distance_list_approx, axis=1)[mask], linestyle=linestyle,
                     label=names_algo_legend[a])
            plt.fill_between(n_samples_s[mask],
                             (np.mean(W_distance_list_approx, axis=1) - np.std(W_distance_list_approx, axis=1))[mask],
                             (np.mean(W_distance_list_approx, axis=1) + np.std(W_distance_list_approx, axis=1))[mask],
                             alpha=0.3)
        else:
            plt.plot(n_samples_s[mask], np.mean(W_distance_list[mask], axis=1), linestyle=linestyle,
                     label=names_algo_legend[a])
            plt.fill_between(n_samples_s[mask],
                             (np.mean(W_distance_list, axis=1) - np.std(W_distance_list, axis=1))[mask],
                             (np.mean(W_distance_list, axis=1) + np.std(W_distance_list, axis=1))[mask],
                             alpha=0.3)
    if log_scale:
        plt.xscale("log")
    plt.legend()
    # plt.title("Wasserstein distance")

    if name_dataset == "same_graph":
        # n_samples_s_temp = [0] * len(names_algo)
        diff_to_identity_list = np.ones((len(n_samples_s), same_param["n_iter_algo"])) * 10000
        plt.figure(3, figsize=figsize)
        for a, name_algo in enumerate(names_algo):
            # n_samples_s_temp[a] = n_samples_s.copy()
            mask = np.ones(len(n_samples_s), dtype=bool)
            for i in range(len(n_samples_s)):
                acces_dict = name_algo + str(n_samples_s[i]) + "_" + str(n_samples_t[i])
                if params[acces_dict] is None:
                    # n_samples_s_temp[a][i] = np.NaN
                    mask[i] = False
                    continue
                diff_to_identity_list[i, :] = params[acces_dict]["diff_to_identity"]
            if "sampled_gromov" in name_algo:
                linestyle = "-"
            elif "S_GWL" in name_algo:
                linestyle = "--"
            elif "sliced_gromov" in name_algo:
                linestyle = "--"
            elif "uniform" in name_algo:
                linestyle = "dotted"
            else:
                linestyle = "--"

            plt.plot(n_samples_s[mask], np.mean(diff_to_identity_list, axis=1)[mask], linestyle=linestyle,
                     label=names_algo_legend[a])
            plt.fill_between(n_samples_s[mask],
                             (np.mean(diff_to_identity_list, axis=1) - np.std(diff_to_identity_list, axis=1))[mask],
                             (np.mean(diff_to_identity_list, axis=1) + np.std(diff_to_identity_list, axis=1))[mask],
                             alpha=0.3)

        if log_scale:
            plt.xscale("log")
        plt.legend()
        # plt.title("Wasserstein distance")

    if save:
        plt.figure(0)
        plt.savefig("./figure/T_and_W_" + name_dataset + "_" + loss_func_name + ".pdf")
        plt.figure(1)
        plt.savefig("./figure/T_" + name_dataset + "_" + loss_func_name + ".pdf")
        plt.figure(2)
        plt.savefig("./figure/distance_" + name_dataset + "_" + loss_func_name + ".pdf")
        if name_dataset == "same_graph":
            plt.figure(3)
            plt.savefig("./figure/diff_to_identity_" + name_dataset + "_" + loss_func_name + ".pdf")

    # print(W_distance_list)
    # best_w = np.min(W_distance_list, axis=(0, 2), keepdims=True)
    # # print(best_w)
    # # print(best_w.shape)
    # # print(W_distance_list)
    # W_distance_list = W_distance_list - best_w
    # # print(W_distance_list)
    # for a, name_algo in enumerate(names_algo):
    #     if "sampled_gromov" in name_algo:
    #         linestyle = "--"
    #     else:
    #         linestyle = "-"
    #     plt.figure(3, figsize=figsize)
    #
    #     plt.plot(n_samples_s_temp[a], np.mean(W_distance_list[a], axis=1),
    #              linestyle=linestyle, label=name_algo)
    #     plt.fill_between(n_samples_s_temp[a],
    #                      np.mean(W_distance_list[a], axis=1) - np.std(W_distance_list[a], axis=1),
    #                      np.mean(W_distance_list[a], axis=1) + np.std(W_distance_list[a], axis=1),
    #                      alpha=0.3)
    #     plt.title("Approx Wasserstein distance")
    #
    #     plt.legend()
    plt.show()


