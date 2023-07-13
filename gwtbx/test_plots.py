import numpy as np
from gw_interact_plot import analyse_data_hyper

def plot_gw_runs(sampled_gromov_e_constant=True,
                 sampled_gromov_no_KL_e_constant=False,
                 sampled_gromov_sliced=True,
                 sliced_gromov=True,
                 identity=True,
                 S_GWL=True,
                 gromov=True,
                 e_gromov=True,
                 e_gromov_KL=True,
                 uniform=True,
                 entropy=[[0.001, 0.005, 0.01, 0.05, 0.1], [0.001], [0.005], [0.01], [0.05], [0.1]],
                 entropy_SGWL=[[0.01, 0.1, 1, 10, 100], [0.01], [0.1], [1], [10], [100]],
                 iteration=[[10, 50, 100, 500, 1000], [10], [50], [100], [500], [1000]],
                 batchsize=[[1, 10, 100, 1000], [1], [10], [100], [1000]],
                 max_n_samples=10000,
                 pickle_path="SampledGW/pickle_compare/",
                 pickle_name=[],
                 name_dataset=["gaussian_point_graph", "gaussian_graph"],
                 log_scale=True,
                 figsize1=8,
                 figsize2=5,
                 loss_func_name=["1_loss", "square_loss"],
                 save=False,
                 paper_plot=True,
                 show_legend=True):
    names_algo = []
    names_algo_legend = []
    if e_gromov:
        names_algo.append("e_gromov")
        names_algo_legend.append("EGW")
    if e_gromov_KL:
        names_algo.append("e_gromov_KL")
        names_algo_legend.append("EGW$^{KL}$")
    if gromov:
        names_algo.append("gromov")
        names_algo_legend.append("EMD-GW")
    if S_GWL:
        names_algo.append("S_GWL")
        names_algo_legend.append("S-GWL")
    if sampled_gromov_e_constant:
        names_algo.append("sampled_gromov_e_constant")
        names_algo_legend.append("$SaGroW^{KL}$")
    if sampled_gromov_no_KL_e_constant:
        names_algo.append("sampled_gromov_no_KL_e_constant")
        names_algo_legend.append("$SaGroW$")
    if sampled_gromov_sliced:
        names_algo.append("sampled_gromov_sliced")
        names_algo_legend.append("PoGroW")
    if sliced_gromov:
        names_algo.append("sliced_gromov")
        names_algo_legend.append("SGW")
    if uniform:
        names_algo.append("uniform")
        names_algo_legend.append("Uniform")
    if identity:
        names_algo.append("identity")
        names_algo_legend.append("Identity")
    n_samples = np.array([20, 50, 100, 200, 500, 1000, 1500, 2000, 5000, 10000])
    n_samples = n_samples[n_samples <= max_n_samples]

    if not pickle_name:
        if name_dataset == "same_graph":
            pickle_name = "same"
        else:
            pickle_name = "_2"
    analyse_data_hyper(name_dataset=name_dataset[0],
                       pickle_path=pickle_path,
                       names_algo=names_algo,
                       names_algo_legend=names_algo_legend,
                       pickle_name=pickle_name,
                       n_samples_s=list(n_samples),
                       loss_func_name=loss_func_name[0],
                       log_scale=log_scale,
                       figsize=(figsize1, figsize2),
                       save=save,
                       entropy=entropy[0],
                       entropy_SGWL=entropy_SGWL[0],
                       batchsize=batchsize[0],
                       iteration=iteration[0],
                       paper_plot=paper_plot,
                       show_legend=show_legend,
                       linestyle="-")

def main():
    plot_gw_runs(save=True)

if __name__ == "__main__":
    main()
