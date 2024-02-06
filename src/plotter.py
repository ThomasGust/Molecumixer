import matplotlib.pyplot as plt
import pickle as pkl
import os


def load_stats(checkpoints_directory):
    """there is probably a much easier way to unpack a dictionary but this will work for now"""
    folder_names = os.listdir(checkpoints_directory)
    #nepochs = len(folder_names)

    descriptor_loss = []
    threed_descriptor_loss = []
    graph_descriptor_loss = []
    accum_d_loss = []
    mfp2_loss = []
    mfp3_loss = []
    maccs_loss = []
    rdkfp_loss = []
    avfp_loss = []
    fingerprint_loss = []
    total_loss = []
    learning_rate = []

    for i, path in enumerate(folder_names):
        fname = os.path.join(checkpoints_directory, f"EPOCH{i}", f"stats_epoch_{i}.stats")
        with open(fname, "rb") as f:
            epoch_stats = pkl.load(f)

            e_descriptor_loss = epoch_stats["descriptor loss"]
            e_threed_descriptor_loss = epoch_stats["3d descriptor loss"]
            e_graph_descriptor_loss = epoch_stats["graph descriptor loss"]
            e_accum_descriptor_loss = epoch_stats["accumulated descriptor loss"]
            e_mfp2_loss = epoch_stats["mfp2 loss"]
            e_mfp3_loss = epoch_stats["mfp3 loss"]
            e_maccs_loss = epoch_stats["maccs loss"]
            e_rdkfp_loss = epoch_stats["rdkfp loss"]
            e_avfp_loss = epoch_stats["avfp loss"]
            e_fingerprint_loss = epoch_stats["fingerprint loss"]
            e_total_loss = epoch_stats["total loss"]
            e_learning_rate = epoch_stats["lr"]

            descriptor_loss.append(e_descriptor_loss)
            threed_descriptor_loss.append(e_threed_descriptor_loss)
            graph_descriptor_loss.append(e_graph_descriptor_loss.item())
            accum_d_loss.append(e_accum_descriptor_loss)
            mfp2_loss.append(e_mfp2_loss)
            mfp3_loss.append(e_mfp3_loss)
            maccs_loss.append(e_maccs_loss)
            rdkfp_loss.append(e_rdkfp_loss)
            avfp_loss.append(e_avfp_loss)
            fingerprint_loss.append(e_fingerprint_loss)
            total_loss.append(e_total_loss)
            learning_rate.append(e_learning_rate)
    
    return {"descriptor loss": descriptor_loss,
            "3d descriptor loss": threed_descriptor_loss,
            "graph descriptor loss": graph_descriptor_loss,
            "accumulated descriptor loss": accum_d_loss,
            "mfp2 loss": mfp2_loss,
            "mfp3 loss": mfp3_loss,
            "maccs loss": maccs_loss,
            "rdkfp loss": rdkfp_loss,
            "avfp loss": avfp_loss,
            "fingerprint loss": fingerprint_loss,
            "total loss": total_loss,
            "learning_rate": learning_rate}

def plot_list(keys, data, title=None, fp=None):

    labels = keys #TODO Make labels better in the future

    for key in keys:
        plt.plot(range(len(data[key])), data[key])
    plt.title(title)
    plt.legend(labels)

    if fp is not None:
        plt.savefig(fp)
        plt.close()

def training_summary_plots(data, path):
    keys = [key for key in data]
    for key in keys:
        plot_list([key], data, key, os.path.join(path, f"{key.replace(' ', '_')}.png"))
    plot_list(keys[0:3], data, "Descriptor Loss", os.path.join(path, "combined_descriptor_loss.png"))
    plot_list(keys[4:9], data, "Fingerprint Loss", os.path.join(path, "combined_fingerprint_loss.png"))

if __name__ == "__main__":
    data = load_stats("molecular_analysis\\checkpoints\\cgtnn")
    #plot_list(["mfp3 loss"], data=data, title="MFP3 Fingerprint Loss", fp="molecular_analysis\\figures\\training\\cgtnn\\mfp3_loss.png")
    training_summary_plots(data, "molecular_analysis\\figures\\training\\cgtnn")