from tqdm import tqdm
from sentence_transformers import util


def calculate_similarity_single_sent(pos_idx, cos_sim):
    """
    pos_idx: list with lindex calues of positively-labelled rows
    target_list: list with strings to claculate similarities on

    returns: list of length target_list, with float values between 0 and 1 corresponding to cosine similarity (1=similar).

    Example:
    idx_pos=[0,2,4]
    inputs=["Artificial Intelligence With DEep Learning on COROnary Microvascular Disease",
            "Neuronal Mechanisms of Human Episodic Memory",
            "Polyp REcognition Assisted by a Device Interactive Characterization Tool - The PREDICT Study",
            "Artificial Intelligence With DEep Learning on COROnary Microvascular Disease",
            "Artificial intelligence and machine learning for Covid 19",
            "Effects of a Mindfulness-Based Eating Awareness Training online intervention in adults of obesity: A randomized controlled trials",
            "Prediction of Phakic Intraocular Lens Vault Using Machine Learning",
            "AI system for egg OFC Prediction System of Infants"]

    calculate_similarity_single_sent(idx_pos, inputs,model)

    """
    # emb_source= my_model.encode(target_list)#get our data column

    # print(emb_source.shape)
    # print("Calculating similarity matrix...")

    # print(cos_sim.shape)

    all_sims = []

    for i in tqdm(cos_sim):  # for each input and its pairwise similarities
        avg_for_record = []  # list to store all pairwwise similarities of this field with the positively labelled fields
        for ind in pos_idx:  # for each positive labelled record
            avg_for_record.append(i[ind].item())  # add the cosine similarity
            # print(i[ind])
        all_sims.append(
            sum(avg_for_record) / len(avg_for_record))  # average similarity is used for now, but could use median etc
        # print(sum(avg_for_record)/len(avg_for_record))
        # print("-----")
    # print(all_sims)
    return all_sims