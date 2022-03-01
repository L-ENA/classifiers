import pandas as pd
from ActiveLearner import ActiveLearner
from neural_classifier import SPECTER_CLS

from other_classifiers import regexClassifier

if __name__ == '__main__':


    data = pd.read_csv("data//dev_500.csv").fillna("")
    data = data.sample(frac=1, random_state=48)
    data.reset_index(drop=True, inplace=True)

    ####################################################Can safely ignore this. These lines look where data was missing from the scan and supplement it with the mined data. I guess at runtime with new projects we won;t have that yet.
    # ints=data["Interventions"]
    # mined=data["mined_intervention_control"]
    # new=[ d if d != "" else mined[i] for i, d in enumerate(ints) ]#use mined data if no intervation pulled from scan. Totally optional
    # data["Interventions"]=new
    #########################################################

    ####################################Filter model
    classifier = regexClassifier
    al = ActiveLearner(classifier, data, field="ScientificTitle", model_name="Filter", filter=r'mouse')
    #al = ActiveLearner(classifier, data, field="Interventions", model_name="Filter")
    al.simulate_learning()

    ################################################Random as reference
    # classifier = emptyClassifier
    # al = ActiveLearner(classifier, data, field="ScientificTitle", model_name="Random", do_preprocess=False)
    # al.simulate_learning()

    ##################################################NEURAL MODEL
    # classifier = SPECTER_CLS
    # al = ActiveLearner(classifier, data, field="ScientificTitle", model_name="Neural", do_preprocess=True)
    # ##al = ActiveLearner(classifier, data, field="Interventions", model_name="Neural", do_preprocess=False)###example chosing a different field
    # al.simulate_learning()#ecample for simulation, can still be used to provide fancy plot to the user to see how the model would have reacted tto their data in active learning scenario

    output= al.reorder_once()
    output.to_csv("data//reordered.csv", index=False)


