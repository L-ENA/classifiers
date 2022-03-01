from utils import calculate_similarity_single_sent
from classifier_base import BaseClassifier
from sentence_transformers import util, SentenceTransformer
import random
class SPECTER_CLS(BaseClassifier):

    def update_field(self, current_data):
        print("Loading and making similarity calculations...")
        self.model_name = 'sentence-transformers/allenai-specter'
        self.model = SentenceTransformer(self.model_name)


        self.emb_source = self.model.encode(list(current_data[self.model_field_name]))  # get our data column
        self.cos_sim = util.pytorch_cos_sim(self.emb_source, self.emb_source)  # .diagonal().tolist()#all similarities
        self.was_prepared = True
        # print(self.cos_sim[0][0])

    def train(self):
        """
        There  is no training for a regular expression filter, but the filter could be reset to something else with each training step
        :param reset_filter:
        :param filter:
        :return:
        """
        pass



    def predict(self, current_data):

        print("Predicting. Currently discovered {} labels".format(
            len(list(current_data[current_data["discovered_labels"] != ""].index.values))))
        # print('Predicting {} data points using <{}> model'.format(len(self.preprocessed),self.model_name))
        self.predictions = []

        idx = list(current_data[current_data[
                                    "discovered_labels"] == 1].index.values)  # filter positive labels and get their index values as list
        if len(idx) >= 10:
            idx_pos = random.choices(idx, k=10)
        else:
            idx_pos = idx
        print("Found {} labels, predicting based on indexes: {}".format(len(idx),
                                                                        idx_pos))  # print first five as sanity check
        #print(idx)

        try:
            # sci_title_sim=calculate_similarity_single_sent(idx_pos,self.cos_sim)
            print(self.was_prepared)
            # sci_title_sim=current_data["predictions"]
            # self.emb_source= self.model.encode(list(current_data["ScientificTitle"]))#get our data column
            # self.cos_sim = util.pytorch_cos_sim(self.emb_source, self.emb_source)

            sims = calculate_similarity_single_sent(idx_pos, self.cos_sim)
            sci_title_sim = []
            print("Index values:")
            for data_index in current_data.index.values:
                sci_title_sim.append(sims[data_index])

            #print(sci_title_sim)
        except:
            print("Resetting an dpreparing data...")
            self.my_idx = idx_pos
            #current_data.reset_index(drop=True, inplace=True)
            #self.update_field(list(current_data["ScientificTitle"]))  # ScientificTitle#Interventions
            current_data.reset_index(drop=True, inplace=True)
            self.update_field(current_data)
            sci_title_sim = calculate_similarity_single_sent(self.my_idx, self.cos_sim)
            #print(sci_title_sim)
            print(current_data.index.values[:10])

        return sci_title_sim