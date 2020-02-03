from utils import Dataset
import pickle
from models import MotifEmbedding

if __name__ == '__main__':


    data = Dataset('./ZG_wpli', './df_raven.csv')

    # read data set
    method = 'threshold'
    percentiles = [50]
    fn = "data_{}_{}.pkl".format(method, "-".join(list(map(str, percentiles))))
    try:
        with open(fn, 'rb') as f:
            human2data = pickle.load(f)
        data.human2data = human2data
    except Exception as e:
        data.get_graph_data(method, percentiles=percentiles)
        with open(fn, 'wb') as f:
            pickle.dump(data.human2data, f)


    motifs = MotifEmbedding(data.human2data)
    motifs.compute_human2embedding(100)
    motifs.save_embeddings()



