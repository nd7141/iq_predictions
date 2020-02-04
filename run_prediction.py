from utils import Dataset
import pickle
from models import MotifEmbedding
from models import GIN
from pipeline import Pipeline
import numpy as np

if __name__ == '__main__':


    data = Dataset('./ZG_wpli', './df_raven.csv')

    # read data set
    # method = 'threshold'
    # percentiles = [50]
    # fn = "data_{}_{}.pkl".format(method, "-".join(list(map(str, percentiles))))
    # try:
    #     with open(fn, 'rb') as f:
    #         human2data = pickle.load(f)
    #     data.human2data = human2data
    # except Exception as e:
    #     data.get_graph_data(method, percentiles=percentiles)
    #     with open(fn, 'wb') as f:
    #         pickle.dump(data.human2data, f)
    #
    #
    # motifs = MotifEmbedding(data.human2data)
    # motifs.compute_human2embedding(100)
    # motifs.save_embeddings()

    data = Dataset('ZG_wpli', 'df_raven.csv')

    with open('gin.txt', 'w+') as f:
        f.write('perc rmse mae\n')

    for percentile in range(10, 91, 10):
        try:
            data.get_graph_data('threshold', percentiles=[percentile])

            model = GIN(1)
            pipe = Pipeline(model, data.human2data)
            gen = pipe.kfold_split(10)

            mean_rmse = []
            mean_mae = []
            for train_loader, test_loader in gen:
                test_losses = pipe.train_and_evaluate(100, train_loader, test_loader)
                test_rmse, test_mae = sorted(test_losses, key=lambda x: x[2])[0][-2:]
                mean_rmse.append(test_rmse)
                mean_mae.append(test_mae)
            m1, m2 =np.mean(mean_rmse), np.mean(mean_mae)
            print('Percentile', percentile, 'GNN: ', m1, m2)
            with open('gin.txt', 'a+') as f:
                f.write("{} {} {}\n".format(percentile, m1, m2))
        except:
            pass



