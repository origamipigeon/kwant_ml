import numpy as np
from scipy import stats 			  		 			 	 	 		 		 	  		   	  			  	
import baseLearner


class BagLearner(baseLearner):

    def __init__(self, learner, kwargs={}, bags=5, boost=False, verbose=False):
        self.learner = learner
        self.array_learners = []
        self.kwargs = kwargs
        self.bags=bags
        self.boost = boost
        self.verbose = verbose	

        for i in range(0, self.bags):
            self.array_learners.append(learner(**kwargs))			 	 	 		 		 	  		   	  			  	

    def addEvidence(self, dataX, dataY):
        """ 			  		 			 	 	 		 		 	  		   	  			  	
        @summary: Add training data to learner 			  		 			 	 	 		 		 	  		   	  			  	
        @param dataX: X values of data to add 			  		 			 	 	 		 		 	  		   	  			  	
        @param dataY: the Y training values 			  		 			 	 	 		 		 	  		   	  			  	
        """
        index_list = dataX.shape[0]

        for learner in self.array_learners:
            bagX = []
            bagY = []
            for i in range(0, index_list):
                index = np.random.randint(0, dataX.shape[0]-1)
                tempX = dataX.ix[index, :]
                tempY = dataY.ix[index]
                bagX.append(tempX)
                bagY.append(tempY)
            
            learner.addEvidence(bagX, bagY)

    def query(self, points):
        """ 			  		 			 	 	 		 		 	  		   	  			  	
        @summary: Estimate a set of test points given the model we built. 			  		 			 	 	 		 		 	  		   	  			  	
        @param points: should be a numpy array with each row corresponding to a specific query. 			  		 			 	 	 		 		 	  		   	  			  	
        @returns the estimated values according to the saved model. 			  		 			 	 	 		 		 	  		   	  			  	
        """ 		
        
        # need to search the tree for a the predicted leaf value
        query_list = []
        for learner in self.array_learners:
            query_list.append(learner.query(points))
            
        query_array = np.array(query_list)

        answers = stats.mode(query_array).mode[0]

        return answers.tolist()		  		 			 	 	 		 		 	  		   	  			  	


if __name__ == "__main__":
    None
