import numpy as np
import baseLearner


class DTLearner(baseLearner):

    def __init__(self, leaf_size=1, verbose = False): 			  		 			 	 	 		 		 	  		   	  			  	
        self.leaf_size = leaf_size
        self.verbose = verbose	
        self.decision_tree = None		

        # for debug
        # np.seterr(all='raise')
        np.seterr(divide='ignore', invalid='ignore')  		 			 	 	 		 		 	  		   	  			  	

    def addEvidence(self, dataX, dataY):
        """ 			  		 			 	 	 		 		 	  		   	  			  	
        @summary: Add training data to learner 			  		 			 	 	 		 		 	  		   	  			  	
        @param dataX: X values of data to add 			  		 			 	 	 		 		 	  		   	  			  	
        @param dataY: the Y training values 			  		 			 	 	 		 		 	  		   	  			  	
        """ 			  		 			 	 	 		 		 	  		   	  			  	

        # amalgamate dataY to last column
        dataY = np.array([dataY])
        # get transpose / inverse dimension array
        new_Y = dataY.T
        data = np.append(dataX, new_Y, axis=1)	  			  	
        self.decision_tree = self.build_tree(data) 	
        if self.verbose:
            print(self.decision_tree)

    def query(self, points):
        """ 			  		 			 	 	 		 		 	  		   	  			  	
        @summary: Estimate a set of test points given the model we built. 			  		 			 	 	 		 		 	  		   	  			  	
        @param points: should be a numpy array with each row corresponding to a specific query. 			  		 			 	 	 		 		 	  		   	  			  	
        @returns the estimated values according to the saved model. 			  		 			 	 	 		 		 	  		   	  			  	
        """ 		
        
        # need to search the tree for a the predicted leaf value
        estimated_values = []
        for i in range(0, len(points)):
            estimated_values.append(self.search_tree(points[i]))	  		 			 	 	 		 		 	  		   	  			  	
        return np.asarray(estimated_values)
    
    def search_tree(self, point):
        """
        @summary: search the tree using our features for our prediction estimate. 			  		 			 	 	 		 		 	  		   	  			  	
        @param point: a numpy array representing our feature data to match. 			  		 			 	 	 		 		 	  		   	  			  	
        @returns the estimated values according to the saved model. 			  		 			 	 	 		 		 	  		   	  			  	
        """ 	
        
        # start row is root node
        row = 0
        feature, split_value, left, right = self.decision_tree[row, :]
        
        while int(feature) != -1:
            if point[int(feature)] <= split_value:
                row = row + int(left)
            else:
                row = row + int(right)
                
            feature, split_value, left, right = self.decision_tree[row, :]
        return split_value

    def build_tree(self, data):
        """ 			  		 			 	 	 		 		 	  		   	  			  	
        @summary: Recursively build a tree by splitting on a value and making the two branches. 			  		 			 	 	 		 		 	  		   	  			  	
        @param data: numpy ndarray with our features.
        @returns:  numpy ndarray with each row in format of [Factor, SplitVal, LeftIndex, RightIndex]			  		 			 	 	 		 		 	  		   	  			  	
        """ 	
        
        current_leaf_value = np.mean(data[:,-1])
        
        if data.shape[0] <= self.leaf_size:
            # max leaf size num rows, return mean    
            return np.array([[int(-1), current_leaf_value, np.nan, np.nan]])
        
        # TODO: check all data is the same
        if data[0, -1] == data[-1,-1]:
            return np.array([[int(-1), current_leaf_value, np.nan, np.nan]])

        # calc the absolute (in-case best val is negative) of the corrolation
        # of each of the features
        dataY = data[:,-1]
        dataX = data[:,0:-1]
        np.seterr(divide='ignore', invalid='ignore')
        corrs = np.corrcoef(dataX, y=dataY, rowvar=False)
        # print "Corrs: ", corrs
        corrs = np.abs(corrs)[:-1,-1]
        
        if self.verbose:
            print(corrs)
            
        # highest ignoring nans
        split_feature = np.nanargmax(corrs)

        split_val = np.median(data[:, split_feature])
        
        if max(data[:,split_feature]) == split_val:
            return np.array([[int(-1), current_leaf_value, np.nan, np.nan]])
        
        left_tree = self.build_tree(data[data[:,split_feature] <= split_val])
        
        # print(left_tree)
        right_tree = self.build_tree(data[data[:,split_feature] > split_val])
        # print(right_tree)
        tree_root = np.array([[int(split_feature), split_val, int(1), int(left_tree.shape[0]+1)]])
        # print(tree_root)
        root = np.append(tree_root, left_tree,axis=0)

        return np.append(root, right_tree, axis=0)        			 	 	 		 		 	  		   	  			  	


if __name__ == "__main__":
    None
