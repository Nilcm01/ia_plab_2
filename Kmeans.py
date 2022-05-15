__authors__ = ['1565175','1566740',]
__group__ = 'DM.10'

from random import random
import numpy as np
import utils


class KMeans:

    # NOT MODIFIED
    def __init__(self, X, K=1, options=None):
        """
         Constructor of KMeans class
             Args:
                 K (int): Number of cluster
                 options (dict): dictºionary with options
            """
        self.num_iter = 0
        self.K = K
        self._init_X(X)
        self._init_options(options) # DICT options

    #############################################################
    ##  THIS FUNCTION CAN BE MODIFIED FROM THIS POINT, if needed
    #############################################################


    # DONE + COMMENTED
    def _init_X(self, X):
        """Initialization of all pixels, sets X as an array of data in vector form (PxD)
            Args:
                X (list or np.array): list(matrix) of all pixel values
                    if matrix has more than 2 dimensions, the dimensionality of the smaple space is the length of
                    the last dimension
        """
        #######################################################
        ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
        ##  AND CHANGE FOR YOUR OWN CODE
        #######################################################
        #self.X = np.random.rand(100, 5)

        """1r: Assegurar-se que tots els valors siguin de tipus float"""
        # if type of X is not float -> save array as
        if X.dtype != np.float32: X = np.asfarray(X)

        """
        2n: Si cal, converteix les dades a una matriu de només dues dimensions NxD
        Si l'entrada és una imatge en matriu de dimensions FxCx3 llavors:
        -> transformar-la i retornar els píxels en una matriu de Nx3 i guarda-la
        a la variable X: self.X
        """
        if X.shape[2] != 2: X = np.reshape(X, (X.shape[0] * X.shape[1], X.shape[2]))
        
        # Save the new X
        self.X = X


    # NOT EDITED
    def _init_options(self, options=None):
        """
        Initialization of options in case some fields are left undefined
        Args:
            options (dict): dictionary with options
        """
        if options == None:
            options = {}
        if not 'km_init' in options:
            options['km_init'] = 'first'
        if not 'verbose' in options:
            options['verbose'] = False
        if not 'tolerance' in options:
            options['tolerance'] = 0
        if not 'max_iter' in options:
            options['max_iter'] = np.inf
        if not 'fitting' in options:
            options['fitting'] = 'WCD'  #within class distance.

        # If your methods need any other prameter you can add it to the options dictionary
        self.options = options

        #############################################################
        ##  THIS FUNCTION CAN BE MODIFIED FROM THIS POINT, if needed
        #############################################################


    # DONE + COMMENTED
    def _init_centroids(self):
        """
        Initialization of centroids
        """

        #######################################################
        ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
        ##  AND CHANGE FOR YOUR OWN CODE
        #######################################################
        #if self.options['km_init'].lower() == 'first':
        #    self.centroids = np.random.rand(self.K, self.X.shape[1])
        #    self.old_centroids = np.random.rand(self.K, self.X.shape[1])
        #else:
        #    self.centroids = np.random.rand(self.K, self.X.shape[1])
        #    self.old_centroids =np.random.rand(self.K, self.X.shape[1])

        """
        Inicialitza centroids i old_centroids
        Les dues variables tenen mida KxD, on:
            K => Nº de centroides passats a la classe KMeans
            D => Nº de canals
        
        Opcions:
            First: assigna als centroides els primers K punts de la imatge X
                    que siguin diferents entre ells. (opció per defecte)
            Random: triarà, de forma que no es repeteixin, centroides a l'atzar.
            Custom: seguir qualsevol política de selecció inicial que vulguem
        """

        # Assign the first different K points of X
        if self.options['km_init'] == 'first':

            # Create a matrix to store al unique centroids
            uniq_cent = np.zeros(shape = (self.K, 3))
            pos = 0

            # Iterate thruogh every value (centroid)
            for value in self.X:

                check = False
                i = 0

                # Iterate through every value in the new matrix
                while i < len(uniq_cent):

                    # If values from row == 0 -> value should be added
                    if uniq_cent[i][0] == value[0] and uniq_cent[i][1] == value[1] and uniq_cent[i][2] == value[2]: 

                        # Exit the iteration with confirmation
                        check = True
                        i = len(uniq_cent)

                    i += 1

                # If all values from a row are 0 or previous iteration exited with success
                # -> save row to the new matrix
                if check != True or (value[0] == 0 and value[1] == 0 and value[2] == 0):

                    uniq_cent[pos] = value
                    pos += 1

                    # If position == K -> end for loop
                    if pos == self.K: break
                
            # Save the new matrix to the centroids matrix
            self.centroids = uniq_cent

            # Save an empty matrix the same size as the new one to the old centroids matrix
            self.old_centroids = np.empty(shape = (self.K, 3))

        # Random assigned centroids
        elif self.options['km_init'] == 'random':

            # Save shape for new matrixes
            shape = self.X.shape[1]

            # Save random assigned centroids to old centroids matrix
            self.old_centroids = random(self.K, shape)

            # Save random assigned centroids to centroids matrix
            self.centroids = random(self.K, shape)

        # Custom assignation
        elif self.options['km_init'] == 'custom':
            pass

        # No assignation defined or not
        else:
            pass


    # DONE + COMMENTED
    def get_labels(self):
        """        Calculates the closest centroid of all points in X
        and assigns each point to the closest centroid
        """
        #######################################################
        ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
        ##  AND CHANGE FOR YOUR OWN CODE
        #######################################################
           
        # Get distances between each pixel and the closest centroid
        # and gets the position of the minimum value of the matrix
        # from the axis 1 (horizontal -> row)
        self.labels = np.argmin(distance(self.X, self.centroids), axis = 1) 


    # DONE + COMMENTED
    def get_centroids(self):
        """
        Calculates coordinates of centroids based on the coordinates of all the points assigned to the centroid
        """
        #######################################################
        ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
        ##  AND CHANGE FOR YOUR OWN CODE
        #######################################################

        # Saves an array of the centroids to old centroids
        self.old_centroids = np.array(self.centroids)

        # For each value in every column calculate the average and saves it to centroids
        for i in range(self.K): self.centroids[i] = np.mean(self.X[self.labels == i], axis = 0)
        
    
    # DONE + COMMENTED
    def converges(self):
        """
        Checks if there is a difference between current and old centroids
        """
        #######################################################
        ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
        ##  AND CHANGE FOR YOUR OWN CODE
        #######################################################
        
        # Returns if the two matrixes are equal within a tolerance defined in options
        return np.allclose(self.centroids, self.old_centroids, atol = self.options['tolerance'])


    # DONE + COMMENTED
    def fit(self):
        """
        Runs K-Means algorithm until it converges or until the number
        of iterations is smaller than the maximum number of iterations.
        """
        #######################################################
        ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
        ##  AND CHANGE FOR YOUR OWN CODE
        #######################################################

        # Initialize centroids
        self.num_iter = 0
        self._init_centroids()

        # Loop a maximum of times defined by the options
        while self.num_iter < self.options['max_iter']:

            # Calculates the closest centroid of all points in X and assigns each point to the closest centroid
            self.get_labels()

            # Calculates the closest centroid of all points in X and assigns each point to the closest centroid
            self.get_centroids()

            # Continue the loop
            self.num_iter += 1

            # If there is a difference between current and old centroids -> exit while loop
            if (self.converges() == True): break


    # DONE + COMMENTED
    def whitinClassDistance(self):
        """
         returns the whithin class distance of the current clustering
        """
                
        #######################################################
        ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
        ##  AND CHANGE FOR YOUR OWN CODE
        #######################################################
        
        wcd = 0.0

        # For each value until K
        # -> sum all values on the horitzontal axis from the resultant matrixes
        # and sum it to the current value
        for i in range(self.K): wcd += np.sum(np.sum((((self.X[self.labels == i])-self.centroids[i])**2), axis = 1))

        # Save the WCD calcuated divided by the shape of X and return it
        self.WCD = wcd / self.X.shape[0]
        return self.WCD
    

    # DONE + COMMENTED
    def find_bestK(self, max_K):
        """
         sets the best k anlysing the results up to 'max_K' clusters
        """          
                
        #######################################################
        ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
        ##  AND CHANGE FOR YOUR OWN CODE
        #######################################################
        
        # Initialize values
        self.K = 2
        r = 0.0

        # Runs K-Means algorithm until it converges or until ->
        # the number of iterations is smaller than the maximum number of iterations
        self.fit()

        # Returns the whithin class distance of the current clustering
        self.whitinClassDistance()

        # Iterate thrugh the range
        for i in range(3, max_K+1):

            # Save K as the index of the iteration
            self.K = i

            # Save the current WCD as the old one
            oldWCD = self.WCD

            # Runs K-Means algorithm until it converges or until ->
            # the number of iterations is smaller than the maximum number of iterations
            self.fit()

            # If the index != max K -> operate
            if i != max_K:

                # Returns the whithin class distance of the current clustering
                self.whitinClassDistance()

                # 100 minus the percentual diference of WCD is less than 20 -> end
                if 100 - (100 * (self.WCD/oldWCD)) < 80:

                    # Save K as the index minus 1
                    self.K = i - 1

                    # Runs K-Means algorithm until it converges or until ->
                    # the number of iterations is smaller than the maximum number of iterations
                    self.fit()

                    # Returns K
                    return self.K

            # Else case -> return the max K value
            else: return max_K
                

# DONE + COMMENTED
def distance(X, C):
    """
    Calculates the distance between each pixcel and each centroid
    Args:
        X (numpy array): PxD 1st set of data points (usually data points)
        C (numpy array): KxD 2nd set of data points (usually cluster centroids points)

    Returns:
        dist: PxK numpy array position ij is the distance between the
        i-th point of the first set an the j-th point of the second set
    """

    #########################################################
    ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
    ##  AND CHANGE FOR YOUR OWN CODE
    #########################################################

    # Save the shapes of C and X in a tuple
    shapedm = C.shape[0], X.shape[0]

    # Create a matrix full of 0 in the shape saved
    distances = np.zeros(shape = shapedm)

    # For each value in the range of C -> save the values in the new matrix
    # from: the square root of the sum of the element-wise square of X minus C[i]
    # on the horizontal axis
    for i in range(len(C)): distances[i] = np.sqrt(np.sum(np.square(X-C[i]), axis=1))
        
    # Return the transposed matrix
    return np.transpose(distances)


# DONE + COMMENTED
def get_colors(centroids):
    """
    for each row of the numpy matrix 'centroids' returns the color laber folllowing the 11 basic colors as a LIST
    Args:
        centroids (numpy array): KxD 1st set of data points (usually centroind points)

    Returns:
        lables: list of K labels corresponding to one of the 11 basic colors
    """

    #########################################################
    ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
    ##  AND CHANGE FOR YOUR OWN CODE
    #########################################################

    # Gets the maximum value from the horizontal axis to get the closest color
    return utils.colors[utils.get_color_prob(centroids).argmax(axis = 1)]
