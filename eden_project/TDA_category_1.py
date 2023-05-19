import math
import numpy as np
class categoryLosses:
    def __init__(self,data):
        self.map = data
        self.cat_2_distances = []
        self.cat_3_distances = []
        self.all_changed_parcels = []
        self.cur_cat_2_min = 0
        self.solutions = []
    def category_loss_2(self,solution:np.array,parcel_params:np.array,land_type:int):
        self.all_changed_parcels.append(parcel_params) # change parcel to changed there will be more than 1
        self.solutions.append(solution)
        # all currently changed parcels distance to specific


        # d_p_p_j =np.array([math.dist(parcel,x) for x in self.find_specific_land_types(land_typ,solution) for parcel in self.parcels_to_consider])
        all_shortest_disntaces = []

        for parcel in self.parcels_to_consider:
            temp_min_value = []
            # find all distance for currently consider singular parcel to all land_types
            for desired_land_type in self.find_specific_land_types(land_type, solution):
                temp_min_value.append(math.dist(parcel,desired_land_type))
            #save the shortest_one
            all_distances= np.array(temp_min_value)
            d_p_p_j_min = all_distances.min()
            self.cat_2_distances.append(d_p_p_j_min)
            d_p_p_j_sum  = all_distances.sum()
            d_p_p_j_min = all_distances.min()

            self.cat_2_distances.append(d_p_p_j_sum)

            d_p_p_j_global_min = np.array(self.cat_2_distances).min()



            formula = (d_p_p_j_global_min/d_p_p_j_sum) * (d_p_p_j_min.min())
        self.parcels_to_consider = [] # Set the list to empty once they are all considered

    def category_loss_2(self, solution: np.array, parcel_params: np.array, land_type: int):
    def set_parcels_to_development(self,parcels:list):
        self.parcels_to_consider = parcels



    def find_specific_land_types(self,value,solution):
        test_list = solution
        value = value
        arr =  np.where(test_list == value)

        x_param,y_param = arr
        points = [np.array(x,y) for x,y in zip(x_param,y_param)]
        return points
