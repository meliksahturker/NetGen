import pandas as pd
import numpy as np
from itertools import combinations 
import random
import networkx as nx
from scipy.stats import skewnorm

class Network:
    def __init__(self):
        """
        Example usage:
        net = Network()
        net.build_layer_one(1000, 3.9, 1.2, 1.7, 0.5)
        net.build_layer_two(0.21, 10, 5, 0, 1000, 0.4)
        net.build_layer_three(0.33, 10, 5, 0, 1000, 0.3)
        net.build_layer_four(0.25, 20, 2, 0, 1000, 3, 0.2)
        net.build_layer_five(10, 5, 0, 1000, 0.1)
        net.build_layer_six(0.10, 100, 20, 0, 1000, 0.05)
        net.build_layer_seven(50, 20, 0, 1000, 0.01)
        """
        self.G = nx.Graph()
        self.df = None
        self.df_works = None
        self.df_schools = None
    
    # Layer 1 - Households
    def build_layer_one(self, number_of_houses, a, loc, scale, beta_one):
        """
        number_of_houses (int): Number of houses
        a (float): alpha parameter in skewnorm distribution
        loc (float): location parameter in skewnorm distribution
        scale (float): scale parameter in skewnorm distribution
        beta_one: 𝛽_{1} transmission probability
        """
        
        # Create houses, and connect households within
        size_of_houses = []
        for i in range(number_of_houses):
            # household size varies between 1 and 13
            x = np.linspace(1, 13, 1000) 
            # household size follow a skewed Gaussian distribution with parameters fitted by skewnorm.fit as below:
            # a, loc, scale = 3.959, 1.220, 1.748
            probs = skewnorm.pdf(x, a, loc, scale)

            number = int(round(random.choices(x, probs, k = 1)[0]))
            size_of_houses.append(number)
            if number == 1:
                self.G.add_node(len(self.G))
            else:
                for pair in list(combinations(np.arange(len(self.G), len(self.G) + number) , 2)):
                    self.G.add_edge(pair[0], pair[1], weight = beta_one, layer = 1)
        
        # Create df
        # List of Vertices and Houses
        vertices = list(self.G.nodes())
        df = pd.DataFrame(vertices)
        df.columns = ['vertex']

        index_of_house = []
        for i in range(len(size_of_houses)):
            for j in range(size_of_houses[i]):
                index_of_house.append(i)

        df['house'] = index_of_house

        # Layer 2 - Blue
        df['color'] = np.nan
        df['work'] = -1
        df['distance_to_work'] = np.nan
        df['distance_to_work'] = df['distance_to_work'].astype('Int64')

        # Layer 3 - WhiteBlue
        # No initiation is needed here since no new variable is added to df table

        # Layer 4 - School
        df['school'] = -1
        df['distance_to_school'] = np.nan
        df['distance_to_school'] = df['distance_to_school'].astype('Int64')

        # Layer 5 - Friends
        df['number_of_friends'] = np.nan
        df['open_rooms_in_friends'] = np.nan
        df['list_of_friends'] = np.nan

        # Layer 6 - Service Industry
        df['customers'] = np.nan

        # Layer 7 - Random Encounters
        df['number_of_random_encounters'] = np.nan
        df['open_rooms_in_random_encounters'] = np.nan
        df['list_of_random_encounters'] = np.nan


        self.df = df
        
        # Layer 2 - Blue Work
    def build_layer_two(self, ratio_of_blue, mu_work_population, sigma_work_population, 
                        mu_distance_to_work, sigma_distance_to_work, beta_two):
        """
        ratio_of_blue: Ratio of blue vertices in the population
        mu_work_population: Average work population
        sigma_work_population: Standard deviation of work population
        mu_distance_to_work: Average distance between a blue vertex and its work
        sigma_distance_to_work: Standard deviation of distance between a blue vertex and its work
        beta_two: 𝛽_{2} transmission probability
        """

        df = self.df.copy()

        # Assign White-Blue flag to vertices according to ratio_of_blue
        size_of_houses = df.groupby('house').count()['vertex'].values
        list_of_colors = []
        for house_population in size_of_houses:
            for i in range(house_population):
                if np.random.rand() > ratio_of_blue:
                    color = 'W'
                else:
                    color = 'B'
                list_of_colors.append(color)

        df['color'] = list_of_colors


        list_of_works = []

        # Full employment
        while np.sum(list_of_works) < df[df['color'] == 'B'].shape[0]:
            work_size = round(random.gauss(mu_work_population, sigma_work_population))
            if work_size < 2:
                continue
            list_of_works.append(work_size)
        df_works = pd.DataFrame(list_of_works).reset_index()
        df_works.columns = ['work', 'population']
        df_works['open_jobs'] = df_works['population'].values.copy()

        ranges = (df_works['population'] * (df.shape[0] / df_works['population'].sum())).cumsum().round().astype(int)

        df_works['low_range'] = ranges.shift().fillna(0).astype(int)
        df_works['high_range'] = ranges
        df_works.iloc[-1, -1]+= 1 # otherwise I get out of bonds error for the last work

        # assign blue vertices to jobs
        for blue_vertex in df.loc[df['color'] == 'B', 'vertex'].values:
            list_of_available_jobs = df_works.loc[df_works['open_jobs'] != 0, 'work']

            distance_to_work = round(random.gauss(mu_distance_to_work, sigma_distance_to_work))

            work_location = blue_vertex + distance_to_work
            # Cycle around ring lattice
            while work_location >= df.shape[0]:
                work_location -= df.shape[0]
            while work_location < 0:
                work_location += df.shape[0]

            corresponding_work = df_works.loc[(df_works['low_range'] <= work_location) & (df_works['high_range'] > work_location), 'work'].values[-1]
            # if it's full, get closest one and assign to it
            assigned_work = min(list_of_available_jobs, key=lambda x:abs(x - corresponding_work)) 

            # calculate distance to assigned_work
            distance_to_work = round(df_works.loc[df_works['work'] == assigned_work, ['low_range', 'high_range']].mean(axis = 1).values[0]) - blue_vertex

            # reduce the number of open jobs by 1
            df_works.loc[df_works['work'] == assigned_work, 'open_jobs']-= 1

            # update the vertices table
            df.loc[df['vertex'] == blue_vertex, 'work'] = assigned_work
            df.loc[df['vertex'] == blue_vertex, 'distance_to_work'] = distance_to_work


        # Add edges with 𝛽 for people at same work
        df_workers = df[df['color'] == 'B']
        for item in df_workers[['vertex', 'work']].groupby('work'):
            for pair in list(combinations(item[1]['vertex'].values , 2)):
                if (not self.G.has_edge(pair[0], pair[1])) & (pair[0] != pair[1]):
                    self.G.add_edge(pair[0], pair[1], weight = beta_two, layer = 2)

        self.df = df
        self.df_works = df_works


    def build_layer_three(self, ratio_of_whiteblue, mu_work_population, sigma_work_population, 
                          mu_distance_to_work, sigma_distance_to_work, beta_three):
        """
        ratio_of_whiteblue: Ratio of white vertices that works (len(WB) / df.shape[0])
        mu_work_population: Average work population
        sigma_work_population: Standard deviation of work population
        mu_distance_to_work: Average distance between a white vertex and its work
        sigma_distance_to_work: Standard deviation of distance between a blue vertex and its work
        beta_two: 𝛽_{3} transmission probability
        """
        df = self.df.copy()
        df_works = self.df_works.copy()
        df_works.iloc[-1, -1]-= 1 # refixing before working on it
        
        ratio_of_blue = df['color'].value_counts(normalize = True).loc['B']
        probability_white_works = ratio_of_whiteblue / (1 - ratio_of_blue)

        # Re-label white colors that have jobs as WB
        for white_vertex in df.loc[df['color'] == 'W', 'vertex'].values:
            if probability_white_works > np.random.rand():
                df.loc[df['vertex'] == white_vertex, 'color'] = 'WB'


        list_of_works_wb = []

        # Full employment
        while np.sum(list_of_works_wb) < df[df['color'] == 'WB'].shape[0]:
            work_size = round(random.gauss(mu_work_population, sigma_work_population))
            if work_size < 2:
                continue
            list_of_works_wb.append(work_size)

        df_works_wb = pd.DataFrame(list_of_works_wb).reset_index()
        df_works_wb.columns = ['work', 'population']
        df_works_wb['open_jobs'] = df_works_wb['population'].values.copy()
        ranges = (df_works_wb['population'] * (df.shape[0] / df_works_wb['population'].sum())).cumsum().round().astype(int)
        df_works_wb['low_range'] = ranges.shift().fillna(0).astype(int)
        df_works_wb['high_range'] = ranges
        df_works_wb.iloc[-1, -1]+= 1 # otherwise I get out of bonds error for the last work

        # re-index newly created works
        df_works_wb.loc[:, 'work'] = np.arange(df_works.shape[0], df_works_wb.shape[0] + df_works.shape[0])

        # Merge two df_works table under one
        df_works = pd.concat([df_works, df_works_wb])
        df_works.reset_index(drop = True, inplace = True)


        for whiteblue_vertex in df.loc[df['color'] == 'WB', 'vertex'].values:
            list_of_available_jobs = df_works.loc[df_works['open_jobs'] != 0, 'work']

            distance_to_work = round(random.gauss(mu_distance_to_work, sigma_distance_to_work))

            work_location = whiteblue_vertex + distance_to_work
            # Cycle around ring lattice
            while work_location >= df.shape[0]:
                work_location -= df.shape[0]
            while work_location < 0:
                work_location += df.shape[0]

            corresponding_work = df_works.loc[(df_works['low_range'] <= work_location) & (df_works['high_range'] > work_location), 'work'].values[-1]
            # if it's full, get closest one and assign to it
            assigned_work = min(list_of_available_jobs, key=lambda x:abs(x - corresponding_work)) 

            # calculate distance to assigned_work
            distance_to_work = round(df_works.loc[df_works['work'] == assigned_work, ['low_range', 'high_range']].mean(axis = 1).values[0]) - whiteblue_vertex

            # reduce the number of open jobs by 1
            df_works.loc[df_works['work'] == assigned_work, 'open_jobs']-= 1

            # update the vertices table
            df.loc[df['vertex'] == whiteblue_vertex, 'work'] = assigned_work
            df.loc[df['vertex'] == whiteblue_vertex, 'distance_to_work'] = distance_to_work

        # Add edges with 𝛽 for people at same work
        df_workers = df[df['color'] == 'WB']
        for item in df_workers[['vertex', 'work']].groupby('work'):
            for pair in list(combinations(item[1]['vertex'].values , 2)):
                if not self.G.has_edge(pair[0], pair[1]):
                    self.G.add_edge(pair[0], pair[1], weight = beta_three, layer = 3)

        self.df = df
        self.df_works = df_works
        
    def build_layer_four(self, ratio_of_school, mu_school_population, sigma_school_population, 
                         mu_distance_to_school, sigma_distance_to_school, number_of_teachers_per_school, 
                         beta_four):
        """
        ratio_of_school: Ratio of students (len(S) / df.shape[0]) in the population
        mu_school_population: Mean school population
        sigma_school_population: Standard deviation of school population
        mu_distance_to_school: Average distance between a student and its school
        sigma_distance_to_school: Standard deviation of distance between a student and its school
        number_of_teachers_per_school: number of teachers that work in a school (teachers are assigned among workers)
        beta_four: 𝛽_{4} transmission probability
        """
    
        df = self.df.copy()
        df_works = self.df_works.copy()
        
        percentage_of_white = df['color'].value_counts(normalize = True).loc['W']
        probability_white_student = ratio_of_school / percentage_of_white
        
        # Re-label white colors that will go to school as S
        for white_vertex in df.loc[df['color'] == 'W', 'vertex'].values:
            if probability_white_student > np.random.rand():
                df.loc[df['vertex'] == white_vertex, 'color'] = 'S'
                
        list_of_schools = []

        # Full school attendance for all S
        while np.sum(list_of_schools) < df[df['color'] == 'S'].shape[0]:
            school_size = round(random.gauss(mu_school_population, sigma_school_population))
            if school_size < 2:
                continue
            list_of_schools.append(school_size)

        df_schools = pd.DataFrame(list_of_schools).reset_index()
        df_schools.columns = ['school', 'population']
        df_schools['open_rooms'] = df_schools['population'].values.copy()
        ranges = (df_schools['population'] * (df.shape[0] / df_schools['population'].sum())).cumsum().round().astype(int)
        df_schools['low_range'] = ranges.shift().fillna(0).astype(int)
        df_schools['high_range'] = ranges
        df_schools.iloc[-1, -1]+= 1 # otherwise I get out of bonds error for the last school
        
        # Assign students
        for s_vertex in df.loc[df['color'] == 'S', 'vertex'].values:
            list_of_available_schools = df_schools.loc[df_schools['open_rooms'] != 0, 'school']

            distance_to_school = round(random.gauss(mu_distance_to_school, sigma_distance_to_school))

            school_location = s_vertex + distance_to_school
            # Cycle around ring lattice
            while school_location >= df.shape[0]:
                school_location -= df.shape[0]
            while school_location < 0:
                school_location += df.shape[0]

            corresponding_school = df_schools.loc[(df_schools['low_range'] <= school_location) & (df_schools['high_range'] > school_location), 'school'].values[-1]
            # if it's full, get closest one and assign to it
            assigned_school = min(list_of_available_schools, key=lambda x:abs(x - corresponding_school)) 

            # calculate distance to assigned_school
            distance_to_school = round(df_schools.loc[df_schools['school'] == assigned_school, ['low_range', 'high_range']].mean(axis = 1).values[0]) - s_vertex

            # reduce the number of open rooms by 1
            df_schools.loc[df_schools['school'] == assigned_school, 'open_rooms']-= 1

            # update the vertices table
            df.loc[df['vertex'] == s_vertex, 'school'] = assigned_school
            df.loc[df['vertex'] == s_vertex, 'distance_to_school'] = distance_to_school
        
        # Assign teachers
        for school in df_schools['school'].values:
            # determine location of school according to mean of its range
            location_of_school = df_schools.loc[df_schools['school'] == school, ['low_range', 'high_range']].mean(axis = 1).values[0]

            # determine closest work to the school where at least number_of_teachers_per_school number of workers work, but are also available to be assigned to the school
            # potential_schools must contain at least as number of available workers as the number_of_teachers_per_school
            # otherwise we end up with a work place with not enough teachers and have sampling from an empty list error
            potential_schools = df.loc[(df['school'] == -1) & (df['work'] != -1) & ((df['color'] == 'B') | (df['color'] == 'WB')), ['vertex', 'work']].groupby('work').count()
            potential_schools = potential_schools[potential_schools['vertex'] >= number_of_teachers_per_school].index.values
            closest_work_to_school = (df_works.loc[potential_schools, ['low_range', 'high_range']].mean(axis = 1) - location_of_school).abs().idxmin()
                
            # randomly select number_of_teachers_per_school vertices from this work
            teachers_assigned = df.loc[(df['work'] == closest_work_to_school) & (df['school'] == -1) & ((df['color'] == 'B') | (df['color'] == 'WB')), 'vertex'].sample(number_of_teachers_per_school, replace = True).values

            # assign these vertices as teachers to school, by updating school column in df table
            df.loc[df['vertex'].isin(teachers_assigned), 'school'] = school
            
            
        # Add edges with 𝛽 for people at same school
        df_schoolers = df[df['school'] != -1]
        for item in df_schoolers[['vertex', 'school']].groupby('school'):
            for pair in list(combinations(item[1]['vertex'].values , 2)):
                if (not self.G.has_edge(pair[0], pair[1])) & (pair[0] != pair[1]):
                    self.G.add_edge(pair[0], pair[1], weight = beta_four, layer = 4)
                    
        # Set df and df_schools
        self.df = df
        self.df_works = df_works
        self.df_schools = df_schools
        
        
    def build_layer_five(self, mu_number_of_friends, sigma_number_of_friends, 
                         mu_distance_to_friend, sigma_distance_to_friend, 
                         beta_five):
        """
        mu_number_of_friends: Average number of friends a vertex has
        sigma_number_of_friends: Standard deviation of number of friends a vertex has
        mu_distance_to_friend: Average distance between a vertex and its friend
        sigma_distance_to_friend: Standard deviation of distance between a vertex and its friend
        beta_five: 𝛽_{5} transmission probability
        """

        df = self.df.copy()
        df['number_of_friends'] = df['number_of_friends'].astype('Int64')
        df['open_rooms_in_friends'] = df['number_of_friends'].copy()
        df['list_of_friends'] = [[] for i in range(df.shape[0])]
        
        # Determine number of friends each vertex has and write it to df table
        for vertex in df['vertex'].values:
            number_of_friends = abs(round(random.gauss(mu_number_of_friends, sigma_number_of_friends)))
            df.loc[vertex, 'number_of_friends'] = number_of_friends
            df.loc[vertex, 'open_rooms_in_friends'] = number_of_friends
            
            
        # Assign and connect friends
        # while there are still rooms that must be filled with friends
        while df['open_rooms_in_friends'].sum() > 0:
            for vertex in df.loc[df['open_rooms_in_friends']> 0, 'vertex'].values:

                # if this vertex was befriended by another in the for loop
                if (df.loc[vertex, 'open_rooms_in_friends'] == 0):
                    continue

                # it makes more sense for students to be friends with students
                color_of_vertex = df.loc[vertex, 'color']
                if color_of_vertex == 'S':
                    potential_friends = df.loc[(df['vertex'] != vertex) & (df['color'] == 'S') & (df['open_rooms_in_friends'] > 0), 'vertex'].values
                else:
                    potential_friends = df.loc[(df['vertex'] != vertex) & (df['color'] != 'S') & (df['open_rooms_in_friends'] > 0), 'vertex'].values

                # if no one left in the network to make friends with, just close open rooms for this vertex
                if len(potential_friends) == 0:
                    df.loc[vertex, 'open_rooms_in_friends'] = 0
                    continue

                # Determine distance
                distance_to_friend = round(random.gauss(mu_distance_to_friend, sigma_distance_to_friend))
                friend_location = vertex + distance_to_friend

                # Cycle around ring lattice
                while friend_location >= df.shape[0]:
                    friend_location -= df.shape[0]
                while friend_location < 0:
                    friend_location += df.shape[0]

                # determine the closest one among potential friends
                idx_of_friend = np.argmin(abs(potential_friends - friend_location))
                friend = potential_friends[idx_of_friend]

                # if friends is already fulfilled in for loop by another vertex:
                # if (df.loc[vertex, 'open_rooms_in_friends'] == 0) | (df.loc[friend, 'open_rooms_in_friends'] == 0):
                #    continue

                # update metadata
                df.loc[vertex, 'open_rooms_in_friends'] -=1
                df.loc[friend, 'open_rooms_in_friends'] -=1
                df.loc[vertex, 'list_of_friends'].append(friend)
                df.loc[friend, 'list_of_friends'].append(vertex)

        # Add edges between friends
        for vertex in df['vertex'].values:
            friends_of_vertex = df.loc[vertex, 'list_of_friends']
            for friend in friends_of_vertex:
                pair = (vertex, friend)
                if (not self.G.has_edge(pair[0], pair[1])) & (pair[0] != pair[1]):
                    self.G.add_edge(pair[0], pair[1], weight = beta_five, layer = 5)
                    
        self.df = df
        
    def build_layer_six(self, ratio_of_service_industry, mu_number_of_service_industry_connections, sigma_number_of_service_industry_connections, 
                         mu_distance_service_industry_connections, sigma_distance_service_industry_connections, 
                         beta_six):        
        """
        ratio_of_service_industry: Ratio of service industry workers (len(BSI) / df.shape[0]) in the population
        mu_number_of_service_industry_connections: Average number of connections a service industry worker makes with customers
        sigma_number_of_service_industry_connections: Standard deviation of number of connections a service industry worker makes with customers
        mu_distance_service_industry_connections: Average distance between a service industry worker and its customer
        sigma_distance_service_industry_connections: Standard deviation of distance between a service industry worker and its customer
        beta_six: 𝛽_{6} transmission probability
        """
        df = self.df.copy()
        df_works = self.df_works.copy()
        
        percentage_of_service_industry = df['color'].value_counts(normalize = True).loc['B']
        probability_service_industry = ratio_of_service_industry / percentage_of_service_industry

        df['customers'] = [[] for i in range(df.shape[0])]
        
        # Re-label some of the blue as service industry who doesn't serve as teacher
        for blue_vertex in df.loc[(df['color'] == 'B') & (df['school'] == -1), 'vertex'].values:
            if probability_service_industry > np.random.rand():
                df.loc[df['vertex'] == blue_vertex, 'color'] = 'BSI'
        
        # match servers with customers
        for bsi_vertex in df.loc[df['color'] == 'BSI', 'vertex']:
            # determine the place this bsi vertex works in
            work_location = bsi_vertex + df.loc[bsi_vertex, 'distance_to_work']

            # Cycle around ring lattice
            while work_location >= df.shape[0]:
                work_location -= df.shape[0]
            while work_location < 0:
                work_location += df.shape[0]

            # determine number of connections this bsi vertex will make
            number_of_service_industry_connections = round(random.gauss(mu_number_of_service_industry_connections, sigma_number_of_service_industry_connections))
            while (number_of_service_industry_connections < 1):
                number_of_service_industry_connections = round(random.gauss(mu_number_of_service_industry_connections, sigma_number_of_service_industry_connections))

            # determine customers to be served
            for _ in range(number_of_service_industry_connections):
                distance_to_customer = round(random.gauss(mu_distance_service_industry_connections, sigma_distance_service_industry_connections))
                customer_location = bsi_vertex + distance_to_customer
                # Cycle around ring lattice
                while customer_location >= df.shape[0]:
                    customer_location -= df.shape[0]
                while customer_location < 0:
                    customer_location += df.shape[0]
                customer = customer_location
                df.loc[bsi_vertex, 'customers'].append(customer)
                
        # Add edges between service industry and customers
        for bsi_vertex in df.loc[df['color'] == 'BSI', 'vertex'].values:
            customers = df.loc[bsi_vertex, 'customers']
            for customer in customers:
                pair = (bsi_vertex, customer)
                if (not self.G.has_edge(pair[0], pair[1])) & (pair[0] != pair[1]):
                    self.G.add_edge(pair[0], pair[1], weight = beta_six, layer = 6)
                    
        self.df = df
        self.df_works = df_works
        
        
    def build_layer_seven(self, mu_number_of_random_encounters, sigma_number_of_random_encounters, 
                         mu_distance_to_random_encounter, sigma_distance_to_random_encounter, 
                         beta_seven):
        """
        mu_number_of_random_encounters: Average number of random encounters a vertex makes
        sigma_number_of_random_encounters: Standard deviation of number of random encounters a vertex makes
        mu_distance_to_random_encounter: Average distance between a vertex and its random encounter
        sigma_distance_to_random_encounter: Standard deviation of distance between a vertex and its random encounter
        beta_seven: 𝛽_{7} transmission probability
        """
        
        df = self.df.copy()
        df['number_of_random_encounters'] = df['number_of_random_encounters'].astype('Int64')
        df['open_rooms_in_random_encounters'] = df['number_of_random_encounters'].copy()
        df['list_of_random_encounters'] = [[] for i in range(df.shape[0])]
        
        # Determine number of random encounters each vertex does and write it to df table
        for vertex in df.loc[:, 'vertex'].values:
            number_of_random_encounters = abs(round(random.gauss(mu_number_of_random_encounters, sigma_number_of_random_encounters)))
            df.loc[vertex, 'number_of_random_encounters'] = number_of_random_encounters
            df.loc[vertex, 'open_rooms_in_random_encounters'] = number_of_random_encounters
            
            
        # Assign and connect random encounters
        # while there are still rooms that must be filled with random encounters
        while df['open_rooms_in_random_encounters'].sum() > 0:
            for vertex in df.loc[df['open_rooms_in_random_encounters']> 0, 'vertex'].values:

                # if this vertex was encountered and exhausted by vertex during the for loop
                if (df.loc[vertex, 'open_rooms_in_random_encounters'] == 0):
                    continue

                potential_encounters = df.loc[(df['vertex'] != vertex) & (df['open_rooms_in_random_encounters'] > 0), 'vertex'].values

                # if no one left in the network to make encounter with, just close open rooms for this vertex
                if len(potential_encounters) == 0:
                    df.loc[vertex, 'open_rooms_in_random_encounters'] = 0
                    continue

                # Determine distance
                distance_to_random_encounter = round(random.gauss(mu_distance_to_random_encounter, sigma_distance_to_random_encounter))
                encounter_location = vertex + distance_to_random_encounter

                # cycle around the lattice
                while encounter_location > df.shape[0]:
                    encounter_location -= df.shape[0]
                while encounter_location < 0:
                    encounter_location += df.shape[0]

                # determine the closest one among potential encounters
                idx_of_encounter = np.argmin(abs(potential_encounters - encounter_location))
                encounter = potential_encounters[idx_of_encounter]

                # update metadata
                df.loc[vertex, 'open_rooms_in_random_encounters'] -=1
                df.loc[encounter, 'open_rooms_in_random_encounters'] -=1
                df.loc[vertex, 'list_of_random_encounters'].append(encounter)
                df.loc[encounter, 'list_of_random_encounters'].append(vertex)

        # Add edges between random encounters
        for vertex in df.loc[:, 'vertex'].values:
            encounters_of_vertex = df.loc[vertex, 'list_of_random_encounters']
            for encounter in encounters_of_vertex:
                pair = (vertex, encounter)
                if (not self.G.has_edge(pair[0], pair[1])) & (pair[0] != pair[1]):
                    self.G.add_edge(pair[0], pair[1], weight = beta_seven, layer = 7)
                    
        self.df = df
