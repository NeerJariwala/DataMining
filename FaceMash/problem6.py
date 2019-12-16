import pandas as pd
import numpy as np
from problem3 import filtering, sum_column
from problem5 import runs_created
#-------------------------------------------------------------------------
'''
    Problem 6: Player selection for Oakland A's Team (OAK)
    In this problem, you will choose baseball players for Oakland A's using different methods.
'''

#--------------------------
def sum_salaries(T, D):
    '''
        Given a team of players (T), compute the sum of salaries of all the players in the team.
        Input:
            T: a python list of playerID's, for example, ['zitoba01', 'hiljuer01', ...] 
            D: a data frame loaded from 'Batting2001AJS.csv', which we processed in problem4.py.
        Output:
            S: an integer scalar, the sum of salaries of all the players in the team.
    '''
    filtered = filtering(D, 'playerID', T)
    S = sum_column(filtered, 'salary')
    return S

 

#--------------------------
def sum_stat(T, D, key='H'):
    '''
        Given a team of players (T), compute the sum of game statistics of all the players in the team.
        For example, suppose we have a team with two players, the number of hits: 100, 200.
        Then the sum of hits in the team will be: 100+200 = 300
        Input:
            T: a python list of playerID's, for example, ['zitoba01', 'hiljuer01', ...] 
            D: a data frame loaded from 'Batting2001AJS.csv', which we processed in problem4.py.
            key: the column to be summed, for example, 'H' represents the number of hits
        Output:
            S: an integer scalar, the sum of statistics of all the players in the team.
    '''
    filtered = filtering(D, 'playerID', T)
    S = sum_column(filtered, key)
    return S



#--------------------------
def runs(T, D):
    '''
        compute the expected runs created by a team based upon Bill James' runs created formula. 
        You need to first compute the total number of hits (H) in the team, and total number of second base (_2B) in the team, etc.
        Then you could use Bill James' runs created formula to compute the expected runs created by the team. 
        Input:
            T: a python list of playerID's, for example, ['zitoba01', 'hiljuer01', ...] 
            D: a data frame loaded from 'Batting2001AJS.csv', which we processed in problem4.py.
        Output:
            RC: the expected runs created/scored by a team, a float scalar.
    '''

    H = sum_stat(T, D, 'H')
    _2B = sum_stat(T, D, '2B')
    _3B = sum_stat(T, D, '3B')
    HR = sum_stat(T, D, 'HR')
    BB = sum_stat(T, D, 'BB')
    AB = sum_stat(T, D, 'AB')
    RC = runs_created(H, _2B, _3B, HR, BB, AB)

    return RC 


#-------------------------------------------------------------------------
# Team Building Methods
#-------------------------------------------------------------------------


#--------------------------
def scout():
    '''
        Hand-pick three players for OAK in 2002 to replace Jason Giambi, Johnny Damon and Jason Isringhausen. 
        Please manually look through the player information in file "Batting2001AJS.csv" and build the OAK team by hand.
        (1) Please note that the overall budget for OAK team is $40,004,167.
        So the sum of salaries in your team should NOT exceed this budget.
        (2) the number of players in the team should be at least 20.
        (3) the expected number of runs created by the team should be at least 700
        (4) the new players chosen cannot be the current members of OAK team in year 2001.
        Output:
            T: a python list of three playerID's, for example, ['zitoba01', 'hiljuer01', 'jonesan01'] 
    '''

    T = ['abreubo01', 'lockhke01', 'towerjo01']
    return T




#--------------------------
def rank_BA(D, min_AB=300,max_salary=1200000):
    '''
        Rank the players based upon Batting Average (BA). 
        The players with the highest BA will be ranked to the top.
        Note, we want to exclude small samples, like 1/1 (H/AB) = 100% (BA). 
        So if the number of AB for a player is smaller than a threshold (min_AB), we will simply set the BA = 0 for that player.
        If a player's salary is higher than the max_salary, we will also set his BA =0, to ignore expensive players.
        Input:
            D: a data frame loaded from 'Batting2001AJS.csv', which we processed in problem4.py.
            min_AB: an integer scalar, the threshold on AB (at-Bat). 
            max_salary: an integer scalar, the maximum salary that we can afford for a player. 
        Output:
            R: a python list of playerID's, for example, ['zitoba01', 'hiljuer01', ...], with descending order of BA scores.
    '''
    D['BA'] = D['H']/D['AB']
    D['BA'] = np.where(D['AB'] < min_AB, 0, D['BA'])
    D['BA'] = np.where(D['salary'] > max_salary, 0, D['BA'])
    D = D.sort_values(by='BA', ascending=False)
    R = list(D['playerID'])
    return R



#--------------------------
def rank_OBP(D,min_AB=300,max_salary=1200000):
    '''
        Rank the players based upon On Base Percentage(OBP)
        The players with the highest OBP will be ranked to the top.
        Note, we want to exclude small samples, like 1/1 (H/AB) = 100% (BA). 
        So if the number of AB is smaller than a threshold (min_AB), we will simply set the OBP = 0
        If a player's salary is higher than the max_salary, we will also set his OBP =0, to ignore expensive players.
        Input:
            D: a data frame loaded from 'Batting2001AJS.csv', which we processed in problem4.py.
            min_AB: an integer scalar, the threshold on AB (at-Bat). 
            max_salary: an integer scalar, the maximum salary that we can afford for a player. 
        Output:
            R: a python list of playerID's, for example, ['zitoba01', 'hiljuer01', ...], with descending order of OBP scores.
    '''
    D['OBP'] = (D['H'] + D['BB'] + D['HBP'])/(D['AB'] + D['BB'] + D['HBP'] + D['SF'])
    D['OBP'] = np.where(D['AB'] < min_AB, 0, D['OBP'])
    D['OBP'] = np.where(D['salary'] > max_salary, 0, D['OBP'])
    D = D.sort_values(by='OBP', ascending=False)
    R = list(D['playerID'])
    return R
