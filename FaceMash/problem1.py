#-------------------------------------------------------------------------
'''
    Problem 1: Elo ranking algorithm 
    In this problem, you will implement the Elo rating algorithm.
'''

#--------------------------
def Terms_and_Conditions():
    return True
 
#--------------------------
def compute_EA(RA, RB):
    '''
        compute the expected probability of player A (with rating RA) to win in a game against a player B (with rating RB).
        Input:
            RA: the rating of player A, a float scalar value
            RB: the rating of player B, a float scalar value
        Output:
            EA: the expected probability of A wins, a float scalar value between 0 and 1.
    '''
    EA = 1/(1 + (10 ** ((RB-RA)/400)))
    return EA



#--------------------------
def update_RA(RA, SA, EA, K = 16.):
    '''
        compute the new rating of player A after playing a game.
        Input:
            RA: the current rating of player A, a float scalar value
            SA: the game result of player A, a float scalar value.
                if A wins in a game, SA = 1.; if A loses, SA =0.
            EA: the expected probability of player A to win in the game, a float scalar between 0 and 1.
             K: k-factor, a constant number which controls how fast to correct the ratings based upon the latest game result.
        Output:
            RA_new: the new rating of player A, a float scalar value
    '''
    RA_new = RA + K * (SA - EA)
    return RA_new


#--------------------------
def elo_rating(W, n_player, K= 16.):
    ''' 
        An implementation of Elo rating algorithm, which was used in FaceMash.
        Given a collection of game results W, compute the Elo rating scores of all the players.
        Input: 
                W: the game results, a numpy matrix of shape (n_game,2), dtype as integers. 
                    n_game is the number of games in the datasets.
                    Each row of W, contains the result of one game: if player i wins player j in the k-th game, W[k][0] = i, W[k][1] = j.
                n_player: the total number of players in the dataset, an integer scalar.
                K: k-factor, a constant number which controls how fast to correct the ratings with the new game results.
        Output: 
                R: the Elo rating scores,  a python array of float values, such as [1000., 200., 500.], of length num_players
    '''

    # initialize the ratings of all players with 400
    R = n_player * [400.]
   
    # for each game, update the ratings based upon the result
    for (A, B) in W:
        print(A)
        # the game result: player A wins, player B loses
        # A is the index of player A, B is the index of player B
        # For example,  A=0, B=2, which means that in this game, the first player (A=0) wins the game against the 3rd player (B=2).

        # extract player A's current rating from R
        RA = R[A]

        # extract player B's current rating from R
        RB = R[B]

        # compute the expected winning probability of player A
        EA = compute_EA(RA, RB)


        # compute the expected winning probability of player B
        EB = compute_EA(RB, RA)


        # update player A's rating based upon the game result (Player A wins)
        RA = update_RA(RA, 1, EA, K)
        R[A] = RA

        # update player B's rating based upon the game result (Player B loses)
        RB = update_RA(RB, 0, EB, K)
        R[B] = RB

    return R
