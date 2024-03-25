import numpy as np
 
class Q_Learning:
     
    def __init__(self,env,alpha,gamma,epsilon,numberEpisodes,numberOfBins,lowerBounds,upperBounds):
        import numpy as np
         
        self.env=env
        self.alpha=alpha
        self.gamma=gamma 
        self.epsilon=epsilon 
        self.actionNumber=env.action_space.n 
        self.numberEpisodes=numberEpisodes
        self.numberOfBins=numberOfBins
        self.lowerBounds=lowerBounds
        self.upperBounds=upperBounds
         
        # this list stores sum of rewards in every learning episode
        self.sumRewardsEpisode=[]
         
        # this matrix is the action value function matrix 
        self.Qmatrix=np.random.uniform(low=0, high=1, size=(numberOfBins[0],numberOfBins[1],numberOfBins[2],numberOfBins[3],self.actionNumber))

    def returnIndexState(self,state):
        position =      state[0]
        velocity =      state[1]
        angle    =      state[2]
        angularVelocity=state[3]
         
        cartPositionBin=np.linspace(self.lowerBounds[0],self.upperBounds[0],self.numberOfBins[0])
        cartVelocityBin=np.linspace(self.lowerBounds[1],self.upperBounds[1],self.numberOfBins[1])
        poleAngleBin=np.linspace(self.lowerBounds[2],self.upperBounds[2],self.numberOfBins[2])
        poleAngleVelocityBin=np.linspace(self.lowerBounds[3],self.upperBounds[3],self.numberOfBins[3])
         
        indexPosition=np.maximum(np.digitize(state[0],cartPositionBin)-1,0)
        indexVelocity=np.maximum(np.digitize(state[1],cartVelocityBin)-1,0)
        indexAngle=np.maximum(np.digitize(state[2],poleAngleBin)-1,0)
        indexAngularVelocity=np.maximum(np.digitize(state[3],poleAngleVelocityBin)-1,0)
         
        return tuple([indexPosition,indexVelocity,indexAngle,indexAngularVelocity])   

    def selectAction(self,state,index):
         
        # first 500 episodes we select completely random actions to have enough exploration
        if index<500:
            return np.random.choice(self.actionNumber)   
             
        # Returns a random real number in the half-open interval [0.0, 1.0)
        # this number is used for the epsilon greedy approach
        randomNumber=np.random.random()
         
        # after 7000 episodes, we slowly start to decrease the epsilon parameter
        if index>7000:
            self.epsilon=0.999*self.epsilon
         
        # if this condition is satisfied, we are exploring, that is, we select random actions
        if randomNumber < self.epsilon:
            # returns a random action selected from: 0,1,...,actionNumber-1
            return np.random.choice(self.actionNumber)            
         
        # otherwise, we are selecting greedy actions
        else:
            return np.random.choice(np.where(self.Qmatrix[self.returnIndexState(state)]==np.max(self.Qmatrix[self.returnIndexState(state)]))[0])

      
    def simulateEpisodes(self):
        import numpy as np
        # here we loop through the episodes
        for indexEpisode in range(self.numberEpisodes):
             
            # list that stores rewards per episode - this is necessary for keeping track of convergence 
            rewardsEpisode=[]
             
            # reset the environment at the beginning of every episode
            (stateS,_)=self.env.reset()
            stateS=list(stateS)
           
            print("Simulating episode {}".format(indexEpisode))
             
             
            # here we step from one state to another
            # this will loop until a terminal state is reached
            terminalState=False
            while not terminalState:
                # return a discretized index of the state
                 
                stateSIndex=self.returnIndexState(stateS)
                 
                # select an action on the basis of the current state, denoted by stateS
                actionA = self.selectAction(stateS,indexEpisode)
                 
                 
                # here we step and return the state, reward, and boolean denoting if the state is a terminal state
                # prime means that it is the next state
                (stateSprime, reward, terminalState,_,_) = self.env.step(actionA)          
                 
                rewardsEpisode.append(reward)
                 
                stateSprime=list(stateSprime)
                 
                stateSprimeIndex=self.returnIndexState(stateSprime)
                 
                # return the max value, we do not need actionAprime...
                QmaxPrime=np.max(self.Qmatrix[stateSprimeIndex])                                               
                                              
                if not terminalState:
                    # stateS+(actionA,) - we use this notation to append the tuples
                    # for example, for stateS=(0,0,0,1) and actionA=(1,0)
                    # we have stateS+(actionA,)=(0,0,0,1,0)
                    error=reward+self.gamma*QmaxPrime-self.Qmatrix[stateSIndex+(actionA,)]
                    self.Qmatrix[stateSIndex+(actionA,)]=self.Qmatrix[stateSIndex+(actionA,)]+self.alpha*error
                else:
                    # in the terminal state, we have Qmatrix[stateSprime,actionAprime]=0 
                    error=reward-self.Qmatrix[stateSIndex+(actionA,)]
                    self.Qmatrix[stateSIndex+(actionA,)]=self.Qmatrix[stateSIndex+(actionA,)]+self.alpha*error
                 
                # set the current state to the next state                    
                stateS=stateSprime
         
            print("Sum of rewards {}".format(np.sum(rewardsEpisode)))        
            self.sumRewardsEpisode.append(np.sum(rewardsEpisode))
     
    # simulate the final learned optimal policy
    def simulateLearnedStrategy(self):
        import gym 
        import time
        env1=gym.make('CartPole-v1',render_mode='human')
        (currentState,_)=env1.reset()
        env1.render()
        timeSteps=1000
        # obtained rewards at every time step
        obtainedRewards=[]
         
        for timeIndex in range(timeSteps):
            print(timeIndex)
            # select greedy actions
            actionInStateS=np.random.choice(np.where(self.Qmatrix[self.returnIndexState(currentState)]==np.max(self.Qmatrix[self.returnIndexState(currentState)]))[0])
            currentState, reward, terminated, truncated, info =env1.step(actionInStateS)
            obtainedRewards.append(reward)   
            time.sleep(0.05)
            if (terminated):
                time.sleep(1)
                break
        return obtainedRewards,env1
   
    def simulateRandomStrategy(self):
        import gym 
        import time
        import numpy as np
        env2=gym.make('CartPole-v1')
        (currentState,_)=env2.reset()
        env2.render()
        # number of simulation episodes
        episodeNumber=100
        # time steps in every episode
        timeSteps=1000
        # sum of rewards in each episode
        sumRewardsEpisodes=[]
         
         
        for episodeIndex in range(episodeNumber):
            rewardsSingleEpisode=[]
            initial_state=env2.reset()
            print(episodeIndex)
            for timeIndex in range(timeSteps):
                random_action=env2.action_space.sample()
                observation, reward, terminated, truncated, info =env2.step(random_action)
                rewardsSingleEpisode.append(reward)
                if (terminated):
                    break     
            sumRewardsEpisodes.append(np.sum(rewardsSingleEpisode))
        return sumRewardsEpisodes,env2          