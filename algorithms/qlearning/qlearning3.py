import numpy as np
import gym 
import time
 
class Q_Learning:
     
    def __init__(self,env,alpha,gamma,epsilon,n_episodes,n_bins,lower_bound,upper_bound):
        import numpy as np
         
        self.env=env
        self.alpha=alpha
        self.gamma=gamma 
        self.epsilon=epsilon 
        self.n_actions=env.action_space.n 
        self.n_episodes=n_episodes
        self.n_bins=n_bins
        self.lower_bound=lower_bound
        self.upper_bound=upper_bound
        self.total_rewards=[]
        self.Q=np.random.uniform(low=0, high=1, size=(n_bins[0],n_bins[1],n_bins[2],n_bins[3],self.n_actions))

    def discretize_state(self,state):
         
        cart_position=np.linspace(self.lower_bound[0],self.upper_bound[0],self.n_bins[0])
        cart_velocity=np.linspace(self.lower_bound[1],self.upper_bound[1],self.n_bins[1])
        pole_angle=np.linspace(self.lower_bound[2],self.upper_bound[2],self.n_bins[2])
        pole_angle_velocity=np.linspace(self.lower_bound[3],self.upper_bound[3],self.n_bins[3])
         
        position_index=np.maximum(np.digitize(state[0],cart_position)-1,0)
        velocity_index=np.maximum(np.digitize(state[1],cart_velocity)-1,0)
        angle_index=np.maximum(np.digitize(state[2],pole_angle)-1,0)
        angular_velocity_index=np.maximum(np.digitize(state[3],pole_angle_velocity)-1,0)
         
        return tuple([position_index,velocity_index,angle_index,angular_velocity_index])   

    def select_action(self,state,index):
        if index<50:
            return np.random.choice(self.n_actions)   
             
        randomNumber=np.random.random()

        if index>700:
            self.epsilon=0.999*self.epsilon
        if randomNumber < self.epsilon:
            return np.random.choice(self.n_actions)            
        else:
            return np.random.choice(np.where(self.Q[self.discretize_state(state)]==np.max(self.Q[self.discretize_state(state)]))[0])

      
    def train(self):
        for indexEpisode in range(self.n_episodes):
            episodic_reward=[]

            (current_state,_)=self.env.reset()
            current_state=list(current_state)
           
            print("Episode {}==============================================================================================================================".format(indexEpisode))
            terminalState=False
            while not terminalState:
                current_state_index=self.discretize_state(current_state)
                actionA = self.select_action(current_state,indexEpisode)
                (next_state, reward, terminalState,_,_) = self.env.step(actionA)          
                 
                episodic_reward.append(reward)
                 
                next_state=list(next_state)
                 
                next_state_index=self.discretize_state(next_state)
                new_Q=np.max(self.Q[next_state_index])                                               
                                              
                if not terminalState:
                    error=reward+self.gamma*new_Q-self.Q[current_state_index+(actionA,)]
                    self.Q[current_state_index+(actionA,)]=self.Q[current_state_index+(actionA,)]+self.alpha*error
                else:
                    error=reward-self.Q[current_state_index+(actionA,)]
                    self.Q[current_state_index+(actionA,)]=self.Q[current_state_index+(actionA,)]+self.alpha*error
                 
                # set the current state to the next state                    
                current_state=next_state
         
            print("Cumulative Rewards: {}".format(np.sum(episodic_reward)))        
            self.total_rewards.append(np.sum(episodic_reward))
     
    def test(self):
        env=gym.make('CartPole-v1',render_mode='human')
        (current_state,_)=env.reset()
        env.render()
        time_steps=1000
        cumulative_reward=[]
         
        for t in range(time_steps):
            action=np.random.choice(np.where(self.Q[self.discretize_state(current_state)]==np.max(self.Q[self.discretize_state(current_state)]))[0])
            current_state, reward, terminated, _, _ =env.step(action)
            cumulative_reward.append(reward)   
            time.sleep(0.05)
            if (terminated):
                time.sleep(1)
                break
        return cumulative_reward,env      