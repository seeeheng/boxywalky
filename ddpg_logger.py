import pandas as pd
import time

class DDPGLogger:
	def __init__(self, start_time):
		self.time = []
		self.episodes = []
		self.scores = []
		self.n_log = 0
		self.start_time = start_time
		self.update_time = start_time

	def update_time_and_episodes(self,current_time,n_episode):
		current_time = time.time()
		if current_time - self.start_time > 60*15:
			self.update_time = current_time
		self.time += [current_time]
		self.episodes += [n_episode]

	def update_scores(self,score):
		self.scores += [score]

	def check(self, n_episode):
		current_time = time.time()
		if current_time - self.start_time > 5:
			self.update_time_and_episodes(current_time, n_episode)
			self.make_csv()
			self.start_time = current_time

	def make_csv(self):
		self.n_log += 1
		pd.DataFrame(data=zip(self.time,self.episodes),columns=("time","episodes")).to_csv("benchmark/log{}.csv".format(self.n_log),index=False)
		pd.Series(data=self.scores).to_csv("benchmark/scores.csv",index=True)
