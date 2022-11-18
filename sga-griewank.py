import csv
import math
import time

import numpy as np
import pandas as pd
from numpy.random import randint
from numpy.random import rand

def objective(chromosome):
	part1 = 0
	part2 = 1
	for i in range(len(chromosome)):
		part1 += chromosome[i] ** 2
		part2 *= math.cos(float(chromosome[i]) / math.sqrt(i + 1))
	return 1 + (float(part1) / 4000.0) - float(part2)

def decode(bounds, n_bits, bitstring):
	decoded = list()
	largest = 2**n_bits
	for i in range(len(bounds)):
		start, end = i * n_bits, (i * n_bits)+n_bits
		substring = bitstring[start:end]
		chars = ''.join([str(s) for s in substring])
		integer = int(chars, 2)
		value = bounds[i][0] + (integer/largest) * (bounds[i][1] - bounds[i][0])
		decoded.append(value)
	return decoded

def calc_probability_rws(population, costs, beta=1):
  costs = np.array(costs)
  avg_cost = np.mean(costs)  
  if avg_cost != 0:
    costs = costs / avg_cost
  return np.exp(-beta * costs)

def roulette_wheel_selection(population, normalized_scores):
	c = np.cumsum(normalized_scores)
	r = sum(normalized_scores) * np.random.rand()
	ind = np.argwhere(r <= c)
	choiced_index = ind[0][0]
	return population[choiced_index]

def tournament_selection(pop, scores, k=3):
	selection_ix = randint(len(pop)) 
	for ix in randint(0, len(pop), k-1):
		if scores[ix] < scores[selection_ix]:
			selection_ix = ix
	return pop[selection_ix]

def crossover(p1, p2, r_cross):
	c1, c2 = p1.copy(), p2.copy()
	if rand() < r_cross:
		pt = randint(1, len(p1)-2)
		c1 = p1[:pt] + p2[pt:]
		c2 = p2[:pt] + p1[pt:]
	return [c1, c2]

def mutation(bitstring, r_mut):
	for i in range(len(bitstring)):
		if rand() < r_mut:
			bitstring[i] = 1 - bitstring[i]

def generate_initial_population(bounds, n_bits, n_pop):
	return [randint(0, 2, n_bits*len(bounds)).tolist() for _ in range(n_pop)]

def genetic_algorithm(objective, bounds, n_bits, n_iter, n_pop, r_cross, r_mut):
	pop = generate_initial_population(bounds, n_bits, n_pop) 
	pop_length = len(pop)
	nfob = 0
	best, best_eval = 0, objective(decode(bounds, n_bits, pop[0])) 
	dados =[]
	for gen in range(n_iter):
		decoded = [decode(bounds, n_bits, p) for p in pop] 
		scores = [objective(d) for d in decoded] 
		nfob += pop_length
		for i in range(n_pop): 
			if scores[i] < best_eval:
				best, best_eval = pop[i], scores[i]
				dados_parciais = []
				dados_parciais.append(nfob)
				dados_parciais.append(best_eval)
				dados_parciais.append(scores)
				print("[->] NFOB: %d [+] New best: %f " % (nfob, scores[i]))
				dados.append(dados_parciais)
		probs = calc_probability_rws(pop, scores)
		selected = [roulette_wheel_selection(pop, probs) for _ in range(n_pop)] # select parents
		children = list() # create the next generation
		for i in range(0, n_pop, 2):
			p1, p2 = selected[i], selected[i+1] # get selected parents in pairs
			for c in crossover(p1, p2, r_cross):
				mutation(c, r_mut) # mutation
				children.append(c) # store for next generation
		pop = children # replace population
	df = pd.DataFrame(dados)
	df.to_excel("dados-nfob.xlsx", sheet_name="dados-nfob")
	return [best, best_eval]

bounds = [[-600.0, 600.0],[-600.0, 600.0],[-600.0, 600.0],[-600.0, 600.0],[-600.0, 600.0],
		  [-600.0, 600.0],[-600.0, 600.0],[-600.0, 600.0],[-600.0, 600.0],[-600.0, 600.0]] 

# default parameters
n_iter = 1000 
n_bits = 140
n_pop = 100 
r_cross = 0.9 
t_times = list()
r_mut = 1.0 / (float(n_bits) * len(bounds)) # mutation rate

#scenario 1
execution50_n_pop = list()
r_cross = 0.6
r_mut = 0.01
for i in range(50):
    execution = list()
    for i in [10,20,40,80,120,160]:
        n_pop = i
        best, score = genetic_algorithm(objective, bounds, n_bits, n_iter, n_pop, r_cross,r_mut) 
        print("[%d][n_pop] Done" %(i))
        execution.append(score)
        print("Execution finished")
    execution50_n_pop.append(execution)

# Scenario 2
execution50_p_cross = list()
n_pop = 160
r_mut = 0.01
for j in range(50):
	execution = list()
	for i in [0.1, 0.2, 0.4, 0.6, 0.8, 1]:
		r_cross = i
		best, score = genetic_algorithm(objective, bounds, n_bits, n_iter, n_pop, r_cross,r_mut)  
		execution.append(score)
	execution50_p_cross.append(execution)
df = pd.DataFrame(execution50_p_cross)
df.to_excel('p_cross.xlsx', sheet_name='p_cross')

# Scenario 3
execution50_r_mut = list()
n_pop = 160 
r_cross = 1
r_mut = 0.005
r_cross = 1 
for j in range(50):
	execution = list()
	for i in [0.005, 0.01, 0.05, 0.1, 0.3, 0.5]:
		r_mut = i
		start = time.time()
		best, score = genetic_algorithm(objective, bounds, n_bits, n_iter, n_pop, r_cross,r_mut)  
		execution.append(score)
	execution50_r_mut.append(execution)
df = pd.DataFrame(execution50_r_mut)
df.to_excel('p_mut.xlsx', sheet_name='p_mut')

# best, score = genetic_algorithm(objective, bounds, n_bits, n_iter, n_pop, r_cross, r_mut) # perform the genetic algorithm search
# print('Done!')
# decoded = decode(bounds, n_bits, best)
# print('%.2f -> f(%s) = ' % (score,decoded))

