# genetic algorithm search of the one max optimization problem
from secrets import choice
from numpy.random import randint
from numpy.random import rand
import json
 
M = 3
J = 4
	
# objective function
def normalizeScore(scores):
	total = 0
	for i in scores:
		total += i
	for i in range(len(scores)):
		scores[i] = scores[i]/total
	return scores

def objective(x):
	f = open('test.json')
	data = json.load(f)
	sum = [0]*M
	for i in range(J):
		for j in range(M):
			sum[j] += data["Time"][x[i+(j*J)]] if x[i+(j*J)] > -1 else 0
	totalTime = max(sum)
	OptimalTime = 1
	return OptimalTime/totalTime
 
# tournament selection
def selection(pop, scores, k=3):
	# first random selection
	scores = normalizeScore(scores)
	return choice(pop,scores)
 
# crossover two parents to create two children
def crossover(p1,p2,prob):
    c1 , c2 = p1.copy() , p2.copy()

    if(rand() < prob):
        pt1 = randint(1, len(p1)-4)
        pt2 = 0
        while True:
          pt2 = randint(1, len(p1)-2)
          if pt2 > pt1:
            break
        map1 = {}
        map2 = {}
        lst1 = []
        lst2 = []

        for i in range(pt1, pt2 +1):
            lst1.append(p1[i])
            lst2.append(p2[i])
 
            c1[i] = p2[i]
            c2[i] = p1[i]
    
        for i in range(len(p1)):
            map1[p1[i]] = p2[i]
            map2[p2[i]] = p1[i]

        for i in range(pt1):
            while( c1[i] in lst2) :
                c1[i] = map2[c1[i]]
            while( c2[i] in lst1) :
                c2[i] = map1[c2[i]]
        
        for i in range(pt2+1 , len(p1)):
            while( c1[i] in lst2) :
                c1[i] = map2[c1[i]]
            while( c2[i] in lst1) :
                c2[i] = map1[c2[i]]
        
        print(p1, p2, pt1, pt2)
        print(c1, c2)

    return [c1,c2]
 
# mutation operator
def mutation(bitstring, r_mut):
	if rand() < r_mut:
		s1 = -1
		s2 = -1
		while True:
			s1 = randint(0,len(bitstring))
			if(int(bitstring[s1]) < J):
				break
		while True:
			s2 = randint(0,len(bitstring))
			if(int(bitstring[s2]) < J and s2 != s1):
				break

	print(bitstring,s1,s2)
	temp = bitstring[s1]
	bitstring[s1] = bitstring[s2]
	bitstring[s2] = temp
	print(bitstring)
	pass
 
def population_init(): #include the list scheduling here
	f = open('test.json')
	data = json.load(f)
	"""
	for i in data:
		print(data[i])
	"""
	n = -1
	m = list()
	for i in range(M):
		temp = []
		for j in range(J):
			temp.append(n)
			#n+=1
		m.append(temp)
	#m = [[-1]*J for i in range(M)]
	mi = [0] * M
	for id in data["P ID"]:
		i = randint(0,M)
		m[i][mi[i]] = int(id)
		mi[i]+=1
	final = list()
	for i in range(M):
		final.extend((m[i]))
	t2 = J
	for i in range(len(final)):
		if final[i] == -1:
			final[i] = t2
			t2 += 1
	return(final)

# genetic algorithm
def genetic_algorithm(objective, n_bits, n_iter, n_pop, r_cross, r_mut):
	# initial population of random bitstring
	pop = [randint(0, 2, n_bits).tolist() for _ in range(n_pop)]
	# keep track of best solution
	best, best_eval = 0, objective(pop[0])
	# enumerate generations
	for gen in range(n_iter):
		# evaluate all candidates in the population
		scores = [objective(c) for c in pop]
		# check for new best solution
		for i in range(n_pop):
			if scores[i] < best_eval:
				best, best_eval = pop[i], scores[i]
				print(">%d, new best f(%s) = %.3f" % (gen,  pop[i], scores[i]))
		# select parents
		selected = [selection(pop, scores) for _ in range(n_pop)]
		# create the next generation
		children = list()
		for i in range(0, n_pop, 2):
			# get selected parents in pairs
			p1, p2 = selected[i], selected[i+1]
			# crossover and mutation
			for c in crossover(p1, p2, r_cross):
				# mutation
				mutation(c, r_mut)
				# store for next generation
				children.append(c)
		# replace population
		pop = children
	return [best, best_eval]
 
# define the total iterations
n_iter = 100
# bits
n_bits = 20
# define the population size
n_pop = 100
# crossover rate
r_cross = 0.9
# mutation rate
r_mut = 1.0 / float(n_bits)
# perform the genetic algorithm search
#mutation(population_init(),1.0)
#print(population_init())
crossover(population_init(),population_init(),1.0)
"""
best, score = genetic_algorithm(objective, n_bits, n_iter, n_pop, r_cross, r_mut)
print('Done!')
print('f(%s) = %f' % (best, score))
"""
