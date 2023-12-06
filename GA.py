from random import choices
from numpy.random import randint
from numpy.random import rand
import numpy
import json
import time
import pandas as pd

from fcfs import fcfs,initialize_order,sWPT

#initialize all the variables
def initializeVariables():
	global J 
	global M
	global N

	f = open('test2.json')
	data = json.load(f)

	M = 3
	J = int(max(data["P ID"].values())) + 1
	N = 10

	n_iter = 40
	n_bits = J*M
	n_pop = 100
	r_cross = 0.6
	r_mut = 1.0 / float(n_bits)
	crossK1 = 0.7
	crossK2 = 0.4
	mutK1 = 0.8
	mutK2 = 0.4

	return n_iter,n_bits,n_pop,r_cross,r_mut,data,crossK1,crossK2,mutK1,mutK2

#used to print the gene
def printGene(gene):
	print("| ", end="")
	for i in range(M):
		for j in range(J):
			if gene[(i*J)+j] >= J:
				print("*", end= " ")
			else:
				print(gene[(i*J)+j],end=" ")
		print("| ", end="")
	pass

#convert the gene to a more readable string format
def convertGene(gene):
	temp = "| "
	for i in range(M):
		for j in range(J):
			if gene[(i*J)+j] >= J:
				temp += "* "
			else:
				temp += str(gene[(i*J)+j]) + " "
		temp += "| "
	return temp
	
# used to normalize the score to use in selection
def normalizeScore(scores):
	total = 0
	for i in scores:
		total += i
	for i in range(len(scores)):
		scores[i] = scores[i]/total
	return scores

#use the dataset to find the time required for each chromosome
def timeObjective(x,data):
	sum = [0]*M
	for j in range(M):
		for i in range(J):
			sum[j] += data["Time"][str(x[i+(j*J)])] if x[i+(j*J)] < J else 0
	totalTime = max(sum)
	OptimalTime = 1000
	return OptimalTime/totalTime

#objective with priority 
def priorityObjective(x,data):
	sum = [0]*M
	pSum = 0
	for m in range(M):
		for i in range(J):
			pSum += sum[m] * data["Weight"][str(x[i+(m*J)])] if x[i+(m*J)] < J else 0
			sum[m] += data["Time"][str(x[i+(m*J)])] if x[i+(m*J)] < J else 0

	totalTime = max(sum)
	OptimalTime = 100000000
	
	return OptimalTime/( totalTime * pSum)
 
# tournament selection
def tournamentSelection(pop, scores, k=3):
	# first random selection
	scores = normalizeScore(scores)
	temp =  choices(pop,scores)
	#print(temp[0])
	return temp[0]
 
# crossover two parents to create two children
def pmxCrossover(p1,p2,prob):
    c1 , c2 = p1.copy() , p2.copy()
    p3, p4 = p1.copy() , p2.copy()
    lenP = len(p1)#length of each chromosome
    if(rand() < prob):
        pt1 = randint(1, lenP-4)
        pt2 = 0
        while True:
          pt2 = randint(1, lenP-2)
          if pt2 > pt1:
            break
        map1 = {}
        map2 = {}
        lst1 = []
        lst2 = []

        for i in reversed(range(pt1, pt2 +1)):
            lst1.append(p1[i])
            lst2.append(p2[i])
 
            c1[i] = p2[i]
            c2[i] = p1[i]
    
        for i in reversed(range(lenP)):
            map1[p1[i]] = p2[i]
            map2[p2[i]] = p1[i]

        for i in reversed(range(pt1)):
            while( c1[i] in lst2) :
                c1[i] = map2[c1[i]]
            while( c2[i] in lst1) :
                c2[i] = map1[c2[i]]
        
        for i in reversed(range(pt2+1 , lenP)):
            while( c1[i] in lst2) :
                c1[i] = map2[c1[i]]
            while( c2[i] in lst1) :
                c2[i] = map1[c2[i]]
               
        # printGene(p1)
        # printGene(p2)
        # print(pt1, pt2)
        # printGene(c1)
        # printGene(c2)
        # print()
	
        # printGene(correctGene(c1))
        # printGene(correctGene(c2))
        # print(c1)
        # print(c2)

        # printGene(c1)
        # printGene(c2)
        # printGene(correctGene(c1))
        # printGene(correctGene(c2))
    c1Test = correctGene(c1)
    c2Test = correctGene(c2)
    if c1Test or c2Test:
        return [p3,p4]
    else:
        return [correctGene(c1),correctGene(c2)]

def correctGene(gene):
	#correct the gene description
	for m in range(M):
		lastR = 0
		FirstF = -1
		k = 0
		while True:
			try:
				if gene[m*J + k] >= J and k < J:
					FirstF = m*J + k
					break
				else:
					k+=1
			except:

				# print("issue has been faced")
				# printGene(gene)
				# print("J = {}, M = {}, k = {}".format(J,m,k))
				return True	
						
		#print(FirstF)
		for i in range(m*J,(m+1)*J):
			if gene[i] < J:
				lastR = i
			if lastR > FirstF:
				gene[lastR],gene[FirstF] = gene[FirstF],gene[lastR]
				lastR = FirstF
				FirstF += 1
	return gene
 
# mutation operator swap
def swapMutation(bitstring, r_mut):
	s1 = 0
	s2 = 0 
	if rand() < r_mut:
		while True:
			s1 = randint(0,len(bitstring))
			if(int(bitstring[s1]) < J):
				break
		while True:
			s2 = randint(0,len(bitstring))
			if(int(bitstring[s2]) < J and s2 != s1):
				break

	#print(bitstring,s1,s2)
	temp = bitstring[s1]
	bitstring[s1] = bitstring[s2]
	bitstring[s2] = temp
	#print(bitstring)

def newMutation(c1,prob):
	lenP = len(c1) #length of each chromosome
	if(rand() < prob):
		# printGene(c1)
		# print()
	
		pt1 = randint(M)
		pt2 = randint(M)
		while(pt2 == pt1):
			pt2 = randint(M)
		
		k = randint(J)
		while c1[pt1*J + k] >= J or c1[pt2*J + k] >= J: 
			if k == 0:
				break
			k = randint(0,k)

		# print(pt1,pt2, k)
		while k<J:
			c1[pt1*J + k],c1[pt2*J + k] = c1[pt2*J + k], c1[pt1*J + k]
			k+=1
		# printGene(c1)

#initialize the popuation
def populationInitRandom(data): #include the list scheduling here
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

def populationInitHeuristic(data):
	#initialize fcfs N times and take the best one
	global N # number of iterations
	outputTime=[0]*N #finalTime of each run to find the minimum
	order1 = [0]*N #final order of each run
	for i in range(N):
		order = initialize_order(data)
		outputTime[i],order1[i] = fcfs(data,order,M)
	
	arrayTemp = numpy.array(outputTime)
	#print("totat Time : ",min(outputTime), order[arrayTemp.argmin()])
	#return list(int(x) for x in order1[arrayTemp.argmin()]) #used to convert to a integer list
	#print(order1[arrayTemp.argmin()])
	return order1[arrayTemp.argmin()]

def hammingDistance(g1,g2):
	# print(g1)
	# print(g2)
	total = 0
	for i in range(len(g1)):
		if g1[i] != g2[i] and (g1[i] < J or g2[i] < J):
			total += 1
	return total

def move(g1,g2,prob):
	for i in range(len(g1)):
		if rand() < prob:
			#find value at g2 then swap g1[i] with g1[g1.index(g2[i])]
			g1[i], g1[g1.index(g2[i])] =  g1[g1.index(g2[i])], g1[i]

def Firefly(populationInit,objective,n_pop,n_iter,data):
	prevbest = [0]*5
	pop = [populationInit(data) for _ in range(n_pop)]
	best, best_eval = 0, objective(pop[0],data)
	absorptionCoefficient = 0.01 #find the real value
	aStep = 1 #change the value
	for gen in range(n_iter):
		for xi in range(len(pop)):
			for xj in range(len(pop)):
				if(objective(pop[xi],data) < objective(pop[xj],data)):
					distance = hammingDistance(pop[xi],pop[xj])
					attraction = 1/(1+ absorptionCoefficient*distance*distance)
					#move xi to xj
					move(pop[xi],pop[xj],attraction)
					#move 2
					swapMutation(pop[xi],aStep)
				else:
					#random movement
					swapMutation(pop[xi],aStep)

		#dynamic stop
		prevbest.pop(0)
		prevbest.append(best_eval)
		stop = True
		for i in range(4):
			if prevbest[i] != prevbest[i+1]:
				stop = False
		if stop and gen > 10:
			break

		lightValue = [objective(c,data) for c in pop]
		for i in range(len(pop)):
			if lightValue[i] > best_eval:
					best, best_eval = pop[i], lightValue[i]

		#print(best_eval)

	return [best,best_eval,gen]

def FastFirefly(populationInit,objective,n_pop,n_iter,data):
	prevbest = [0]*5
	pop = [populationInit(data) for _ in range(n_pop)]
	best, best_eval = pop[0], objective(pop[0],data)
	absorptionCoefficient = 0.01 #find the real value
	aStep = 0.1 #change the value
	for gen in range(n_iter):
		for xi in range(len(pop)):
			for xj in range(xi):
				if(objective(pop[xi],data) < objective(best,data)):
					distance = hammingDistance(pop[xi],best)
					attraction = 1/(1+ absorptionCoefficient*distance*distance)
					#move xi to xj
					move(pop[xi],best,attraction)
					#move 2
					swapMutation(pop[xi],aStep)
				else:
					#random movement
					swapMutation(pop[xi],aStep)

		#dynamic stop
		prevbest.pop(0)
		prevbest.append(best_eval)
		stop = True
		for i in range(4):
			if prevbest[i] != prevbest[i+1]:
				stop = False
		if stop and gen > 10:
			break

		lightValue = [objective(c,data) for c in pop]
		for i in range(len(pop)):
			if lightValue[i] > best_eval:
					best, best_eval = pop[i], lightValue[i]

		#print(best_eval)

	return [best,best_eval,gen]

# genetic algorithm
def GANonAdaptive(populationInit,mutation,objective, n_bits, n_iter, n_pop, r_cross, r_mut,data):
	# initial population of random bitstring
	pop = [populationInit(data) for _ in range(n_pop)]
	# keep track of best solution
	best, best_eval = pop[0], objective(pop[0],data)
	# enumerate generations
	for gen in range(n_iter):
		newBFlag = 0
		# evaluate all candidates in the population
		scores = [objective(c,data) for c in pop]
		# check for new best solution
		for i in range(n_pop):
			if scores[i] > best_eval:
				best, best_eval = pop[i], scores[i]
				print(">%d, new best f(%s) = %.3f" % (gen,  convertGene(pop[i]), scores[i]))
				newBFlag = 1
		# select parents
		selected = [tournamentSelection(pop, scores) for _ in range(n_pop)]
		# create the next generation
		children = list()
		for i in range(0, n_pop, 2):
			# get selected parents in pairs
			p1, p2 = selected[i], selected[i+1]
			# crossover and mutation
			for c in pmxCrossover(p1, p2, r_cross):
				# mutation
				mutation(c, r_mut)
				# store for next generation
				children.append(c)
		# replace population
		pop = children
		if newBFlag == 0:
			print(">%d, old best f(%s) = %.3f" % (gen,  convertGene(best), best_eval))
	return [best, best_eval,gen]

# genetic algorithm
def GAAdaptive(populationInit,mutation,objective, n_bits, n_iter, n_pop, data,crossK1, crossK2, mutK1, mutK2):
	#copy the crossover and mutation values
	#r_cross = r_cross_1.copy()
	#r_mut = r_mut_1.copy()
	# initial population of random bitstring
	prevbest= [0]*5
	pop = [populationInit(data) for _ in range(n_pop)]
	# keep track of best solution
	best, best_eval = pop[0], objective(pop[0],data)
	# enumerate generations
	for gen in range(n_iter):
		newBFlag = 0
		# evaluate all candidates in the population
		scores = [objective(c,data) for c in pop]
		# check for new best solution
		for i in range(n_pop):
			if scores[i] > best_eval:
				best, best_eval = pop[i], scores[i]
				# print(">%d, new best f(%s) = %.6f" % (gen,  convertGene(pop[i]), scores[i]))
				newBFlag = 1
		# select parents
		selected = [tournamentSelection(pop, scores) for _ in range(n_pop)]
		# create the next generation
		children = list()
		for i in range(0, n_pop, 2):
			# get selected parents in pairs
			p1, p2 = selected[i], selected[i+1]
			# crossover and mutation

			#change the crossover value adaptively
			fa = objective(p1,data)
			fb = objective(p2,data)
			r_cross = crossK1 *(best_eval - fa)/(best_eval - fb) if fa >= fb and fb!= best_eval else crossK2
			fd = sum(scores)/len(scores) #average fitness

			for c in pmxCrossover(p1, p2, r_cross):
				#change the mutation value adaptively
				fc = objective(c,data)
				r_mut = mutK1 *(best_eval - fc)/(best_eval - fd) if fc >= fd and fd!= best_eval else mutK2
				# mutation
				mutation(c, r_mut)
				# store for next generation
				children.append(c)
		# replace population
		pop = children
		# if newBFlag == 0:
		# 	print(">%d, old best f(%s) = %.6f" % (gen,  convertGene(best), best_eval))

		#dynamic stop
		prevbest.pop(0)
		prevbest.append(best)
		stop = True
		for i in range(4):
			if prevbest[i] != prevbest[i+1]:
				stop = False
		if stop and gen > 10:
			break

	return [best, best_eval, gen]



def ga(populationInit, mutation, objective, path):
	# define the micro values
	n_iter, n_bits, n_pop, r_cross, r_mut, 	data, crossK1, crossK2, mutK1, mutK2 = initializeVariables()

	# prevTime = time.time()
	# #best, score = GANonAdaptive(populationRandom,swapMutation,timeObjective, n_bits, n_iter, n_pop, r_cross, r_mut,data)
	# best, score, gen = GAAdaptive(populationInitHeuristic,newMutation,timeObjective, n_bits, n_iter, n_pop, data, crossK1, crossK2, mutK1, mutK2)
	# curTime = time.time()
	# print("time ",curTime-prevTime)
	# print('Done!')
	# print('f(%s) = %f' % (convertGene(best), score))

	array = []
	df = {}

	for iteration in range(100):
		prevTime = time.time()
		# best, score, gen = GANonAdaptive(populationInit,mutation,objective, n_bits, n_iter, n_pop, r_cross, r_mut,data)
		best, score, gen = GAAdaptive(populationInit,mutation,objective, n_bits, n_iter, n_pop, data, crossK1, crossK2, mutK1, mutK2)
		curTime = time.time()
		array.append({"Time":curTime - prevTime,"Score":score,"gen":gen})
	
	df = pd.DataFrame.from_dict(array)
	df.to_json(path_or_buf=path)

def firefly(populationInit, objective, path):
	n_iter, n_bits, n_pop, r_cross, r_mut, 	data, crossK1, crossK2, mutK1, mutK2 = initializeVariables()

	#print(hammingDistance(populationInitRandom(data),populationInitRandom(data)))
	# prevTimeFirefly = time.time()
	# Firefly(populationInitRandom,timeObjective,n_pop,n_iter,data)
	# currTimeFirefly = time.time()
	# print(currTimeFirefly-prevTimeFirefly)

	array = []
	df = {}

	for iteration in range(100):
		prevTime = time.time()
		#best, score = GANonAdaptive(populationRandom,swapMutation,timeObjective, n_bits, n_iter, n_pop, r_cross, r_mut,data)
		best, score, gen = FastFirefly(populationInitRandom,objective,n_pop,n_iter,data)
		curTime = time.time()
		array.append({"Time":curTime - prevTime,"Score":score,"gen":gen})
		print("gen : ",iteration)
	
	df = pd.DataFrame.from_dict(array)
	df.to_json(path_or_buf=path)

def commentStuff():
	#swapMutation(populationInitRandom(data),1.0)
	#convertGene(populationInitRandom())
	#pmxCrossover(populationInitRandom(data),populationInitRandom(data),1.0)

	# g1 = populationInitRandom(data)
	# g2 = populationInitRandom(data)
	# print(g1)
	# print(g2)
	# move(g1,g2,1)
	# print(g1)

	#print(hammingDistance(populationInitRandom(data),populationInitRandom(data)))
	# prevTimeFirefly = time.time()
	# Firefly(populationInitRandom,timeObjective,n_pop,n_iter)
	# currTimeFirefly = time.time()
	# print(currTimeFirefly-prevTimeFirefly)
	pass

if __name__ == "__main__":
	# ga(populationInitRandom,newMutation,timeObjective,"D:\Projects\MAJOR PROJECT\output\ga_40_r_t.json")
	# ga(populationInitHeuristic,newMutation,timeObjective,"D:\Projects\MAJOR PROJECT\output\ga_40_h_t.json")

	# ga(populationInitRandom,newMutation,priorityObjective,"D:\Projects\MAJOR PROJECT\output\ga_40_r_p.json")
	# ga(populationInitHeuristic,newMutation,priorityObjective,"D:\Projects\MAJOR PROJECT\output\ga_40_h_p.json")

	# firefly(populationInitRandom,timeObjective,"D:\Projects\MAJOR PROJECT\output\\ff_40_r_t.json")
	# firefly(populationInitRandom,priorityObjective,"D:\Projects\MAJOR PROJECT\output\\ff_40_r_p.json")


	n_iter, n_bits, n_pop, r_cross, r_mut, 	data, crossK1, crossK2, mutK1, mutK2 = initializeVariables()

	prevTimeFirefly = time.time()
	best, score, gen = FastFirefly(populationInitRandom,timeObjective,n_pop,n_iter,data)
	currTimeFirefly = time.time()
	print("Firefly ")
	print("time ",currTimeFirefly-prevTimeFirefly)
	print('Done!')
	print('f(%s) = %f' % (convertGene(best), score))

	print("\nhybrid GA")
	prevTime = time.time()
	best, score, gen = GAAdaptive(populationInitHeuristic,newMutation,timeObjective, n_bits, n_iter, n_pop, data, crossK1, crossK2, mutK1, mutK2)
	curTime = time.time()
	print("time ",curTime-prevTime)
	print('Done!')
	print('f(%s) = %f' % (convertGene(best), score))