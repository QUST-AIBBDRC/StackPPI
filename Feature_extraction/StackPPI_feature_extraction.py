import re, os, sys
from collections import Counter
import numpy as np
import platform
import math
import pandas as pd

ALPHABET='ACDEFGHIKLMNPQRSTVWY'
###############################
##read the protein sequences in fasta format
def readFasta(file):
    with open(file) as f:
         records=f.read()
    if re.search('>',records)==None:
       print('error in fasta format')
       sys.exit(1)
    records=records.split('>')[1:]
    myFasta=[]
    for fasta in records:
        array=fasta.split('\n')
        name, sequence=array[0].split()[0], re.sub('[^ARNDCQEGHILKMFPSTWYV-]', '-', ''.join(array[1:]).upper())
        myFasta.append([name,sequence])
    return myFasta
##################################
#read one sequence in pssm format 
def read_pssm_file(file):
    with open(file) as f:
         records=f.read()
    record=records.split('\n')[3:]
    matrix=[]
    for row in record:
        if len(row)==0:
            break
#        name=row[6]
        value=row[10:]
        value_=value.split()
        #value_list=[float(i) for i in value_]
        matrix.append(value_)
    pssm_matrix=np.array(matrix)
    PSSM_matrix=pssm_matrix.astype(float)
    return PSSM_matrix
##########################################
def minSequenceLength(fastas):
	min_length = 10000
	for i in fastas:
		if min_length > len(i[1]):
		   min_length = len(i[1])
	return min_length

def get_CTD_group():
    #the goroups of amino acid sequence
    group1 = {
		'hydrophobicity_PRAM900101': 'RKEDQN',
		'hydrophobicity_ARGP820101': 'QSTNGDE',
		'hydrophobicity_ZIMJ680101': 'QNGSWTDERA',
		'hydrophobicity_PONP930101': 'KPDESNQT',
		'hydrophobicity_CASG920101': 'KDEQPSRNTG',
		'hydrophobicity_ENGD860101': 'RDKENQHYP',
		'hydrophobicity_FASG890101': 'KERSQD',
		'normwaalsvolume': 'GASTPDC',
		'polarity':        'LIFWCMVY',
		'polarizability':  'GASDT',
		'charge':          'KR',
		'secondarystruct': 'EALMQKRH',
		'solventaccess':   'ALFCGIVW'
	   }
    group2 = {
		'hydrophobicity_PRAM900101': 'GASTPHY',
		'hydrophobicity_ARGP820101': 'RAHCKMV',
		'hydrophobicity_ZIMJ680101': 'HMCKV',
		'hydrophobicity_PONP930101': 'GRHA',
		'hydrophobicity_CASG920101': 'AHYMLV',
		'hydrophobicity_ENGD860101': 'SGTAW',
		'hydrophobicity_FASG890101': 'NTPG',
		'normwaalsvolume': 'NVEQIL',
		'polarity':        'PATGS',
		'polarizability':  'CPNVEQIL',
		'charge':          'ANCQGHILMFPSTWYV',
		'secondarystruct': 'VIYCWFT',
		'solventaccess':   'RKQEND'
	    }
    group3 = {
		'hydrophobicity_PRAM900101': 'CLVIMFW',
		'hydrophobicity_ARGP820101': 'LYPFIW',
		'hydrophobicity_ZIMJ680101': 'LPFYI',
		'hydrophobicity_PONP930101': 'YMFWLCVI',
		'hydrophobicity_CASG920101': 'FIWC',
		'hydrophobicity_ENGD860101': 'CVLIMF',
		'hydrophobicity_FASG890101': 'AYHWVMFLIC',
		'normwaalsvolume': 'MHKFRYW',
		'polarity':        'HQRKNED',
		'polarizability':  'KMHFRYW',
		'charge':          'DE',
		'secondarystruct': 'GNPSD',
		'solventaccess':   'MSPTHY'
	   }
    return group1,group2,group3
def Count(seq1, seq2):
    #the counting process of CTD
	amount = 0
	for aa in seq1:
		amount = amount + seq2.count(aa)
	return amount

def CTDC(input_data):
    #This is the composition,transition and distribution 
    #CTDC can can produce 39-dim vector
	fastas=readFasta(input_data)
	group1, group2, group3=get_CTD_group()
	groups = [group1, group2, group3]
	property = (
	'hydrophobicity_PRAM900101', 'hydrophobicity_ARGP820101', 'hydrophobicity_ZIMJ680101', 'hydrophobicity_PONP930101',
	'hydrophobicity_CASG920101', 'hydrophobicity_ENGD860101', 'hydrophobicity_FASG890101', 'normwaalsvolume',
	'polarity', 'polarizability', 'charge', 'secondarystruct', 'solventaccess')

	encodings = []
	header = ['#']
	for p in property:
		for g in range(1, len(groups) + 1):
			header.append(p + '.G' + str(g))
	encodings.append(header)
	for i in fastas:
		name, sequence = i[0], re.sub('X', '', i[1])
		code = [name]
		for p in property:
			c1 = Count(group1[p], sequence) / len(sequence)
			c2 = Count(group2[p], sequence) / len(sequence)
			c3 = 1 - c1 - c2
			code = code + [c1, c2, c3]
		encodings.append(code)
	return encodings

#CTDC_out=CTDC('protein.txt')
#csv_data=pd.DataFrame(data=CTDC_out)
#csv_data.to_csv('CTDC_out.csv',header=False,index=False)

def CTDT(input_data):
    #This is the compostion,transition and distribution
    #CTDD can generate 195-dim vector
	fastas=readFasta(input_data) 
	group1, group2, group3=get_CTD_group()
	property = (
	'hydrophobicity_PRAM900101', 'hydrophobicity_ARGP820101', 'hydrophobicity_ZIMJ680101', 'hydrophobicity_PONP930101',
	'hydrophobicity_CASG920101', 'hydrophobicity_ENGD860101', 'hydrophobicity_FASG890101', 'normwaalsvolume',
	'polarity', 'polarizability', 'charge', 'secondarystruct', 'solventaccess')
	encodings = []
	header = ['#']
	for p in property:
		for tr in ('Tr1221', 'Tr1331', 'Tr2332'):
			header.append(p + '.' + tr)
	encodings.append(header)
	for i in fastas:
		name, sequence = i[0], re.sub('X', '', i[1])
		code = [name]
		aaPair = [sequence[j:j + 2] for j in range(len(sequence) - 1)]
		for p in property:
			c1221, c1331, c2332 = 0, 0, 0
			for pair in aaPair:
				if (pair[0] in group1[p] and pair[1] in group2[p]) or (pair[0] in group2[p] and pair[1] in group1[p]):
					c1221 = c1221 + 1
					continue
				if (pair[0] in group1[p] and pair[1] in group3[p]) or (pair[0] in group3[p] and pair[1] in group1[p]):
					c1331 = c1331 + 1
					continue
				if (pair[0] in group2[p] and pair[1] in group3[p]) or (pair[0] in group3[p] and pair[1] in group2[p]):
					c2332 = c2332 + 1
			code = code + [c1221/len(aaPair), c1331/len(aaPair), c2332/len(aaPair)]
		encodings.append(code)
	return encodings

#CTDT_out=CTDT('protein.txt')
#csv_data=pd.DataFrame(data=CTDT_out)
#csv_data.to_csv('CTDT_out.csv',header=False,index=False)

def Count_D(aaSet, sequence):
    number=0
    code=[]
    select=[]
    for aa in sequence:
        if aa in aaSet:
            number=number+1
    cutoffNums=[1, math.floor(0.25*number),math.floor(0.5*number),math.floor(0.75*number),number]
    myCount=0
    for i in range(len(sequence)):
        if sequence[i] in aaSet:
            myCount=myCount+1
            if myCount in cutoffNums:
                code.append((i+1)/len(sequence))
    if len(code)<5:
        code=[]
        for i in range(len(sequence)):          
            if sequence[i] in aaSet:
               select.append(i)
        if len(select)<1:
            code=[0,0,0,0,0]  
        else:
            if 0 in cutoffNums:
               cutoffNums=np.array(cutoffNums)
               cutoffNums[cutoffNums==0]=1
            for j in range(5):
                label=select[cutoffNums[j]-1]
                code.append((label+1)/len(sequence))
                label=[]
    return code

def CTDD(input_data):
    #This is the composition transition and distribution 
    #CTDD can generate 39-dim vector
	fastas=readFasta(input_data)     
	group1, group2, group3=get_CTD_group()
	property = (
	'hydrophobicity_PRAM900101', 'hydrophobicity_ARGP820101', 'hydrophobicity_ZIMJ680101', 'hydrophobicity_PONP930101',
	'hydrophobicity_CASG920101', 'hydrophobicity_ENGD860101', 'hydrophobicity_FASG890101', 'normwaalsvolume',
	'polarity', 'polarizability', 'charge', 'secondarystruct', 'solventaccess')
	encodings = []
	header = ['#']
	for p in property:
		for g in ('1', '2', '3'):
			for d in ['0', '25', '50', '75', '100']:
				header.append(p + '.' + g + '.residue' + d)
	encodings.append(header)

	for i in fastas:
		name, sequence = i[0], re.sub('X', '', i[1])
		code = [name]
		for p in property:
			code = code + Count_D(group1[p], sequence) + Count_D(group2[p], sequence) + Count_D(group3[p], sequence)
		encodings.append(code)
	return encodings

#CTDD_out=CTDD('protein.txt')
#csv_data=pd.DataFrame(data=CTDD_out)
#csv_data.to_csv('CTDD_out.csv',header=False,index=False)



def NMBroto(input_data, props=['CIDH920105', 'BHAR880101', 'CHAM820101', 'CHAM820102',
										 'CHOC760101', 'BIGC670101', 'CHAM810101', 'DAYM780201'],
				nlag = 30):
    
	fastas=readFasta(input_data)
	if minSequenceLength(fastas) < nlag + 1:
		print('Error: all the sequence length should be larger than the nlag+1: ' + str(nlag + 1) + '\n\n')
		return 0
	fileAAidx = os.path.split(os.path.realpath(__file__))[0] + r'\data\AAidx.txt' if platform.system() == 'Windows' else sys.path[0] + '/data/AAidx.txt'
	with open(fileAAidx) as f:
		records = f.readlines()[1:]
	myDict = {}
	for i in records:
		array = i.rstrip().split('\t')
		myDict[array[0]] = array[1:]
	AAidx = []
	AAidxName = []
	for i in props:
		if i in myDict:
			AAidx.append(myDict[i])
			AAidxName.append(i)
		else:
			print('"' + i + '" properties not exist.')
			return None
	AAidx1 = np.array([float(j) for i in AAidx for j in i])
	AAidx = AAidx1.reshape((len(AAidx),20))
	pstd = np.std(AAidx, axis=1)
	pmean = np.average(AAidx, axis=1)
	for i in range(len(AAidx)):
		for j in range(len(AAidx[i])):
			AAidx[i][j] = (AAidx[i][j] - pmean[i]) / pstd[i]
	index = {}
	for i in range(len(ALPHABET)):
		index[ALPHABET[i]] = i
	encodings = []
	header = ['#']
	for p in props:
		for n in range(1, nlag + 1):
			header.append(p + '.lag' + str(n))
	encodings.append(header)
	for i in fastas:
		name, sequence = i[0], re.sub('-', '', i[1])
		code = [name]
		N = len(sequence)
		for prop in range(len(props)):
			for n in range(1, nlag + 1):
				if len(sequence) > nlag:
					# if key is '-', then the value is 0
					rn = sum([AAidx[prop][index.get(sequence[j], 0)] * AAidx[prop][index.get(sequence[j + n], 0)] for j in range(len(sequence)-n)]) / (N - n)
				else:
					rn = 'NA'
				code.append(rn)
		encodings.append(code)
	return encodings

#NMBroto_out=NMBroto('protein.txt')
#csv_data=pd.DataFrame(data=NMBroto_out)
#csv_data.to_csv('NMBroto_out.csv',header=False,index=False)

def Moran(input_data, props=['CIDH920105', 'BHAR880101', 'CHAM820101', 'CHAM820102',
						 'CHOC760101', 'BIGC670101', 'CHAM810101', 'DAYM780201'],
				nlag = 30):
	fastas=readFasta(input_data)
	if minSequenceLength(fastas) < nlag + 1:		
		print('Error: all the sequence length should be larger than the nlag+1: ' + str(nlag + 1) + '\n\n')
		return 0
	fileAAidx = os.path.split(os.path.realpath(__file__))[0] + r'\data\AAidx.txt' if platform.system() == 'Windows' else sys.path[0] + '/data/AAidx.txt'

	with open(fileAAidx) as f:
		records = f.readlines()[1:]
	myDict = {}
	for i in records:
		array = i.rstrip().split('\t')
		myDict[array[0]] = array[1:]
	AAidx = []
	AAidxName = []
	for i in props:
		if i in myDict:
			AAidx.append(myDict[i])
			AAidxName.append(i)
		else:
			print('"' + i + '" properties not exist.')
			return None
	AAidx1 = np.array([float(j) for i in AAidx for j in i])
	AAidx = AAidx1.reshape((len(AAidx), 20))
	propMean = np.mean(AAidx,axis=1)
	propStd = np.std(AAidx, axis=1)
	for i in range(len(AAidx)):
		for j in range(len(AAidx[i])):
			AAidx[i][j] = (AAidx[i][j] - propMean[i]) / propStd[i]
	index = {}
	for i in range(len(ALPHABET)):
		index[ALPHABET[i]] = i
	encodings = []
	header = ['#']
	for p in props:
		for n in range(1, nlag+1):
			header.append(p + '.lag' + str(n))
	encodings.append(header)
	for i in fastas:
		name, sequence = i[0], re.sub('-', '', i[1])
		code = [name]
		N = len(sequence)
		for prop in range(len(props)):
			xmean = sum([AAidx[prop][index[aa]] for aa in sequence]) / N
			for n in range(1, nlag + 1):
				if len(sequence) > nlag:
					# if key is '-', then the value is 0
					fenzi = sum([(AAidx[prop][index.get(sequence[j], 0)] - xmean) * (AAidx[prop][index.get(sequence[j + n], 0)] - xmean) for j in range(len(sequence) - n)]) / (N - n)
					fenmu = sum([(AAidx[prop][index.get(sequence[j], 0)] - xmean) ** 2 for j in range(len(sequence))]) / N
					rn = fenzi / fenmu
				else:
					rn = 'NA'
				code.append(rn)
		encodings.append(code)
	return encodings

#Moran_out=Moran('protein.txt')
#csv_data=pd.DataFrame(data=Moran_out)
#csv_data.to_csv('Moran_out.csv',header=False,index=False)

def Geary(input_data, props=['CIDH920105', 'BHAR880101', 'CHAM820101', 'CHAM820102',
						 'CHOC760101', 'BIGC670101', 'CHAM810101', 'DAYM780201'],
				nlag = 30):
    fastas=readFasta(input_data)	
    if minSequenceLength(fastas) < nlag + 1:
        print('Error: all the sequence length should be larger than the nlag+1: ' + str(nlag + 1) + '\n\n')
        return 0
    AAidx_file=os.path.split(os.path.realpath(__file__))[0] + r'\data\AAidx.txt' if platform.system() == 'Windows' else sys.path[0] + '/data/AAidx.txt' 
    with open(AAidx_file) as f: 
         records=f.readlines()[1:]
    myDict = {}
    for i in records:
        array=i.rstrip().split('\t')
        myDict[array[0]]=array[1:]
    AAidx = []
    AAidxName = []
    for i in props:
        for i in myDict:
            AAidx.append(myDict[i])
            AAidxName.append(i)
        else:
            print(i + 'properties not exist')
            return None
    AAdix1=np.array([float(j) for i in AAidx for j in i])
    AAidx=AAdix1.reshape(len(AAidx),20)
    propMean = np.mean(AAidx, axis=1)
    propStd=np.std(AAidx,axis=1)
    for i in range(len(AAidx)):
        for j in range(len(AAidx[i])):
            AAidx[i][j]=(AAidx[i][j]-propMean[i])/propStd[i]
    index={}
    for i in range(len(ALPHABET)):
        index[ALPHABET[i]]=i
    encodings=[]
    header=['#']
    for p in props:
        for n in range(1,nlag+1):
            header.append(p+str(n))
    encodings.append(header)
    for i in fastas:
        name,sequence=i[0],re.sub('-','',i[1])
        code=[name]
        N=len(sequence)
        for prop in range(len(props)):
            xmean=sum([AAidx[prop][index[aa]] for aa in sequence])/N
            for n in range(1,nlag+1):
                if len(sequence)>nlag:
                    rn=(N-1)/(2*(N-n)) * ((sum([(AAidx[prop][index.get(sequence[j], 0)] - AAidx[prop][index.get(sequence[j + n], 0)])**2 for j in range(len(sequence)-n)])) / (sum([(AAidx[prop][index.get(sequence[j], 0)] - xmean) ** 2 for j in range(len(sequence))])))
                else:
                    rn='NAN'
                code.append(rn)
        encodings.append(code)
    return encodings

#Geary_out=Geary('protein.txt')
#csv_data=pd.DataFrame(data=Geary_out)
#csv_data.to_csv('Geary_out.csv',header=False,index=False)



def Rvalue(aa1, aa2, AADict, Matrix):
    R_value=sum([(Matrix[i][AADict[aa1]] - Matrix[i][AADict[aa2]]) ** 2 for i in range(len(Matrix))]) / len(Matrix)
    return R_value

def PAAC(input_data, lambdaValue=30, w=0.05):
	fastas=readFasta(input_data)
	if minSequenceLength(fastas) < lambdaValue + 1:
		print('Error: all the sequence length should be larger than the lambdaValue+1: ' + str(lambdaValue + 1) + '\n\n')
		return 0
	dataFile = os.path.split(os.path.realpath(__file__))[0] + r'\data\PAAC.txt' if platform.system() == 'Windows' else re.sub('codes$', '', os.path.split(os.path.realpath(__file__))[0]) + '/data/PAAC.txt'
	with open(dataFile) as f:
		records = f.readlines()
	AA = ''.join(records[0].rstrip().split()[1:])
	AADict = {}
	for i in range(len(AA)):
		AADict[AA[i]] = i
	AAProperty = []
	AAPropertyNames = []
	for i in range(1, len(records)):
		array = records[i].rstrip().split() if records[i].rstrip() != '' else None
		AAProperty.append([float(j) for j in array[1:]])
		AAPropertyNames.append(array[0])
	AAProperty1 = []
	for i in AAProperty:
		meanI = sum(i) / 20
		fenmu = math.sqrt(sum([(j-meanI)**2 for j in i])/20)
		AAProperty1.append([(j-meanI)/fenmu for j in i])
	encodings = []
	header = ['#']
	for aa in AA:
		header.append('PAAC.' + aa)
	for n in range(1, lambdaValue + 1):
		header.append('PAAC.lambda' + str(n))
	encodings.append(header)
	for i in fastas:
		name, sequence = i[0], re.sub('-', '', i[1])
		code = [name]
		theta = []
		for n in range(1, lambdaValue + 1):
			theta.append(
				sum([Rvalue(sequence[j], sequence[j + n], AADict, AAProperty1) for j in range(len(sequence) - n)]) / (
				len(sequence) - n))
		myDict = {}
		for aa in AA:
			myDict[aa] = sequence.count(aa)/100
		code = code + [myDict[aa] / (1 + w * sum(theta)) for aa in AA]
		code = code + [(w * j) / (1 + w * sum(theta)) for j in theta]
		encodings.append(code)
	return encodings

#PAAC_out=PAAC('protein.txt')
#csv_data=pd.DataFrame(data=PAAC_out)
#csv_data.to_csv('PAAC_out.csv',header=False,index=False)


############################################################
#evolutionary information

def exponentPSSM(PSSM):
    PSSM=np.array(PSSM)
    seq_cn=np.shape(PSSM)[0]
    PSSM_exponent=[ [0.0] * 20 ] * seq_cn
    for i in range(seq_cn):
        for j in range(20):
            PSSM_exponent[i][j]=math.exp(PSSM[i][j])
    PSSM_exponent=np.array(PSSM_exponent)
    return  PSSM_exponent

def aac_pssm(input_matrix,exp=True):
    if exp==True:
       input_matrix=exponentPSSM(input_matrix)
    else:
        input_matrix=input_matrix
    seq_cn=float(np.shape(input_matrix)[0])
    aac_pssm_matrix=input_matrix.sum(axis=0)
    aac_pssm_vector=aac_pssm_matrix/seq_cn
    vec=[]
    result=[]
    header=[]
    for f in range(20):
        header.append('aac_pssm.'+str(f))
    result.append(header)   
    for v in aac_pssm_vector:
        vec.append(v)
    result.append(vec)
    return aac_pssm_vector,result

def bi_pssm(input_matrix,exp=True):
    if exp==True:
       input_matrix=exponentPSSM(input_matrix)
    else:
        input_matrix=input_matrix
    PSSM=input_matrix
    PSSM=np.array(PSSM)
    header=[]
    for f in range(400):
        header.append('bi_pssm.'+str(f))
    result=[]
    result.append(header)
    vec=[]
    bipssm=[[0.0]*400]*(PSSM.shape[0]-1)
    p=0
    for i in range(20):
        for j in range(20):
            for h in range(PSSM.shape[0]-1):
                bipssm[h][p]=PSSM[h][i]*PSSM[h+1][j]
            p=p+1
    vector=np.sum(bipssm,axis=0)
    for v in vector:
        vec.append(v)
    result.append(vec)   
    return vector,result


# find the path
Script_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
# choose the method
option = sys.argv[1]
# the input sequence
file = sys.argv[2]

    
if(option == "1"):

    vector=CTDC(file)
    csv_data=pd.DataFrame(data=vector)
    csv_data.to_csv('CTDC_out.csv',header=False,index=False)
    
elif(option == "2"):

    vector=CTDT(file)
    csv_data=pd.DataFrame(data=vector)
    csv_data.to_csv('CTDT_out.csv',header=False,index=False)
    
elif(option == "3"):

    vector=CTDD(file)
    csv_data=pd.DataFrame(data=vector)
    csv_data.to_csv('CTDD_out.csv',header=False,index=False)
    
elif(option == "4"):

    vector=NMBroto(file, nlag = 30)
    csv_data=pd.DataFrame(data=vector)
    csv_data.to_csv('NMBroto_out.csv',header=False,index=False)
    
elif(option == "5"):

    vector=Moran(file,nlag = 30)
    csv_data=pd.DataFrame(data=vector)
    csv_data.to_csv('Moran_out.csv',header=False,index=False)
    
elif(option == "6"):

    vector=Geary(file,nlag = 30)
    csv_data=pd.DataFrame(data=vector)
    csv_data.to_csv('Geary_out.csv',header=False,index=False)
    
    
elif(option == "7"):

    vector=PAAC(file,lambdaValue=30, w=0.05)
    csv_data=pd.DataFrame(data=vector)
    csv_data.to_csv('PAAC_out.csv',header=False,index=False)
    
    
elif(option == "8"):
  
    vector=aac_pssm(file,exp=True)
    csv_data=pd.DataFrame(data=vector)
    csv_data.to_csv('aac_pssm_out.csv',header=False,index=False)
    
    
elif(option == "9"):

    vector=bi_pssm(file,exp=True)
    csv_data=pd.DataFrame(data=vector)
    csv_data.to_csv('bi_pssm_out.csv',header=False,index=False)
    
    
else:
    print("Invalid method number. Please check the method table!")



