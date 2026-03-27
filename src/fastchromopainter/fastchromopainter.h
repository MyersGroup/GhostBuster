////Copyright Simon Myers 2016. All rights reserved

// stdafx.h : include file for standard system include files,
// or project specific include files that are used frequently, but
// are changed infrequently
//

//#pragma once
/*Debugging*/
# define DEBUG 3
# define DEBUG2 0
# define DEBUG3 0
# define DEBUG4 0
# define DEBUG5 0
/*Displaying*/
# define DISPLAY 1
# define DISPLAY2 0
# define DISPLAY3 0
# define DISPLAY4 0
# define DISPLAY5 0

//#include "targetver.h"

#include <stdio.h>
//#include <tchar.h>



// TODO: reference additional headers your program requires here
#include <math.h>
#include <time.h>
# include <stdlib.h>
# include <iostream>
//# include <iomanip.h>
# include <fstream>
# include <string>
# include <vector>
# include <time.h>
# include <algorithm>
# include <iomanip>
#include <sstream>
#include <bitset>

// #define PROBS_COMPACT 1

#ifdef PROBS_COMPACT
typedef float probs_type;
float sum_tol = 1e-30;
float sum_inv_tol = 1e30;
#endif

#ifndef PROBS_COMPACT
typedef double probs_type;
double sum_tol = 1e-40;
double sum_inv_tol = 1e40;
#endif 

using namespace std;
void *c_malloc(long bytes);
void binary_read(std::ifstream& fin, std::vector<bool>& x);
void binary_write(std::ofstream& fout, const std::vector<bool>& x);

clock_t total_time, forward_time;

/////Class to hold data of different types
////class to hold data of haplotype or genotype form
class dataclass
{
public:
	///where is data
	char *filename;
	char *labelsfilename;
	char *identifierfilename;
	///positions
	vector <long> sites;
	///number of sites
	long n_sites;
	/////may not need below
	long region_lend, region_rend;
	///Below - the number of haplotypes or lines
	int n_sequences;
	int binaryoutmode;
	int orderout;
	//Recombination probabilities between pairs of successive sites (already scaled by rho)
	vector<probs_type> rates;
	//Weights on each SNP in the genetic distance scale (cM if rates file is cM)
	vector<probs_type> weights;

	//Data itself
	vector< vector <char> > list;

	///Population labels if this is panel
	vector<string> labels;
	vector<string> indnames;
	vector<string> popnames;
	int totalinds;
	////mapping between the labels and the data if these are used as sources
	////in the vector includeinds, entries of 0 mean ignore this individual
	vector<int> identifiers;
	vector<int> includeinds;
	vector<string> checkset;
	//Function to get the list, not called automatically. Sets n_sites and n_seq
	void get_sequences();
	void get_panelinfo();
	void getstateprobs(dataclass & inddata, vector<probs_type> & sumprobs,probs_type theta, probs_type rho, const char * filename_out);
	void output_binary(ofstream & binaryfile, ofstream & panelfile);
	dataclass();
	///seed with existing compact data

	virtual ~dataclass();
	
private:

};

class rate
{
public:
	int npositions;
	vector<long> positions;
	vector<probs_type> rates;
	char *rates_file;
	rate();
	virtual ~rate();
	int get_rates();
	int scale_rates(probs_type rho, dataclass & ourdata);
	
private:
};

