////Copyright Simon Myers 2016. All rights reserved

//. / fastchromopainter.exe chr22_imputed.phase chr22test.binphase chr22test.ids externalsamples.donor chr22.recomb testnew.out 0.0011 225

#include "fastchromopainter.h"
#include <zlib.h>
#include <limits>

//#include "stdafx.h"

///To implement: read in data as binary, and output panel as binary if needed
///To implement: output an ordered matrix for a "back-in-time" analysis?

///Code takes in a set of files (see "Usage" below) and then runs chromosome painting

///User must input a mutation probability (prob. of two different characters per site, not normally above 0.5)
///Also input a recombination file - rates can be zero, and a recombination scaling parameter rho
///prob of recombining between two sites a recombination distance "dist" apart is (1-exp(-rho*dist))
///switches are uniform on the possibilities, though altering this will NOT reduce speed too much
///Code is deliberately pared down so does not allow e.g. missing data in the input file
///Aim is to be a very fast painter, reproducing basic "ChromoPainter" functionality
///Would be easy to adapt for e.g. all vs. all painting of the panel
///Currently only outputs "chunklengths" for an individual. This could be altered with small speed reductions

// void print_matrix_to_file(probs_type **stateprobs, int n_sites, int n_sequences, const char* filename) {
//     std::ofstream file(filename);
//     if (!file.is_open()) {
//         std::cerr << "Error opening file: " << filename << std::endl;
//         return;
//     }
    
//     // Array to hold the sum of probabilities for each site
//     probs_type *sums = new probs_type[n_sites]();
    
//     // Calculate sums for each site
//     for (int i = 0; i < n_sites; ++i) {
//         for (int j = 0; j < n_sequences; ++j) {
//             sums[i] += stateprobs[i][j];
//         }
//     }
    
//     // Normalize and output to file
//     for (int i = 0; i < n_sites; ++i) {
//         for (int j = 0; j < n_sequences; ++j) {
//             if (sums[i] != 0) // Check to prevent division by zero
//                 file << stateprobs[i][j] / sums[i] << " ";
//             else
//                 file << "0 "; // or handle this case as needed
//         }
//         file << std::endl;
//     }
    
//     // Clean up
//     delete[] sums;
//     file.close();
// }

struct Donor {
    int id;
    probs_type probability;
};

void print_matrix_to_file(probs_type **stateprobs, int n_sites, int n_sequences, const char* filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error opening file: " << filename << std::endl;
        return;
    }

    // Array to hold the sum of probabilities for each site
    probs_type *sums = new probs_type[n_sites]();
    
    // Calculate sums for each site
    for (int i = 0; i < n_sites; ++i) {
        for (int j = 0; j < n_sequences; ++j) {
            sums[i] += stateprobs[i][j];
        }
    }

    // Normalize, sort, and find the top 20 donors for each site
    for (int i = 0; i < n_sites; ++i) {
        std::vector<Donor> donors;
        for (int j = 0; j < n_sequences; ++j) {
            probs_type normalized_prob = (sums[i] != 0) ? stateprobs[i][j] / sums[i] : 0;
            donors.push_back({j, normalized_prob});
        }

        // Sort donors by probability in descending order
        std::sort(donors.begin(), donors.end(), [](const Donor& a, const Donor& b) {
            return a.probability > b.probability;
        });

        // Output the top 20 donors to the file
        int num_donors = std::min(20, static_cast<int>(donors.size()));
        for (int k = 0; k < num_donors; ++k) {
            file << donors[k].id << ":" << donors[k].probability << " ";
        }
        file << std::endl;
    }

    // Clean up
    delete[] sums;
    file.close();
}

void binary_write(std::ofstream& fout, const std::vector<bool>& x)
{
	std::vector<bool>::size_type n = x.size();
	//fout.write((const char*)&n, sizeof(std::vector<bool>::size_type));
	for (std::vector<bool>::size_type i = 0; i < n;)
	{
		unsigned char aggr = 0;
		for (unsigned char mask = 1; mask > 0 && i < n; ++i, mask <<= 1)
			if (x.at(i)) {
				aggr |= mask;
				//cout << i<<" ";
			}
		//if (aggr == EOF) { cout << "End of file at character " << i << "!" << endl; }
		//if ((int)aggr == 0) { cout << "Zero at character " << i << "!" << endl; }
		//if (i > 6200 && i < 6400) cout << i << " " << aggr<<" "<<(int) aggr<<" ";
		fout.write((const char*)&aggr, sizeof(unsigned char));
	}
}

/*void binary_read(std::ifstream& fin, std::vector<bool>& x)
{
	std::vector<bool>::size_type n;
	fin.read((char*)&n, sizeof(std::vector<bool>::size_type));
	x.resize(n);
	for (std::vector<bool>::size_type i = 0; i < n;)
	{
		unsigned char aggr;
		fin.read((char*)&aggr, sizeof(unsigned char));
		//cout << i << " " << aggr << " " << (int)aggr << " ";
		for (unsigned char mask = 1; mask > 0 && i < n; ++i, mask <<= 1)
			x.at(i) = aggr & mask;
		
	}
}*/


void nrerror(const string error_text)
/* Output error message */
{
	fprintf(stderr, "Unrecoverable fastchromopainter error...\n");
	fprintf(stderr, "%s\n", error_text);
	fprintf(stderr, "...better luck next time! \n");
	/*Return what we have so far to the screen*/
	exit(1);
}

void *c_malloc(long bytes) {
	void *p;
	p = (void*)malloc((size_t)bytes);
	if (p == NULL) {
		printf("\n\tCan't allocate %ld bytes\n", bytes);
		/*Return what we have so far to the screen*/
		exit(1);
	}
	return p;
}


int main(int argc, char *argv[]) {
	total_time = clock();
	ofstream output_file;

	clock_t main_time, main_temp,rates_time,data_time;
	main_time = 0;
	main_temp = clock();
	////Check parameters - either binary save mode, or painting mode
	ofstream binaryoutput,binarypanel;
	ofstream & binarypanelref = binarypanel;
	ofstream & binaryoutputref = binaryoutput;
	bool binary = 0;
	dataclass paneldata;
	dataclass& paneldataref = paneldata;
	dataclass inddata;
	dataclass& inddataref = inddata;
	int orderout = 0;
	string q, v;
	if (argc == 5) {
		string option = argv[3];
		cout << "Option " << option << " chosen" << endl;
		if (option == "-b") {
			cout << "Outputting selected panel rows to binary file: " << argv[4]<<endl;
			q += string(argv[4]);
			q += string(".binphase");
			v += string(argv[4]);
			v += string(".ids");
			binaryoutput.open(q,ios::binary);
			binarypanel.open(v);
			binary = 1;
			paneldata.binaryoutmode = 1;

		}
		else {
			cout << "\n Usage: fastchromopainter panel_file donor_ids -b output_file_header" << argc << endl;
			exit(1);
		}
	}
	else if (argc == 10) {
		string option = argv[9];
		if (option == "-t") {
			cout << "Outputting ordering information: " << argv[4] << endl;
			orderout = 1;
			paneldata.orderout = 1;
		}

	}
	else if (argc<9) {
		cout << "\n Usage: fastchromopainter ind_data_file panel_file donor_ids panel_labels rates_file output_file theta rho "<< argc << endl;
		exit(1);
	}
	
	///Can assume 8 input parameters
	//panel panellist;
	//panel& panellistref = panellist;
	
	if (argc > 5) {
		paneldata.filename = argv[2];
		paneldata.identifierfilename = argv[3];
		paneldata.labelsfilename = argv[4];
		inddata.filename = argv[1];
		output_file.open(argv[6]);
	}
	else if (argc == 5) {
		paneldata.filename = argv[1];
		paneldata.identifierfilename = argv[2];
	}

	paneldata.get_panelinfo();
	data_time = clock();
	paneldata.get_sequences();
	if (binary == true) {
		paneldata.output_binary(binaryoutputref,binarypanelref);
		return 1;
	}
	inddata.get_sequences();
	data_time = clock() - data_time;

	//////Set the number of decimal places (after decimal point) probs_types are output to
	/////holds throughout the program
	output_file << setiosflags(ios::fixed) << setprecision(20);


	ofstream& output_file_ref = output_file;

	///read in recombination rates over region
	///need code here
	rates_time = clock();
	rate rates;
	rate& ratesref = rates;
	rates.rates_file = argv[5];
	rates.get_rates();
	rates_time = clock() - rates_time;
	probs_type theta,rho;
	
	theta = (probs_type)atof(argv[7]);
	rho = (probs_type)atof(argv[8]);
	if (DISPLAY == 1) {
		cout << "Theta: " << theta << " and rho: " << rho << endl;
	}
	forward_time = clock();

	if(paneldata.n_sites != inddata.n_sites) nrerror("Site mismatch between painting panel and haplotypes to process....naughty!");
	if(paneldata.n_sites != rates.npositions) nrerror("Site mismatch between painting panel and recombination map....shocking!");
	////rescale recombination probabilities and distances between pairs of sites
	rates.scale_rates(rho, paneldataref);
	/////no need to rescale mutation probability
	vector<probs_type> sumprobs;
	vector<probs_type> & sumprobsref=sumprobs;
	sumprobs.resize(paneldata.n_sequences,0.0);
	////run forward backward algorithm to get probabilities of each state
	paneldata.getstateprobs(inddataref,sumprobsref,theta,rho,argv[6]);

	///Output results for each individual
	ofstream indout("indoutput.chunklengths");
	indout << "Recipient ";
	for (int i = 0; i < paneldata.indnames.size(); i++) {
		indout << paneldata.indnames[i] << " ";
	}
	indout << endl;
	indout << "Our_seq ";
	for (int i = 0; i < paneldata.indnames.size(); i++) {
		indout << 100.0*(sumprobs[2*i]+sumprobs[2*i+1]) << " ";
	}
	indout << endl;

	

	vector<probs_type> labelprobs;
	labelprobs.resize(paneldata.labels.size(),0.0);
	//cout << paneldata.labels.size() << " " << sumprobs.size() << " " << labelprobs.size() << " " << paneldata.identifiers.size()<< endl;
	for (int h = 0, maxs = sumprobs.size(); h < maxs; h++) {
		//labelprobs[paneldata.identifiers[h]] += sumprobs[h];
		labelprobs[paneldata.identifiers[h]] += sumprobs[h];
	}


	////output results to file
	size_t i;
	output_file << "Recipient ";
	for (i = 0; i < paneldata.labels.size(); i++) {
		output_file << paneldata.labels[i] << " ";
	}
	output_file << endl;
	output_file << "Our_seq ";
	for (i = 0; i < labelprobs.size(); i++) {
		output_file << (double) 100.0*labelprobs[i] << " ";
	}
	output_file << endl;
	total_time = clock()-total_time;
	forward_time = clock() - forward_time;
	cout << "Time in program (s): " << (double) total_time/ (double) CLOCKS_PER_SEC << endl;
	cout << "Time to get rates (s)" << (double)rates_time / (double)CLOCKS_PER_SEC << endl;
	cout << "Time to get data (s)" << (double)data_time / (double)CLOCKS_PER_SEC << endl;
 	cout << "Time after I/O (s): " << (double) forward_time / (double) CLOCKS_PER_SEC << endl;
	return 0;
}



rate::rate()
{
	if (DISPLAY >= 5) if (DISPLAY5 == 1) cout << "\nCreated instance of rate class\n";
	npositions = 0;

}

rate::~rate() {
	if (DEBUG >= 3) if (DISPLAY5 == 1) cout << "Destroying rate instance" << endl;
}


int rate::get_rates() {
	ifstream datafile;
	int i = 0;
	datafile.open(rates_file);
	if (!datafile.good()) nrerror("Failed to read input rates file");
	char ignoreline[50];
	datafile.getline(ignoreline,50);

	if (DISPLAY == 1) cout << "About to read rates from file " <<rates_file<< endl;
	if (!datafile.good()) nrerror("EOF reached before rates data read in");
	while (datafile.good()) {
		////Read in lines one by one, until end of file is reached
		positions.push_back(-1);
		rates.push_back(-1);
		datafile >> positions[i];
		////check whether actually got another entry
		if (positions[i] != -1) {
			if (i > 0 && positions[i] < positions[i - 1]) {
				cout << positions[i - 1] << " " << positions[i] << endl;
				nrerror("Sites not increasing");
			}
			datafile >> rates[i];
		}
		////if not, undo lengthening of vector - depends on whether EOL character before EOF character
		else {
			positions.pop_back();
			rates.pop_back();
		}
		i++;
	}
	npositions = positions.size();
	datafile.close();
	if (DISPLAY == 1) cout << "Got rates at " <<npositions << " sites"<< endl;
	return 1;
}
/////get a new set of distances between SNPs and return to "ourdata"
////also get weights on each SNP
int rate::scale_rates(probs_type rho, dataclass & ourdata)
{
	if (DISPLAY5 == 1) cout << "Rescaling recombination rates" << endl;
	ourdata.rates.resize(rates.size() - 1);
	int i;
	for (i = 0; i < (int)ourdata.rates.size(); i++) {
		ourdata.rates[i] = rates[i] * (probs_type)(positions[i + 1] - positions[i]);

	}
	ourdata.weights.resize(rates.size());
	ourdata.weights[0] = ourdata.rates[0] * (probs_type) 0.5;
	for (i = 1; i < (int)(ourdata.weights.size()-1); i++) {
		if (i > 0) ourdata.weights[i] = (probs_type) 0.5*(ourdata.rates[i] + ourdata.rates[i - 1]);
	}
	ourdata.weights[ourdata.weights.size() - 1] = ourdata.rates[ourdata.rates.size()-1] * (probs_type) 0.5;
	/////now finally transform these to recombination probabilities, only do this once
	for (i = 0; i < (int)ourdata.rates.size(); i++) {
		if (ourdata.rates[i] < 0) {
			cout << "Rates negative!: " << ourdata.rates[i];
			nrerror("Negative rates cannot be dealt with");
		}
		if (rho*ourdata.rates[i] > 10) {
			cout << "Rates very large after rescaling:" << positions[i] << " " << positions[i + 1] << " " << rho*ourdata.rates[i] << endl;
			ourdata.rates[i] = 1.0 - 1e-5;
		}
		else ourdata.rates[i] = (1 - exp(-rho*ourdata.rates[i]));
		//if (DISPLAY5 == 1) {
		//	cout << ourdata.rates[i] << " ";
		//	cout << ourdata.weights[i] << " ";
		//}
	}
	return 0;
		
}

void dataclass::get_sequences() {
	char p;
	ifstream datafile(filename,ios::binary);
	if (!datafile.good()) nrerror("Failed to read sequence input file");
	if (DISPLAY == 1) cout << "About to read sites/sequences from file " << filename << endl;
	datafile >> n_sequences;
	datafile >> n_sites;
	datafile >> p;
	if (DISPLAY == 1) cout << "\nNumber of sequences in file " << n_sequences << endl;
	if (DISPLAY == 1) cout << "\nNumber of sites " << n_sites << endl;
	if (includeinds.size() == 0) {
		if (DISPLAY == 1) cout << "Reading in all sequences" << endl;
		includeinds.resize(n_sequences, 1);
		identifiers.resize(n_sequences, 0);
		checkset.resize(n_sequences, "Pop1");
	}
	sites.resize(n_sites);
	int i, j;
	for (j = 0; j < n_sites; j++) {

		datafile >> sites[j];
		//if (DISPLAY4 == 1) cout << sites[j] <<" ";
		if (j > 0 && sites[j] < sites[j - 1]) {
			if (DISPLAY5 == 1) cout << j << " " << sites[j - 1] << " " << sites[j] << endl;
			nrerror("Sites not increasing");

		}
		if (!datafile.good()) nrerror("EOF reached before data read in");

	}
	datafile.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
	if (DISPLAY == 1) cout << "\nRead in positions " << n_sites << endl;
	
	//char temp;
	//char *temp;
	char *temp = (char *) c_malloc(2*n_sites+1);
	int curind = 0;
	vector<int> keepinds;
	keepinds.resize(2 * includeinds.size());
	for (i = 0; i < (int) includeinds.size(); i++) {
		keepinds[2 * i] = includeinds[i];
		keepinds[2 * i+1] = includeinds[i];
	}
	list.resize(n_sites);
	size_t n_seq = checkset.size();
	for (i = 0; i < n_sites; i++) {
		list[i].resize(n_seq, '0');
	}
	//cout << checkset.size() << " The number of seq" << endl;
	///test for binary
	string filestring(filename);
	string test;
	test += string(".binphase");

	if (filestring.find(test)!=string::npos) {
		if(DISPLAY==1) cout<<"Reading binary data file"<<endl;
		if (!datafile.good()) nrerror("EOF reached before data read in");
		//vector<bool> data;
		//vector<bool> & seqref = data;
		//binary_read(datafile, seqref);

		////Read in ALL the sequences and make a vector
		long n = n_sequences*n_sites;
		//cout << n << endl;
		//datafile >> n;
		//cout << n << endl;
		//char q;
		//datafile >> q;
		//cout << q << endl;
		//for (int i = 0; i < 100; i++) {
			//datafile >> q;
			//cout << q << endl;
		//}
		//cout << seqref.size()<<" is how much read in!"<<endl;
		//char *temp2 = (char *)c_malloc(n);
		stringstream buffer;
		buffer<< datafile.rdbuf();
		string temp;
		temp = buffer.str();
		//cout << temp.size()<<" Size of buffer"<<endl;
		//cout << temp << endl;
		const char *temp2 = temp.data();
		//datafile.read(temp2,n);
		//string q;
		//string & qq = q;
		//datafile.read(qq,n)
		//for (int w = 0; w < 100; w++) cout << temp2[w] << " ";
		vector<char> data;
		data.resize(n,'1');
		///fill in data vector first
		long k = 0;
		
		unsigned char curchar;
		unsigned char ss;
		k = 0;
		//cout << n << endl;
		char cheat = '0' - '1';
		vector<char> svec;
		svec.resize(8);
		svec[0] = 1;
		for (int i = 1; i < 8; i++) {
			svec[i] = svec[i - 1]; svec[i] <<= 1;
		}
		long chartot = temp.size();
		
		char tempvals[8];
		long begin = 0;
		for (k = 0; k < (chartot-1); k++) {
			curchar = temp2[k];
			for (int z = 0; z < 8; z++) {
				tempvals[z] = '1' + !(curchar & svec[z]) *cheat;
			}
			for (int zz=0; zz < 8; begin++,zz++) {
				data[begin] = tempvals[zz];
			}
		}
		k = chartot - 1;
		curchar = temp2[k];
		for (int z = 0; z < 8; z++) {
			tempvals[z] = '1' + !(curchar & svec[z]) *cheat;
		}
		for (int zz = 0; zz < 8,begin<n; begin++, zz++) {
			data[begin] = tempvals[zz];
		}
		


		/*for (vector<char>::size_type i = 0; i < n;k++)
		{
			curchar = temp2[k];

			for (ss = 1; ss > 0 && i < n; i++, ss <<= 1) {
				//if (i <=100) cout << ss << endl;
				data[i] = data[i] + !(curchar & ss) *cheat;
				//if (!(curchar & ss)) data[i] = '0';
				
			}
		}*/
		
		//cout << k << " k val" << endl;
		k = 0;
		vector<char>::iterator curleft = data.begin();
		vector<char>::iterator curright = curleft + n_sequences;
		vector<char> tempvec,tempvec2;
		////only the ones we will keep
		tempvec.resize(n_seq);

		for (long i = 0; i < n_sites; i++) {

			for (j = 0, k = 0; j < n_sequences; j++, curleft++) {
				if (keepinds[j]) {
					tempvec[k] = *curleft;
						k++;
				}
			}
			vector<char> & listref = list[i];
			listref.assign(tempvec.begin(), tempvec.end());
			//curleft += n_seq;
			//curright += n_seq;
		}
		curind = n_seq;
	}
	else {
		for (i = 0; i < n_sequences; i++) {
			if (!datafile.good()) nrerror("EOF reached before data read in");
			///If including, read in the row - all the sites for an individual
			if (keepinds[i] == 1) {
				//for (j = 0; j < n_sites; j++) {
					//datafile >> temp;	
				datafile.getline(temp, 2 * n_sites);
				for (j = 0; j < n_sites; j++) {
					if (temp[j] == '1') list[j][curind] = temp[j];
					else if (temp[j] != '0') {
						cout << temp[j] << endl;
						nrerror("Bad input file data");
					}
				}
				//if(temp=='0') list[j][curind] = temp;
				//if (temp != '0' & temp != '1') {
				//	cout << temp<<endl;
				//	nrerror("Bad input file data");
				//}
			//}
				curind = curind + 1;
				//datafile.ignore();
				//datafile.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
			}
			////or don't read in the row
			else {
				///ignore next line
				datafile.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
				//for (j = 0; j < n_sites; j++) {
				//	datafile >> temp;
					//list[j][curind] = temp;
					/*if (temp != '0' & temp != '1') {
						cout << temp << endl;
						nrerror("Bad input file data");
					}*/
					//}

			}
		}
	}
	n_sequences = curind;
	datafile.close();
	//cout << list.size() << endl;
	//for (int k = 0; k < list.size(); k++) cout << sums[k] << " ";
	//cout << endl;
	if (DISPLAY == 1) cout << "Read in " << curind << " chromosomes" << endl;
	if (DISPLAY == 1) cout << "...done" << endl;
	// free(temp);
	

}


void dataclass::get_panelinfo() {
	ifstream datafile(identifierfilename);
	if (!datafile.good()) nrerror("Failed to read Donor individual labels input file");
	if (DISPLAY == 1) cout << "About to read Donor individual labels from file " << identifierfilename << endl;
	////process to produce identifiers; first must store three vector

	

	while (!datafile.eof()) {
		string temp1;
		int temp3;
		getline(datafile,temp1, ' ');
		if (temp1.size() > 0) {
			indnames.push_back(temp1);
			///below should just be "D"
			getline(datafile, temp1, ' ');
			popnames.push_back(temp1);
			datafile >> temp3;
			includeinds.push_back(temp3);
			datafile.ignore(10000, '\n');
		}
	}

	totalinds = includeinds.size();

	if (DISPLAY == 1) cout << "...done, read information for " << totalinds<< " individuals"<<endl;
	datafile.close();

	
	for (int i = 0; i < (int)includeinds.size(); i++) {
		if (includeinds[i] == 1) {
			checkset.push_back(popnames[i]);
			checkset.push_back(popnames[i]);
		}
	}

	
	if (binaryoutmode == 1) return;
	
	datafile.open(labelsfilename);
	if (!datafile.good()) nrerror("Failed to read Donor population IDs input file");
	if (DISPLAY == 1) cout << "About to read Donor population IDs from file" << labelsfilename << endl;
	////store in labels
	while (!datafile.eof()) {
		string temp1;
		getline(datafile, temp1, ' ');
		if (temp1.size() > 0) {
			labels.push_back(temp1);
			///below should just be "D"
			datafile >> temp1;
			datafile.ignore(10000, '\n');
		}
	}
	datafile.close();
	
	
	int totalpops = labels.size();
	if (DISPLAY == 1) cout << "...done, read information for " << (int) totalpops<<" populations"<<endl;
	/////now do the bookkeeping
	identifiers.resize(checkset.size(),-1);
	for (int i = 0; i < (int) labels.size(); i++) {
		for (int j = 0; j < (int) checkset.size(); j++) {
			if (labels[i] == checkset[j]) identifiers[j] = i;
		}

	}
	if (DISPLAY4 == 1) {
		for (int j = 0; j < (int) checkset.size(); j++) cout << identifiers[j] << " ";
		cout << endl;
	}
}
void dataclass::output_binary(ofstream & binaryfile, ofstream & panelfile) {
	////function to output binary version of a given input, and an "ids" file to go with it
	////does not output labels file as this does not seem needed
	cout << "..entering here.. " << endl;
	cout << binaryfile.is_open() << endl;
	binaryfile << n_sequences << endl;
	binaryfile << n_sites << endl;
	binaryfile << "P ";
	for (long j = 0; j < n_sites; j++) {
		binaryfile << sites[j] << " ";
	}
	binaryfile << endl;
	///output sequences as a single line
	vector<bool> seqs;
	seqs.resize(n_sites*n_sequences, false);
	for (long i = 0, pos=0; i < n_sites; i++) {
		for (long j = 0; j < n_sequences; j++) {
			if(list[i][j]=='1') seqs[pos] = true;
			pos++;
		}
	}

	vector<bool> seqsref = seqs;
	binary_write(binaryfile, seqsref);
	/*for (long i = 0; i < 100; i++) cout << seqs[i] << " ";
	cout << endl;*/
	for (long i = 0; i < 100; i++) cout << list[list.size()-1][i] << " ";
	cout << endl;
	////now output as binary, the sequences "seqs"
	/*std::vector<bool>::size_type len = seqs.size();
	cout << len << "The length of seq" << endl;
	for (std::vector<bool>::size_type i = 0; i < len;)
	{
		unsigned char outchar = 0;
		for (unsigned char ss = 1; ss > 0 && i < len; ++i, ss <<= 1)
			if (seqs.at(i))
				outchar |= ss;
		binaryfile.write((const char*)&outchar, sizeof(unsigned char));
	}*/

	///panel file, output information on samples we've just output
	for (long i = 0; i < includeinds.size(); i++) {
		if (includeinds[i]!=0) {
			panelfile << indnames[i] << " " <<popnames[i]<< " "<< includeinds[i] << endl;
		}
	}
}
void dataclass::getstateprobs(dataclass & inddata, vector<probs_type> & sumprobs, probs_type theta, probs_type rho, const char * filename_out) {

	////function for forward backward algorithm
	time_t fbtime = clock(),setuptime=clock();
	long i,j,maxii,maxval;
	size_t k;
	vector<vector<char>> diffmat;

	////If need order information
	vector<vector<long>> ordermat;
	vector<vector<probs_type>> ordermat2,ordermat3;
	vector<probs_type> quantile_index;
	vector<probs_type> labeltots;
	if (orderout == 1) {
		ordermat2.resize(n_sequences);
		quantile_index.resize(n_sequences);
		labeltots.resize(labels.size(), 0.0);
		for (long j = 0; j < n_sequences; j++) {
			labeltots[identifiers[j]] += 1.0;
		}
		for (long j = 0; j < labels.size(); j++) {
			labeltots[j] = 1.0 / labeltots[j];
		}

		ordermat3.resize(n_sequences);
		for (long j = 0; j < n_sequences; j++) {
			quantile_index[j] = j;
			ordermat3[j].resize(labels.size(), 0.0);
		}

	}
	
	///For forward backward calculations
	diffmat.resize(n_sites);
	size_t n_seq = checkset.size();
	for (i = 0; i < n_sites; i++) {
		diffmat[i].resize(n_seq);
	}

	////Things will need for individual state probs - initially forward vector
	/*vector<vector<probs_type> > stateprobs;
	stateprobs.resize(n_sites);
	for (i = 0; i < stateprobs.size(); i++) {
		stateprobs[i].resize(n_sequences);
	}*/

	//vector<probs_type> curbackwards;
	//curbackwards.resize(n_sequences);
////compare
	probs_type ** stateprobs;
	stateprobs = (probs_type **)c_malloc(sizeof(probs_type *) * (long) n_sites);
	stateprobs[0] = (probs_type *)c_malloc(sizeof(probs_type) * (long) n_sites * (long) n_sequences);

	for (i = 0; i < n_sites; i++)
		stateprobs[i] = (*stateprobs + n_sequences * i);

	probs_type *curbackwards;
	curbackwards = (probs_type *)c_malloc(sizeof(probs_type)*(long)n_sequences);
	probs_type *curforward;
	curforward = (probs_type *)c_malloc(sizeof(probs_type)*(long)n_sequences);
	probs_type *emp;
	emp = (probs_type *)c_malloc(sizeof(probs_type)*(long)n_sequences);
	//if (DISPLAY == 1) cout << "Set up time " << (clock() - setuptime) / (double)CLOCKS_PER_SEC << endl;
	///Constant term

	///current forward log term
	vector<probs_type> logforward;
	///current backward log term
	vector<probs_type> logbackward;
	vector<probs_type> logsums;

	//vector<char> diffvec;
	//diffvec.resize(n_sequences);
	//vector<probs_type> emp;
	//emp.resize(n_sequences);
	
	logforward.resize(n_sites, 0);
	logbackward.resize(n_sites, 0);


	///current sum forward
	probs_type cursumforward,tempsumforward;
	///current sum backward
	probs_type cursumbackward,tempsumbackward;
	
	///emissions if mismatch
	//cout << "Theta:" << theta<<endl;
	probs_type pmismatch = theta/(1-theta);
	///log term if match
	probs_type pmatch = log(1 - theta);

	if (DISPLAY5 == 1) cout << "Mutation terms "<<pmismatch << " " << pmatch << endl;
	//vector<probs_type> & curforward;
	//vector<probs_type> & prevforward;
	//vector<probs_type> & curbackward;
	//vector<probs_type> & prevbackward;
	//vector<char> & curdata;
	char curtype;
	probs_type normterm;
	vector<probs_type> ratesvec(rates),ratesvec2(rates);
	probs_type tt;
	tt = 1 / (probs_type)n_sequences;
	for (j = 0, maxii = rates.size(); j < maxii; j++) {
		ratesvec[j] = ratesvec[j] / (1 - ratesvec[j])*tt;
	}
		
	for (j = 0, maxii = rates.size(); j < maxii; j++) {
		ratesvec2[j] = log(1 - ratesvec2[j]);	
	}
	//if (DISPLAY5 == 1) {
	//	for (i = 0; i < ratesvec.size(); i++) {
	//		cout << ratesvec[i] << " ";
	//		cout << ratesvec2[i] << " ";
	//	}
	//}
	//cout << "Mismatch prob.:"<<pmismatch << endl;
	//if (DISPLAY == 1) cout << "Set up time " << (clock() - setuptime) / (double)CLOCKS_PER_SEC << endl;
	for (k = 0; k < inddata.list[0].size(); k++) {
		//////forward pass
		/////start by constructing a 0-1 matrix of characters taking value 0 if a match
		////only flip if need to
		i = 0;
		vector<char> & curdata = list[i]; 
		maxval = curdata.size();
		cout << maxval << endl;
		for (i = 0; i < n_sites; i++) {
			curtype = inddata.list[i][k];
			char oldtype;
			if(k>0) oldtype= inddata.list[i][k-1];
			vector<char> & diffvec = diffmat[i];
			vector<char> & curdata = list[i];
			if (k == 0) {
				if (curtype == '0') {
					for (j = 0; j < maxval; j++) {
						diffvec[j] = curdata[j] - '0';
					}
				}
				if (curtype == '1') {
					for (j = 0; j < maxval; j++) {
						diffvec[j] = '1' - curdata[j];
					}
				}
			}
			else {
				if (curtype != oldtype) {
					for (j = 0; j < maxval; j++) {
						diffvec[j] = 1 - diffvec[j];
					}
				}
			}
		}
		probs_type mismatchval[2];
		/*curtype = inddata.list[i][k];
		//probs_type *curforward= stateprobs[i];
		////fill in the initial emission probability time prior state prob
		//for (int j = 0, maxval=curdata.size(); j <maxval; j++) {
		//	curforward[j] = tt;
		//}
		///debug
		vector<char> & curdata = list[i]; 
		maxval = curdata.size();
		vector<char> & diffvec = diffmat[i];
		if (curtype == '0') {
			for (j = 0; j < maxval; j++) {
				diffvec[j] = curdata[j] - '0';
			}
		}
		if (curtype == '1') {
			for (j = 0; j < maxval; j++) {
				diffvec[j] = '1'-curdata[j];
			}
		}*/

		i = 0;
		vector<char> & diffvec = diffmat[i];
		mismatchval[0] = tt; mismatchval[1] = pmismatch*tt;
		for (j = 0; j <maxval; j++) {
			//if (curtype != curdata[j]) curforward[j] = pmismatch*tt;
			//curforward[j] = mismatchval[(int)(curdata[j] - '0')];
			curforward[j] = mismatchval[diffvec[j]];
			//cout << curtype << curdata[j] << " ";
		}
		
		logforward[i] = pmatch;

		probs_type *fillforward = stateprobs[i];
		for (j = 0; j < maxval; j++) {
			fillforward[j] = curforward[j];
		}
		//if (DISPLAY == 1) cout << "Time in forward-backward algorithm after first site: " << (clock() - fbtime) / (double)CLOCKS_PER_SEC << endl;

		/*for (i = 0; i < n_sites; i++) {
			vector<char> & curdata2 = list[i];
			//probs_type *temp = stateprobs[i];
			for (j = 0; j < n_sequences; j++) {
				//*temp = curdata2[j];
				curdata2[j] = curdata2[j] - '0';
				//if (curdata2[j] == '1') curdata2[j] = 1;
				//else curdata2[j] = 0;
			}
		}*/

		mismatchval[0] = 1.0;
		mismatchval[1] = pmismatch;
		for (i = 1; i < (size_t) n_sites; i++) {
			vector<char> & curdata2 = list[i];
			curtype = inddata.list[i][k];
			////faster lookups to where filling in 
			//probs_type *curforward = stateprobs[i];
			probs_type *prevforward = stateprobs[i - 1];
			cursumforward = 0.0;
			///calculate total sum
			probs_type *q;
			///set current forward equal to previous forward value
			////now add up
			q = prevforward;
			for (j = 0; j <maxval; j++,q++) {
				cursumforward += *q;
			}
			tempsumforward = cursumforward;
			cursumforward *= ratesvec[i-1];
			/////constant term
			logforward[i] = logforward[i - 1] + ratesvec2[i-1]+pmatch;

			///try to vectorize the following things
			//curforward.assign(prevforward.begin(), prevforward.end());
			

			/*q = prevforward;
			probs_type *p;
			p = curforward;
			for (int j = 0, maxval = curdata2.size(); j <maxval; j++,p++,q++) {
				*p = *q+cursumforward;
			}*/
			/////this is going to vectorize and then check condition after returning
			////debug

			vector<char> & diffvec = diffmat[i];
			/*if (curtype == '0') {
				for (j = 0; j < maxval; j++) {
					diffvec[j] = curdata2[j] - '0';
				}
			}
		
			if (curtype == '1') {
				for (j = 0; j < maxval; j++) {
					diffvec[j] = '1' - curdata2[j];
				}
			}*/
			for (j = 0; j < maxval; j++) {
				emp[j] = mismatchval[diffvec[j]];
			}
			probs_type *p, *r;
			p = stateprobs[i];
			q = prevforward;
			r = emp;
			////Key step of forward algorithm adding on emission probability, current sum and existing term
			///Therefore use pointers, and combine operations

			for (j = 0; j <maxval; j++,p++,q++,r++) {
				*p = (*r)*(*q + cursumforward);
			}
			//////check if small so roundoff is possible
			if (tempsumforward < sum_tol || tempsumforward>sum_inv_tol) {
				if (DISPLAY4 == 1) cout << "renormalising!"<< endl;
				normterm = (probs_type) 1.0 / tempsumforward;
				logforward[i] += log(tempsumforward);
				p = stateprobs[i];
				for (j = 0; j <maxval; j++,p++) {
					*p *= normterm;
				}
			}
			////record answer, having done local calculations
			/*q = stateprobs[i];
			p = curforward;
			for (int j = 0, maxval = curdata.size(); j < maxval; j++, p++, q++) {
				*q = *p;
			}*/
			//if (DISPLAY5 == 1) cout << logforward[i] << " "<<tempsumforward<<" ";
		}
		//if (DISPLAY == 1) cout << "Time in forward-backward algorithm after forward: " << (clock() - fbtime) / (double)CLOCKS_PER_SEC << endl;

		/////now backward pass
		i =(size_t) (n_sites-1);
		curtype = inddata.list[i][k];
		///curbackwards replaces curforward
		////fill in the initial emission probability time prior state prob
		for (j = 0; j <maxval; j++) {
			curbackwards[j] = 1.0;
		}
		logbackward[i] = 0.0;

		for (long q = n_sites - 2; q>=0; q--) {
			i = (size_t) q;
			////need data at previous site for emission
			vector<char> & curdata3 = list[i+1];
			curtype = inddata.list[i+1][k];
			cursumbackward = 0.0;
			vector<char> & diffvec = diffmat[i+1];
			////first multiply by previous emissions probabilities
			/////this is going to vectorize and then check condition after returning
			//debug
			/*for (int j = 0, maxval = curdata3.size(); j <maxval; j++) {
				if (curdata3[j] != curtype) curbackwards[j] *= pmismatch;
			}*/
			
			/*if (curtype == '0') {
				for (j = 0; j < maxval; j++) {
					diffvec[j] = curdata3[j] - '0';
				}
			}
			if (curtype == '1') {
				for (j = 0; j < maxval; j++) {
					diffvec[j] = '1' - curdata3[j];
				}
			}*/
			for (j = 0; j < maxval; j++) {
				emp[j] = mismatchval[diffvec[j]];
			}
			/////multiply by emission probability before summing
			probs_type *v;
			v = curbackwards;
			for (j = 0; j <maxval; j++,v++) {
				*v *= emp[j];
			}

			///calculate total sum
			v = curbackwards;
			for (j = 0; j <maxval; j++,v++) {
				cursumbackward += *v;
			}
			tempsumbackward = cursumbackward;
			cursumbackward *= ratesvec[i];
			/////constant term
			logbackward[i] = logbackward[i + 1] + ratesvec2[i] + pmatch;

			///add on term for recombining, to each entry
			v = curbackwards;
			for (j = 0; j <maxval; j++,v++) {
				*v += cursumbackward;
			}
			//////check if small so roundoff is possible
			if (tempsumbackward < sum_tol || tempsumbackward> sum_inv_tol) {
				if (DISPLAY4 == 1) cout << "renormalising! " << tempsumbackward<<endl;
				normterm = (probs_type) 1.0 / tempsumbackward;
				logbackward[i] += log(tempsumbackward);
				v = curbackwards;
				for (j = 0; j <maxval; j++,v++) {
					*v *= normterm;
				}
			}
			//if (DISPLAY5 == 1) cout << logbackward[i] << " " << tempsumbackward << " "<<i<<" ";
			///now combine in (to store)
			/////don't need to run below for final site!
			probs_type *jointprobs = stateprobs[i];
			v = curbackwards;
			for (j = 0; j <maxval; j++, jointprobs++,v++) {
				*jointprobs = *jointprobs * (*v);
			}
		}
		//if (DISPLAY == 1) cout << "Time in forward-backward algorithm after backward: " << (clock() - fbtime) / (double)CLOCKS_PER_SEC << endl;

		////now we need to process these to get probabilities
		logsums.assign(logforward.begin(),logforward.end());
		//vector<probs_type> logsums(logforward);
		//cursums.resize(n_sites);
		for (j = 0; j < n_sites; j++) logsums[j] += logbackward[j];
		probs_type total = 0.0;
		probs_type *jointprobs = stateprobs[0];
		for (j = 0; j < maxval; j++) {
			total += jointprobs[j];
		}
		////sum of forward times backward at position 1
		total = log(total);
		/////stored term scaling likelihood
		probs_type vv = logsums[0];
		//////at position 1 likelihood is total+logsums[0]
		/////at position k log-likelihood is log(totalj)+logsums[j]
		/////log(totalj)=total+logsums[0]-logsums[j]=total+vv-logsums[j]
		////vectorize this calculation
		//cout << total << endl;
		//cout << vv << endl;
		if(DISPLAY==1) cout<<"Overall log-likelihood: "<<vv+total<<endl;
		for (j = 0, maxval=n_sites; j < maxval; j++) logsums[j] =(vv-logsums[j]);
		//for (int j = 0; j < n_sites; j++) logsums[j] = exp((double)(total+ logsums[j]));
		for (j = 0, maxval = n_sites; j < maxval; j++) logsums[j] = (total + logsums[j]);
		for (j = 0, maxval = n_sites; j < maxval; j++) logsums[j] = exp(-logsums[j]);
		for (j = 0, maxval = n_sites; j < maxval; j++) logsums[j] *= weights[j];

		//probs_type tempwe=0.0;
		//for (int j = 0, maxval = n_sites; j < maxval; j++) tempwe = tempwe + weights[j];
		//cout << tempwe << endl;
		////a bit misleading the name above, but efficient
		///avoid summing probabilities at individual sites - know result from above code
		///because likelihood is same for every site, use to normalise via logsums
		//if (DISPLAY == 1) cout << "Time in forward-backward algorithm before normalising: " << (clock() - fbtime) / (double)CLOCKS_PER_SEC << endl;

		////add results up for copying from particular individual at particular site
		for (j = 0, maxval = n_sites; j < maxval; j++) {
			probs_type *curprobs = stateprobs[j];
			probs_type tempc = logsums[j];
			for (int k = 0, maxval2=n_sequences; k < maxval2; k++)
				sumprobs[k] += curprobs[k]*tempc;
		}
		
		/*for (i = 0; i < (size_t) n_sites; i++) {
			probs_type valuetemp = 0.0;
			vector<probs_type> & jointprobs = stateprobs[i];
			for (int j = 0, maxval = curdata.size(); j <maxval; j++) {
				valuetemp+= jointprobs[j];
			}
			cursums[i] = log(valuetemp);
			if(DISPLAY4==1) cout << logforward[i] << " " << logbackward[i] << " " << cursums[i] << " " << logsums[i]<<" "<<logforward[i] + logbackward[i] + cursums[i] << endl;
		}*/
		///sum(jointprobs[j]) is same for every position I think so normalise at end.
		//if (DISPLAY == 1) cout << "Time in forward-backward algorithm: " << (clock() - fbtime) / (double)CLOCKS_PER_SEC << endl;
		if (orderout == 1) {
			cout << "Processing ordering information" << endl;
			ordermat.resize(n_sites);
			ordermat2.resize(n_sequences);
			for (long j = 0; j < n_sequences; j++) {
				ordermat2[j].assign(n_sequences, 0.0);
			}
			vector<long> temporder;
			temporder.resize(n_sequences);
			for (long j = 0; j < n_sequences; j++) temporder[j] = j;
			vector<long> &temporderref = temporder;
			///debug
			/*ofstream checkout("plausiblebestmatch.txt");
			vector<long> totnum;
			totnum.resize(n_sites);*/
			///end debug
			/*for (long j = 0; j < n_sequences; j++) {
				for (long k = 0; k < n_sequences; k++) {
					ordermat2[j][k]=0.0;
				}
			}*/
			vector<probs_type> labelstots2;
			labelstots2.resize(n_sequences);
			for(long j = 0; j < n_sequences; j++) labelstots2[j]=labeltots[identifiers[j]];
			vector<probs_type> labeltots3;
			labeltots3.resize(labeltots.size());
			for (long j = 0; j < n_sites; j++) {
				//ordermat[j].resize(paneldata.n_sequences);
				probs_type *curprobs = stateprobs[j];
				//logsums[j] = logsums[j] / weights[j];
				///Order to be decreasing
				//bool q=true;
				//if(is_sorted(temporder.begin(), temporder.end(),[&curprobs](size_t i1, size_t i2) {return curprobs[i1] > curprobs[i2]; })==false){
					//cout<<j<<" ";
					sort(temporder.begin(), temporder.end(), [&curprobs](size_t i1, size_t i2) {return curprobs[i1] > curprobs[i2]; });
				//}
				//else cout<<j<<" ";
				probs_type tempval=weights[j]*100.0;
				////so k,temporder[k] is the identity for copying from k 
				//vector<vector<probs_type>>::iterator tempit=ordermat2.begin();
				long q=0;
				for(long k = 0, maxs=labeltots.size(); k < maxs; k++) {
					labeltots3[k]=labeltots[k]*tempval;
				}
				for (long k = 0; k < n_sequences; k++) {
					///where will increment answer, i.e. order
					vector<probs_type> &tempvec=ordermat3[quantile_index[k]];
					///identifier of which group is kth in order
					q=identifiers[temporder[k]];
					///add this weight on to this group
					tempvec[q]+=labeltots3[q];
					//ordermat2[k][temporder[k]] += tempval;
					//(*tempit)[temporder[k]]+=tempval;
				}

				////how many are plausible MRCAs?
				/*totnum[j] = 0;
				long q = 0;
				probs_type tot = 0;
				while (tot < 0.5) {
				tot = tot + curprobs[temporder[q]] * logsums[j];
				q++;
				totnum[j]++;
				}
				checkout << totnum[j] << " ";*/

			}
			////summarise information
			//vector<probs_type> labelstots2;
			//labelstots2.resize(n_sequences);
			//for(long j = 0; j < n_sequences; j++) labelstots2[j]=labeltots[identifiers[j]];
			/*for (long j = 0; j < n_sequences; j++) {
				vector<probs_type> &tempvec=ordermat3[quantile_index[j]];
				vector<probs_type> &tempvec2=ordermat2[j];
				long q;
				for (long k = 0; k < n_sequences; k++) {
					q=identifiers[k];
					tempvec[q] += tempvec2[k] * labeltots[q];
				}
			}*/

		}

	
	std::stringstream ss;
	ss << filename_out << "_stateprobs_" << k << ".txt";
	std::string filename1 = ss.str();
    print_matrix_to_file(stateprobs, n_sites, n_sequences, filename1.c_str());
	}
	//double vm, used;
	//double & vmref = vm;
	//double & usedres = used;
	//process_mem_usage(vmref, usedres);
	//cout << "Memory used: " << used << " of " << vmref << endl;
	/////summing across sites (keep individual level)
	////sort results
	if(orderout==1){
		cout << "..done processing, now outputting to file" << endl;
		string orderout;
		orderout = filename;
		orderout.append("orders.out");
		ofstream newout(orderout);
		newout << setiosflags(ios::fixed) << setprecision(4);
		for (i = 0; i < labels.size(); i++) {
			newout << labels[i] << " ";
		}

		newout << endl;
		for (long j = 0; j < quantile_index.size(); j++) {
			for (long k = 0; k < labels.size(); k++) {
				newout << ordermat3[j][k] << " ";
			}
			newout << endl;
		}


	}

	if(DISPLAY==1) cout<< "Time in forward-backward algorithm: "<<(clock()-fbtime)/(double)CLOCKS_PER_SEC<<endl;
	
	free(stateprobs[0]);
	free(stateprobs);
	// free(curbackwards);
	// free(curforward);
	// free(emp);
}
	

dataclass::dataclass()
{
	if (DISPLAY5 == 1) cout << "\nCreated instance of data class\n";
	n_sequences = -1;
	binaryoutmode = 0;
	orderout = 0;
	//sites = NULL;
	//list = NULL;
}

dataclass::~dataclass() {
	if (DEBUG >= 3) {
		if (DISPLAY5 == 1) cout << "Destroying data instance" << endl;
	}



	if (DISPLAY5 == 1) cout << "Freed dataset" << endl;
}

