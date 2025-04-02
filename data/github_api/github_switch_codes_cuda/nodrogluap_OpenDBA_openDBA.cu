// Repository: nodrogluap/OpenDBA
// File: openDBA.cu


/*******************************************************************************
 * (c) 2019 Paul Gordon's parallel (CUDA) NVIDIA GPU implementation of the Dynamic Time 
 * Warp Barycenter Averaging algorithm as conceived (without parallel compuation conception) by Francois Petitjean 
 ******************************************************************************/

#include <string>
#include <vector>
#include "openDBA.cuh"

__host__
int main(int argc, char **argv){
	
	int norm_sequences = 1; // signal range
	int prefix_to_skip = 0; // where do we start looking for a prefix when in open_prefix mode?
	int prefix_length = 0; // if non-zero, look only at the first N segments after prefix_to_skip for alignment
	
	char c;
	while( ( c = getopt (argc, argv, "n") ) != -1 ) {
		switch(c) {
			case 'n':
				norm_sequences = 0;
				break;
			default:
				/* You won't actually get here. */
				break;
		}
	}

	if(argc < 9){
		std::cout << "Usage: " << argv[0] << " <binary|text|tsv";
#if SLOW5_SUPPORTED == 1
		std::cout << "|slow5";
#endif	
#if HDF5_SUPPORTED == 1
		std::cout << "|fast5";
#endif
		std::cout << "> ";
#if DOUBLE_UNSUPPORTED == 1
		std::cout << "<short|int|uint|ulong|float> " <<
#else
		std::cout << "<short|int|uint|ulong|float|double> " <<
#endif
		          "<global|open_start|open_end|open> <output files prefix> <minimum unimodal segment length for clustering[,for consensus generation]> <prefix sequence to remove|/dev/null> <clustering threshold> <series.tsv|<series1> <series2> [series3...]>\n";
		exit(1);
     	}

	int num_series = argc-8;
	char *min_segment_length = argv[5]; // reasonable settings for nanopore RNA dwell time distributions would be 4 (lower to 2 for DNA)
	int read_mode = TEXT_READ_MODE;
	if(!strcmp(argv[1],"binary")){
		read_mode = BINARY_READ_MODE;
	}
#if SLOW5_SUPPORTED == 1
	else if(!strcmp(argv[1],"slow5")){
		read_mode = SLOW5_READ_MODE;
	}
#endif	
#if HDF5_SUPPORTED == 1
	else if(!strcmp(argv[1],"fast5")){
		read_mode = FAST5_READ_MODE;
	}
#endif
	else if(!strcmp(argv[1],"tsv")){
		read_mode = TSV_READ_MODE;
	}
	else if(strcmp(argv[1],"text")){
		std::cerr << "First argument (" << argv[1] << ") is neither 'binary' nor 'text'" << std::endl;
		exit(1);
	}

	int use_open_start = 0;
	int use_open_end = 0;
	if(!strcmp(argv[3],"global")){
        }
        else if(!strcmp(argv[3],"open_start")){
		use_open_start = 1;
        }
        else if(!strcmp(argv[3],"open_end")){
		use_open_end = 1;
	}
	// In format open_prefix_#_# where the numbers are the start and end of the segmented sequence positions to inspect
	else if(!strncmp(argv[3],"open_prefix", 11)){
	       	use_open_start = 0;
	       	use_open_end = 1;
		norm_sequences = 1; // TODO: quantile norm when set to 2? 
		std::stringstream ss(std::string(argv[3]+12));
		std::vector <std::string> fields;
		std::string tmp;
		while(std::getline(ss, tmp, '_')){
    			fields.push_back(tmp);
		}
		if(fields.size() != 2){
			std::cerr << "Unexpected alignment type specified, expected open_prefix_##_## but did not find a second underscore in " << argv[3] << std::endl;
			exit(1);
		}
		prefix_to_skip = std::stoi(fields[0]);
		prefix_length = std::stoi(fields[1]);
		std::cerr << "Aligning only the first " << prefix_length << " elements of each sequence" << std::endl;
        }
	else if(!strcmp(argv[3],"open")){
		use_open_start = 1;
		use_open_end = 1;
        }
	else{
		std::cerr << "Third argument (" << argv[3] << ") is not one of the accept values 'global', 'open_start', 'open_end' or 'open'" << std::endl;
                exit(1);
	}

	char *output_prefix = argv[4];

	char *seqprefix_filename = 0;
	if(strcmp(argv[6], "/dev/null")){
		seqprefix_filename = argv[6];
	}

	double cdist = (double) atof(argv[7]);

	int argind = 8; // Where the file names start
	// The following are all the data types supported by CUDA's atomicAdd() operation, so we support them too for best value precision maintenance.
	if(!strcmp(argv[2],"int")){
		setupAndRun<int>(seqprefix_filename, &argv[argind], num_series, output_prefix, read_mode, use_open_start, use_open_end, min_segment_length, norm_sequences, cdist, prefix_to_skip, prefix_length);
	}
	else if(!strcmp(argv[2],"uint")){
		setupAndRun<unsigned int>(seqprefix_filename, &argv[argind], num_series, output_prefix, read_mode, use_open_start, use_open_end, min_segment_length, norm_sequences, cdist, prefix_to_skip, prefix_length);
	}
	else if(!strcmp(argv[2],"ulong")){
		setupAndRun<unsigned long long>(seqprefix_filename, &argv[argind], num_series, output_prefix, read_mode, use_open_start, use_open_end, min_segment_length, norm_sequences, cdist, prefix_to_skip, prefix_length);
	}
	else if(!strcmp(argv[2],"float")){
		setupAndRun<float>(seqprefix_filename, &argv[argind], num_series, output_prefix, read_mode, use_open_start, use_open_end, min_segment_length, norm_sequences, cdist, prefix_to_skip, prefix_length);
	}
	// Only since CUDA 6.1 (Pascal and later architectures) is atomicAdd(double *...) supported.  Remove if you want to compile for earlier graphics cards.
#if DOUBLE_UNSUPPORTED == 1
#else
	else if(!strcmp(argv[2],"double")){
		setupAndRun<double>(seqprefix_filename, &argv[argind], num_series, output_prefix, read_mode, use_open_start, use_open_end, min_segment_length, norm_sequences, cdist, prefix_to_skip, prefix_length);
	}
#endif
	else if(!strcmp(argv[2], "short")){
		// Short is not properly supported in the hardware nor by z-normalization, we will convert to float  (last arg=1)
		setupAndRun<float>(seqprefix_filename, &argv[argind], num_series, output_prefix, read_mode, use_open_start, use_open_end, min_segment_length, norm_sequences, cdist, prefix_to_skip, prefix_length, 1);
	}
	else{
		std::cerr << "Second argument (" << argv[2] << ") was not one of the accepted numerical representations: 'int', 'uint', 'ulong', 'float' or 'double'" << std::endl;
		exit(1);
	}

	// Following needed to allow cuda-memcheck to detect memory leaks
	int deviceCount;
        cudaGetDeviceCount(&deviceCount); CUERR("Getting GPU device count in teardown/cleanup");
	for(int i = 0; i < deviceCount; i++){
                cudaSetDevice(i);
		cudaDeviceReset(); CUERR("Resetting GPU device");
	}

	return 0;
}
