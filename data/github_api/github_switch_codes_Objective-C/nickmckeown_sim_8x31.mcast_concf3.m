// Repository: nickmckeown/sim
// File: inputs/SAMPLE_INPUTS/MCAST_CONC/8x31.mcast_concf3.m

# ../sim -l1000
Numswitches 1
Switch 0
	Numinputs    8
	Numoutputs   31
	InputAction  defaultInputAction 
	OutputAction defaultOutputAction
	Fabric       crossbar
	Algorithm	mcast_conc_residue
	0	bernoulli_iid_uniform -u 0.22 -f 3 -m 1.0 
	1	bernoulli_iid_uniform -u 0.22 -f 3 -m 1.0 
	2	bernoulli_iid_uniform -u 0.22 -f 3 -m 1.0 
	3	bernoulli_iid_uniform -u 0.22 -f 3 -m 1.0 
	4	bernoulli_iid_uniform -u 0.22 -f 3 -m 1.0 
	5	bernoulli_iid_uniform -u 0.22 -f 3 -m 1.0 
	6	bernoulli_iid_uniform -u 0.22 -f 3 -m 1.0 
	7	bernoulli_iid_uniform -u 0.22 -f 3 -m 1.0 
	Stats
		Arrivals
		Departures
		Latency 
		Occupancy (*, m)
	Histograms
		Arrivals 
		Departures
		Latency (1, m)
		Occupancy 
