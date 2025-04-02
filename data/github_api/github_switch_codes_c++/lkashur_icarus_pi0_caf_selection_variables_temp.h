/**
 * @file variables.h
 * @brief Header file for definitions of selection variables.
 * @author justin.mueller@colostate.edu
*/
#ifndef VARIABLES_H
#define VARIABLES_H

#define ELECTRON_MASS 0.5109989461
#define MUON_MASS 105.6583745
#define PION_MASS 139.57039
#define PROTON_MASS 938.2720813

#include <algorithm>
#include <iostream>
#include <TVector3.h>
#include <string>
#include <iostream>

namespace vars
{

    /**
     * Variable for counting interactions/particles.
     * @tparam T the type of object (true or reco, interaction or particle).
     * @param interaction/particle to apply the variable on.
     * @return 1.0 (always).
     */
    template<class T>
        double count(const T & obj) { return 1.0; }
   
    /**
     * Variable for id (unique identifier for the object).
     * @tparam T the type of object (true or reco, interaction or particle).
     * @param interaction/particle to apply the variable on.
     * @return the id of the interaction/particle.
    */
    template<class T>
        double id(const T & obj) { return obj.id; }

    /**
     * Variable for enumerating interaction categories.  This is a basic
     * categorization using only signal, neutrino background, and cosmic
     * background as the three categories.
     * 0: 1mu1pi0 (contained and fiducial)
     * 1: 1mu1pi0 (not contained or fiducial)               
     * 2: Other nu                
     * 3: cosmic
     * @tparam T the type of interaction (true or reco).
     * @param interaction to apply the variable on.
     * @return the enumerated category of the interaction.   
     */
    template<class T>
      double category(const T & interaction)
      {
        // Cosmic background               
        double cat(3);

        // Signal
        if(cuts::signal_1mu0pi1pi0(interaction))
          {
	    if(cuts::fiducial_cut(interaction) && cuts::track_containment_cut(interaction))
	      {
		cat = 0;
	      }
	    else cat = 1;
          }
        // Neutrino Background                         
        else if(cuts::other_nu_1mu0pi1pi0(interaction))
          {
            cat = 2;
          }
        return cat;
      }

    /**
     * Variable for enumerating interaction categories. This classifies the          
     * interactions based on the visible final states.
     * 0: 1mu1pi0, 1: 1muNpi0, 2: 1muCex, 3: NC, 4: Other, 5: Cosmic        
     * @tparam T the type of interaction (true or reco).             
     * @param interaction to apply the variable on.      
     * @return the enumerated category of the interaction.                                               
     */
    template<class T> double category_topology(const T & interaction)
      {

	truth_inter s = cuts::true_interaction_info(interaction);
	
	// Cosmic
	uint16_t cat(8);


	// TEST signal def
	//s.num_primary_muons_thresh == 1 && s.num_primary_pions_thresh == 0 && s.num_primary_pi0s_thresh == 1 && s.num_nonprimary_pi0s == 0 && s.is_cc && s.is_neutrino
	// TEST

	// Neutrino
	if(s.is_neutrino)
	  {
	    // 1mu0pi1pi0 (in-phase)
	    if(s.num_primary_muons_thresh == 1 && s.num_primary_pions_thresh == 0 && s.num_primary_pi0s_thresh == 1 && s.num_nonprimary_pi0s == 0 && s.is_cc && s.is_fiducial && s.has_contained_tracks)
	      {
		cat = 0;
	      }
	    // 1mu0pi1pi0 (out-of-phase)
	    else if( (s.num_primary_muons == 1 && s.num_primary_pions == 0 && s.num_primary_pi0s == 1 && s.num_nonprimary_pi0s == 0 && s.is_cc) && (s.num_primary_muons_thresh != 1 || s.num_primary_pions_thresh != 0 || s.num_primary_pi0s_thresh != 1 || !s.is_fiducial || !s.has_contained_tracks) )
	      {
		cat = 1;
	      }
	    // 1muNpi1pi0 (no secondary pi0s)
	    else if(s.num_primary_muons == 1 && s.num_primary_pions > 0 && s.num_primary_pi0s == 1 && s.num_nonprimary_pi0s == 0 && s.is_cc)
	      {
		cat = 2;
	      }
	    // 1muNpi1pi0 (with secondary pi0s)
	    else if(s.num_primary_muons == 1 && s.num_primary_pions > 0 && s.num_primary_pi0s == 1 && s.num_nonprimary_pi0s > 0 && s.is_cc)
	      {
		cat = 3;
	      }
	    // 1muNpi0pi0 (with secondary pi0s)
	    else if(s.num_primary_muons == 1 && s.num_primary_pions > 0 && s.num_primary_pi0s == 0 && s.num_nonprimary_pi0s > 0 && s.is_cc)
	      {
		cat = 4;
	      }
	    // 1muNpi0
	    else if(s.num_primary_muons == 1 && s.num_primary_pi0s > 1 && s.is_cc)
	      {
		cat = 5;
	      }
	    // NC pi0
	    else if(s.num_primary_muons == 0 && !s.is_cc)
	      {
		cat = 6;
	      }
	    // Other
	    else
	      {
		cat = 7;
	      }

	  }
	
	return cat;
      }

    /**
     * Variable for enumerating interaction categories. This categorization
     * uses the interaction type (generator truth) classify the interactions
     * 0: nu_mu CC QE, 1: nu_mu CC Res, 2: nu_mu CC MEC, 3: nu_mu CC DIS, 4: nu_mu CC Coh, 5: nu_e CC, 6: NC, 7: Cosmic
     * @tparam T the type of interaction (true or reco).
     * @param interaction to apply the variable on.
     * @return the enumerated category of the interaction.
    */
    template<class T>
        double category_interaction_mode(const T & interaction)
        {
            double cat(7);

            if(interaction.nu_id > -1)
            {
                if(interaction.current_type == 0)
                {
                    if(abs(interaction.pdg_code) == 14)
                    {
                        if(interaction.interaction_mode == 0) cat = 0;
                        else if(interaction.interaction_mode == 1) cat = 1;
                        else if(interaction.interaction_mode == 10) cat = 2;
                        else if(interaction.interaction_mode == 2) cat = 3;
                        else if(interaction.interaction_mode == 3) cat = 4;
                        else cat = 8;
                    }
                    else cat = 5;
                }
                else cat = 6;
            }

            return cat;
        }

    /**
     * Variable for counting particles in interactions.
     * @tparam T the type of interaction (true or reco).
     * @param interaction to apply the variable on.
     * @return the number of particles in the interaction.
     */
    template<class T>
        double count_particles(const T & interaction) { return interaction.num_particles; }

    /**
     * Variable for counting primaries in interactions.
     * @tparam T the type of interaction (true or reco).
     * @param interaction to apply the variable on.
     * @return the number of primaries in the interaction.
     */
    template<class T>
        double count_primaries(const T & interaction) { return interaction.num_primaries; }
    
    /**
     * Variable for energy of the neutrino primary of the interaction.
     * @tparam T the type of interaction (true or reco).
     * @param interaction to apply the variable on.
     * @return the neutrino energy.
    */
    template<class T>
        double neutrino_energy(const T & interaction) { return 1000*interaction.nu_energy_init; }

    /**
     * Variable for matched interaction flash time.
     * @tparam T the type of interaction (true or reco).
     * @param interaction to apply the variable on.
     * @return the matched flash time of the interaction.
    */
    template<class T>
        double flash_time(const T & interaction)
        {
            if(!cuts::valid_flashmatch(interaction))
                return -100000.0;
            else
                return interaction.flash_time;
        }

    /**
     * Variable for particle primary categorizations.
     * @tparam T the type of particle (true or reco).
     * @param particle to apply the variable on.
     * @return the primary/non-primary designation of the particle.
    */
    template<class T>
        double primary(const T & particle) { return particle.is_primary ? 1 : 0; }

    /**
     * Variable for particle PID.
     * @tparam T the type of particle (true or reco).
     * @param particle to apply the variable on.
     * @return the PID of the particle.
    */
    template<class T>
        double pid(const T & particle) { return particle.pid; }

    /**
     * Variable for particle PID + primary.
     * @tparam T the type of particle (true or reco).
     * @param particle to apply the variable on.
     * @return the PID+primary information for the particle.
    */
    template<class T>
        double primary_pid(const T & particle) { return particle.pid + (particle.is_primary ? 5 : 0); }

    /**
     * Variable for particle csda_ke.
     * @tparam T the type of particle (true or reco).
     * @param particle to apply the variable on.
     * @return the csda_ke of the particle.
    */
    template<class T>
        double csda_ke(const T & particle) { return particle.csda_ke; }

    /**
     * Variable for particle calo_ke.
     * @tparam T the type of particle (true or reco).
     * @param particle to apply the variable on.
     * @return the calo_ke of the particle.
    */
    template<class T>
        double calo_ke(const T & particle)
        {
	  return particle.calo_ke; // Nominal
        }
 
    /**
     * Variable for true particle energy (total).
     * @tparam T the type of particle (true or reco).
     * @param particle to apply the variable on.
     * @return the total energy of the paricle (truth only).
     */
    template<class T>
      double energy_init(const T & particle) {return particle.energy_init; }
    
    /**
     * Variable for particle kinetic energy.
     * @tparam T the type of particle (true or reco).
     * @param particle to apply the variable on.
     * @return the kinetic energy of the particle.
     */
    template<class T>
      double energy_ke(const T & particle) {return particle.ke;}

    /**
     * Variable for true particle energy starting kinetic energy.
     * @tparam T the type of particle (true or reco).
     * @param particle to apply the variable on.
     * @return the starting kinetic energy of the particle.
    */
    template<class T>
        double ke_init(const T & particle)
        {
            double energy(particle.energy_init);
            switch (particle.pid)
            {
            case 1:
                energy -= ELECTRON_MASS;
                break;
            case 2:
                energy -= MUON_MASS;
                break;
            case 3:
                energy -= PION_MASS;
                break;
            case 4:
                energy -= PROTON_MASS;
                break;
            default:
                break;
            }
            return energy;
        }

    /**
     * Find leading muon index.
     * @tparam T the type of interaction (true or reco).
     * @param the interaction to operate on.
     @ @return the index of the leading muon.
     */
    template <class T> size_t leading_muon_index(const T & interaction)
      {
	double leading_ke(0);
	size_t index(0);
	// Loop over particles
	for(size_t i(0); i < interaction.particles.size(); ++i)
	  {
	    const auto & p = interaction.particles[i];
	    double energy = p.csda_ke;
	    if constexpr (std::is_same_v<T, caf::SRInteractionTruthDLPProxy>)
                           energy = p.ke;
	    if(p.pid == 2 && p.is_primary && energy > leading_ke)
	      {
		leading_ke = energy;
		index = i;
	      }
	  }
	return index;
      }

    /**
     * Find indices for true pi0 decay photons.
     * @tparam T the type of interaction (true).
     * @param the interaction to operate on.
     * @return the indices of the true pi0 decay photons.
     */
    
    template <class T>
      vector<size_t> true_pi0_photon_idxs(const T & interaction)
      {
	// Output 
	vector<size_t> photon_idxs;
	
	// Temp. Storage
	unordered_map<int, vector<pair<size_t, double>> > primary_pi0_map;
	vector< pair<size_t, double> > photon_energies;

	// 1mu1pi0 signal
	if(cuts::signal_1mu0pi1pi0(interaction) && cuts::fiducial_cut(interaction) && cuts::track_containment_cut(interaction))
	  {
	    // Loop over particles
	    for(size_t i(0); i < interaction.particles.size(); ++i)
	      {
		const auto & p = interaction.particles[i];

		// Primary pi0
		// Given 1mu1pi0 cut, we already know that any photons 
		// meeting the below criteria belong to a single primary pi0
		if(p.pid == 0 && p.is_primary && p.ke > MIN_PHOTON_ENERGY && p.parent_pdg_code == 111)
		  {
		    primary_pi0_map[p.parent_track_id].push_back({i, p.ke}); 
		  }
	      } // end particle loop
	    
	    // Primary pi0s
	    for (auto const& pi0 : primary_pi0_map)
	      {
		int num_primary_photon_daughters = 0;
		for(auto pair : pi0.second)
		  {
		    if(pair.second > MIN_PHOTON_ENERGY) ++num_primary_photon_daughters;
		  }
		
		if(num_primary_photon_daughters == 2)
		  {
		    for(auto pair : pi0.second)
		      {
			if(pair.second > MIN_PHOTON_ENERGY)
			  {
			    photon_energies.push_back(pair);
			  }
		      }
		  }
	      }

	    // Sort by photon energy (high to low)
	    sort(photon_energies.begin(), photon_energies.end(), [](auto& a, auto& b) {
		return a.second > b.second; });

	    // Fill idxs
	    photon_idxs.push_back(photon_energies[0].first);
	    photon_idxs.push_back(photon_energies[1].first); 
	  } // end 1mu1pi0 signal

	// Fill with nonsense if not a true 1mu1pi0 event
	else
	  {
	    photon_idxs.push_back(0);
            photon_idxs.push_back(0);
	  }
	return photon_idxs;
      }
    
    /**
     * Find true pi0 photon directions.
     * @tparam T the type of interaction (true).
     * @param interaction to operate on.
     * @return the true pi0 photon directions.
     */    
    template<class T>
      vector<TVector3> true_pi0_photon_dirs(const T & interaction)
      {
	// Output
	vector<TVector3> photon_dirs;
	
	vector<size_t> pi0_photon_idxs;
	pi0_photon_idxs = true_pi0_photon_idxs(interaction);
	size_t i(pi0_photon_idxs[0]);
	size_t j(pi0_photon_idxs[1]);

	// Photon 0
	const auto & p = interaction.particles[i];
	TVector3 ph0_dir (p.momentum[0], p.momentum[1], p.momentum[2]);
	ph0_dir = ph0_dir.Unit();
	photon_dirs.push_back(ph0_dir);

	// Photon 1
	const auto & q = interaction.particles[j];
	TVector3 ph1_dir (q.momentum[0], q.momentum[1], q.momentum[2]);
	ph1_dir = ph1_dir.Unit();
	photon_dirs.push_back(ph1_dir);

	return photon_dirs;
      }
    
    /**
     * Find indices for pi0 decay candidate photons.
     * Assumes 1mu2gamma cut has been made.
     * @tparam T the type of interaction (true or reco).
     * @param interaction to operate on.
     * @return the indices of the pi0 decay candidate photons.
     */
    template<class T>
      vector<size_t> pi0_photon_idxs_by_energy(const T & interaction)
      {
	// Output
	vector<size_t> photon_idxs;

	if(cuts::all_1mu0pi2gamma_cut(interaction)) // REMEMBER TO CHANGE BETWEEN DATA AND MC
	  {
	    // Temp. storage
	    vector< pair<size_t, double> > photon_energies;

	    // Loop over particles
	    for(size_t i(0); i < interaction.particles.size(); ++i)
	      {
		const auto & p = interaction.particles[i];
		if(p.pid == 0 && p.is_primary && calo_ke(p) > MIN_PHOTON_ENERGY)
		  {
		    photon_energies.push_back({i, calo_ke(p)});
		  }
	      }

	    // Sort by photon energy (high to low)
	    sort(photon_energies.begin(), photon_energies.end(), [](auto& a, auto& b) {
		return a.second > b.second; });
	    
	    photon_idxs.push_back(photon_energies[0].first);
            photon_idxs.push_back(photon_energies[1].first);
	    
	  }
	// If not 1mu2gamma, return nonsense
	else
	  {
	    photon_idxs.push_back(0);
	    photon_idxs.push_back(0);
	  }
	return photon_idxs;
      }
    
    /**
     * Find pi0 photon directions.
     * @tparam the type of interaction (true or reco).
     * @param interaction to operate on.
     * @return the true pi0 photon directions.
     */
    
    template<class T>
      vector<TVector3> pi0_photon_dirs(const T & interaction)
      {
        // Output
        vector<TVector3> photon_dirs;

        TVector3 vertex(interaction.vertex[0], interaction.vertex[1], interaction.vertex[2]);

        vector<size_t> pi0_photon_idxs;
        pi0_photon_idxs = pi0_photon_idxs_by_energy(interaction);
	//pi0_photon_idxs = pi0_photon_idxs_by_dir(interaction);
        size_t i(pi0_photon_idxs[0]);
        size_t j(pi0_photon_idxs[1]);

        // Photon 0
        const auto & p = interaction.particles[i];
        TVector3 ph0_dir;
        ph0_dir.SetX(p.start_point[0] - vertex[0]);
        ph0_dir.SetY(p.start_point[1] - vertex[1]);
        ph0_dir.SetZ(p.start_point[2] - vertex[2]);
        ph0_dir = ph0_dir.Unit();
        photon_dirs.push_back(ph0_dir);

        // Photon 1 
        const auto & q = interaction.particles[j];
        TVector3 ph1_dir;
        ph1_dir.SetX(q.start_point[0] - vertex[0]);
        ph1_dir.SetY(q.start_point[1] - vertex[1]);
        ph1_dir.SetZ(q.start_point[2] - vertex[2]);
        ph1_dir = ph1_dir.Unit();
        photon_dirs.push_back(ph1_dir);

        return photon_dirs;
      }
    

    /**
     * Find cosine of pi0 opening angle.
     * @tparam T the type of interaction (true or reco).
     * @param interaction to operate on.
     * @return the cosine of pi0 opening angle.
     */
    
    template<class T>
      double pi0_costheta(const T & interaction)
      {
        // Output
        double costheta;

        vector<TVector3> photon_dirs;
        // Truth
        if constexpr (std::is_same_v<T, caf::SRInteractionTruthDLPProxy>)
                       {
                         photon_dirs = true_pi0_photon_dirs(interaction);
                       }

        // Reco 
        else
          {
            photon_dirs = pi0_photon_dirs(interaction);
          }

        costheta = photon_dirs[0].Dot(photon_dirs[1]);

        return costheta;
      }
    

    /**
     * Find leading pi0 photon energy.
     * @tparam T the type of interaction (true or reco).
     * @param interaction to operate on.
     * @return the leading pi0 photon energy.
     */
    
    template<class T>
      double pi0_leading_photon_energy(const T & interaction)
      {
	vector<size_t> pi0_photon_idxs;
	double energy(0);
	// Truth
	if constexpr (std::is_same_v<T, caf::SRInteractionTruthDLPProxy>)
		       {
			 pi0_photon_idxs = true_pi0_photon_idxs(interaction);
			 size_t i(pi0_photon_idxs[0]);
			 energy = energy_ke(interaction.particles[i]);
		       }
	// Reco
	else
	  {
	    pi0_photon_idxs = pi0_photon_idxs_by_energy(interaction);
	    //pi0_photon_idxs = pi0_photon_idxs_by_dir(interaction);
	    size_t i(pi0_photon_idxs[0]);
	    energy = calo_ke(interaction.particles[i]);
	  }
	return energy;		       
      }
    
    /**
     * Find subleading pi0 photon energy.
     * @tparam T the type of interaction (true or reco).
     * @param interaction to operate on.
     * @return the subleading pi0 photon energy.
     */
    
    template<class T>
      double pi0_subleading_photon_energy(const T & interaction)
      {
	vector<size_t> pi0_photon_idxs;
        double energy(0);
	// Truth
        if constexpr (std::is_same_v<T, caf::SRInteractionTruthDLPProxy>)
                       {
                         pi0_photon_idxs = true_pi0_photon_idxs(interaction);
                         size_t i(pi0_photon_idxs[1]);
                         energy = energy_ke(interaction.particles[i]);
                       }
	// Reco
	else
          {
            pi0_photon_idxs = pi0_photon_idxs_by_energy(interaction);
	    //pi0_photon_idxs = pi0_photon_idxs_by_dir(interaction);
            size_t i(pi0_photon_idxs[1]);
            energy = calo_ke(interaction.particles[i]);
	  }
	return energy;
      }

    /**
     * Variable for finding pi0 leading shower start distance to vertex.
     * @tparam T the type of interaction (true or reco).
     * @param interaction to apply the variable on.
     * @return the leading shower distance to vertex   
     */
    template<class T>
      double pi0_leading_photon_start_to_vertex(const T & interaction)
      {
	
	// Interaction vertex
	TVector3 vertex(interaction.vertex[0], interaction.vertex[1], interaction.vertex[2]);
	
	// Photon info
	vector<size_t> pi0_photon_idxs;
	size_t i;
	TVector3 sh_start;
        double s_to_v(0);

	// Truth
        if constexpr (std::is_same_v<T, caf::SRInteractionTruthDLPProxy>)
                       {
                         pi0_photon_idxs = true_pi0_photon_idxs(interaction);
                         i = pi0_photon_idxs[0];
                       }
	// Reco
	else
	  {
	    pi0_photon_idxs = pi0_photon_idxs_by_energy(interaction);
	    //pi0_photon_idxs = pi0_photon_idxs_by_dir(interaction);
	    i = pi0_photon_idxs[0];
	  }

	sh_start.SetX(interaction.particles[i].start_point[0]);
	sh_start.SetY(interaction.particles[i].start_point[1]);
	sh_start.SetZ(interaction.particles[i].start_point[2]);

	s_to_v = (sh_start - vertex).Mag();
	return s_to_v;
      }

    /**
     * Variable for finding pi0 subleading shower start distance to vertex.
     * @tparam T the type of interaction (true or reco).
     * @param interaction to apply the variable on.
     * @return the leading shower distance to vertex
     */
    template<class T>
      double pi0_subleading_photon_start_to_vertex(const T & interaction)
      {

        // Interaction vertex                                                                                                                    
        TVector3 vertex(interaction.vertex[0], interaction.vertex[1], interaction.vertex[2]);

        // Photon info                                                                                        
        vector<size_t> pi0_photon_idxs;
        size_t i;
        TVector3 sh_start;
        double s_to_v(0);

        // Truth                                                                                                     
        if constexpr (std::is_same_v<T, caf::SRInteractionTruthDLPProxy>)
                       {
                         pi0_photon_idxs = true_pi0_photon_idxs(interaction);
                         i = pi0_photon_idxs[1];
                       }
        // Reco                                                                                                           
        else
          {
	    pi0_photon_idxs = pi0_photon_idxs_by_energy(interaction);
            //pi0_photon_idxs = pi0_photon_idxs_by_dir(interaction);
            i = pi0_photon_idxs[1];
          }

        sh_start.SetX(interaction.particles[i].start_point[0]);
        sh_start.SetY(interaction.particles[i].start_point[1]);
        sh_start.SetZ(interaction.particles[i].start_point[2]);

        s_to_v = (sh_start - vertex).Mag();
        return s_to_v;
      }
      
    /**
     * Find pi0 invariant mass.
     * @tparam T the type of interaction (true or reco).
     * @param interaction to operate on.
     * @return the pi0 invariant mass.
     */
    
    template<class T>
      double pi0_mass(const T & interaction)
      {
	double ph0_energy;
	double ph1_energy;
	double costheta;
	double pi0_mass;

	ph0_energy = pi0_leading_photon_energy(interaction);
	ph1_energy = pi0_subleading_photon_energy(interaction);
	costheta = pi0_costheta(interaction);
	
	pi0_mass = sqrt(2*ph0_energy*ph1_energy*(1-costheta));

	return pi0_mass;
      }

    /**
     * Calculate neutral pion momentum.
     * @tparam T the type of interaction (true or reco).
     * @param interaction to operate on.
     * @return the neutral pion momentum.
     */
    template<class T> TVector3 pi0_momentum(const T & interaction)
      {	
        vector<TVector3> photon_dirs;
	double ph0_energy = pi0_leading_photon_energy(interaction);
	double ph1_energy = pi0_subleading_photon_energy(interaction);
        // Truth
        if constexpr (std::is_same_v<T, caf::SRInteractionTruthDLPProxy>)
                       {
                         photon_dirs = true_pi0_photon_dirs(interaction);
                       }

        // Reco
        else
          {
            photon_dirs = pi0_photon_dirs(interaction);
          }
	
	TVector3 pi0_momentum = (ph0_energy*photon_dirs[0]) + (ph1_energy*photon_dirs[1]);
	return pi0_momentum;
      }


    /**
     * Calculate the magnitude of the neutral pion momentum.
     * @tparam T the type of interaction (true or reco).
     * @param interaction to operate on.
     * @return the magnitude of the neutral pion momentum.
     */
    template <class T> double pi0_momentum_mag(const T & interaction)
      {
	return pi0_momentum(interaction).Mag();
      }

    /**
     * Calculate the neutral pion angle w.r.t. the beam direction.
     * @tparam T the type of interaction (true or reco).
     * @param interaction to operate on.
     * @return the neutral pion angle w.r.t. the beam direction.
     */
    template <class T> double pi0_costheta_z(const T & interaction)
      {
	//TVector3 beamdir(0, 0, 1); // BNB
	TVector3 beamdir(0.39431672, 0.04210058, 0.91800973); // NuMI
	TVector3 pi0_momentum_unit = pi0_momentum(interaction).Unit();
	return pi0_momentum_unit.Dot(beamdir);
      }
    
    /**
     * Calculate the leading muon momentum.
     * @tparam T the type of interaction (true or reco).
     * @param interaction to operate on.
     * @return the leading muon momentum.
     */
    template <class T> TVector3 muon_momentum(const T & interaction)
      {
	TVector3 muon_momentum;
	muon_momentum.SetX(interaction.particles[leading_muon_index(interaction)].momentum[0]);
	muon_momentum.SetY(interaction.particles[leading_muon_index(interaction)].momentum[1]);
	muon_momentum.SetZ(interaction.particles[leading_muon_index(interaction)].momentum[2]);
	return muon_momentum;
      }

    /**
     * Calculate leading muon kinetic energy.
     * @tparam T the type of interaction (true or reco).
     * @param interaction to operate on.
     * @param interaction to operate on.
     * @return the leading muon kinetic energy.
     */
    template <class T> double muon_ke(const T & interaction) 
      {
	return interaction.particles[leading_muon_index(interaction)].ke;
      }

    /**
     * Calculate the transverse momentum of an interaction.
     * @tparam T the type of interaction (true or reco).
     * @param interaction to operate on.
     * @return the transvere momentum of the interaction.
     */
    template <class T> double transverse_momentum(const T & interaction)
      {
	//TVector3 beamdir(0, 0, 1); // BNB
	TVector3 beamdir(0.39431672, 0.04210058, 0.91800973); // NuMI
	
	// Output
	double pT0(0), pT1(0), pT2(0);

	// Loop over particles
	for(auto & part : interaction.particles)
	  {
	    
	    if(!part.is_primary) continue;

	    // pT = p - pL
	    //    = (p dot beamdir) * beamdir
	    TVector3 p;
	    TVector3 pL;
	    TVector3 pT;
	    
	    // Photons
	    if(part.pid == 0)
	      {
		if constexpr (std::is_same_v<T, caf::SRInteractionTruthDLPProxy>)
			       {
				 p.SetX(calo_ke(part)*(part.start_point[0] - interaction.vertex[0]));
				 p.SetY(calo_ke(part)*(part.start_point[1] - interaction.vertex[1]));
				 p.SetZ(calo_ke(part)*(part.start_point[2] - interaction.vertex[2]));
			       }
		else
		  {
		    p.SetX(part.momentum[0]);
		    p.SetY(part.momentum[1]);
		    p.SetZ(part.momentum[2]);
		  }
	      }
	    // All other particles
	    else
	      {
		p.SetX(part.momentum[0]);
		p.SetY(part.momentum[1]);
		p.SetZ(part.momentum[2]);
	      }
	    
	    pL = p.Dot(beamdir) * beamdir;
	    pT = p - pL;
	    pT0 += pT[0];
	    pT1 += pT[1];
	    pT2 += pT[2];

	  } // end particle loop
	
	return sqrt(pow(pT0, 2) + pow(pT1, 2) + pow(pT2, 2));
      }
    
    /**
     * Calculate the magnitude of the leading muon momentum.
     * @tparam T the type of interaction (true or reco).
     * @param interaction to operate on.
     * @return the magnitude of teh leading muon momentum.
     */
    template <class T> double muon_momentum_mag(const T & interaction)
      {
	return muon_momentum(interaction).Mag();
      }

    /**
     * Calculate the leading muon angle w.r.t. the beam direction.
     * @tparam T the type of interaction (true or reco).
     * @param interaction to operate on.
     * @return the leading muon angle w.r.t. the beam.
     */
    template <class T> double muon_costheta_z(const T & interaction)
      {
	//TVector3 beamdir(0, 0, 1); // BNB
	TVector3 beamdir(0.39431672, 0.04210058, 0.91800973); // NuMI
        TVector3 muon_momentum_unit = muon_momentum(interaction).Unit();
        return muon_momentum_unit.Dot(beamdir);
      }
    
}

#endif
