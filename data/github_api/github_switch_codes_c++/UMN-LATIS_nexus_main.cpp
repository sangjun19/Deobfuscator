#include <stdio.h>
#include <iostream>
#include <QStringList>
#include <wrap/system/qgetopt.h>

#include "../nxsbuild/nexusbuilder.h"
#include "../common/traversal.h"
#include "extractor.h"

using namespace std;
using namespace nx;


void printInfo(NexusData &nexus);
void checks(NexusData &nexus);
void recomputeError(NexusData &nexus, QString mode);

bool show_dag = false;
bool show_nodes = false;
bool show_patches = false;
bool show_textures = true;


//TODO REMOVE unused textures when resizing nexus.
int main(int argc, char *argv[]) {

	QString input;
	QString output;
	QString ply;
	float coord_step = 0.0f; //approxismate step for quantization
	int position_bits = 0;
	float error_q = 0.1;
	int luma_bits = 6;
	int chroma_bits = 6;
	int alpha_bits = 5;
	int norm_bits = 10;
	float tex_step = 0.25;

	double error(-1.0);
	double max_size(0.0);
	int max_triangles(0.0);
	QString projection("");
	QString matrix("");
	QString imatrix("");

	bool info = false;
	bool check = false;
	bool compress = false;
	bool drop_level = false;
	QString recompute_error;

	GetOpt opt(argc, argv);
	opt.addArgument("nexus file", "path to the nexus file (add .nxs or not)", &input);
	opt.addSwitch('i', "info", "prints info about the nexus", &info);
	opt.addSwitch('n', "show_nodes", "prints info about nodes", &show_nodes);
	opt.addSwitch('q', "show_patches", "prints info about payches", &show_patches);
	opt.addSwitch('d', "show_dag", "prints info about dag", &show_dag);
	opt.addSwitch('c', "check", "performs various checks", &check);

	//extraction options
	opt.addOption('o', "nexus_file", "filename of the nexus output file", &output);
	opt.addOption('p', "ply_file", "filename of the ply output file", &ply);
	opt.addOption('s', "size", "size in MegaBytes of the final file [requires -E]", &max_size);
	opt.addOption('e', "error", "remove nodes below this error from the node", &error);
	opt.addOption('t', "triangles", "drop nodes until total number of triangles is < triangles [about double final resolution]", &max_triangles);
	opt.addSwitch('l', "last level", "remove nodes from last level", &drop_level);

	//compression and quantization options
	opt.addSwitch('z', "compress", "compress patches", &compress);
	opt.addOption('v', "vertex quantization", "absolute side of compression quantization grid", &coord_step);
	opt.addOption('V', "vertex bits", "number of bits in vertex coordinate when compressing", &position_bits);
	opt.addOption('Y', "luma_bits", "quantization of luma channel (default 6)", &luma_bits);
	opt.addOption('C', "chroma_bits", "quantization of chroma channel (default 6)", &chroma_bits);
	opt.addOption('A', "alha_bits", "quantization of alpha channel (default 5)", &alpha_bits);
	opt.addOption('N', "normal_bits", "quantization of normals (default 10)", &norm_bits);
	opt.addOption('T', "tex_bits", "quantization of textures (default 0.25 pixel)", &tex_step);
	opt.addOption('Q', "quantization_factor", "quantization as a factor of error (default 0.1)", &error_q);


	opt.addOption('E', "recompute_error", "recompute errors [average, quadratic, logarithmic]", &recompute_error);

	//color projection
	opt.addOption('m', "matrix", "multiply by matrix44 in format a:b:c...", &matrix);
	opt.addOption('M', "imatrix", "multiply by inverse of matrix44 in format a:b:c...", &imatrix);
	opt.addOption('P', "project file", "tex taylor project file", &projection);

	opt.parse();

	NexusData nexus;
	bool read_only = true;
	if(!recompute_error.isEmpty())
		read_only = false;

	try {
		if(!nexus.open(input.toLatin1()))
			throw QString("Could not open file: " + input);

		if(info) {
			printInfo(nexus);

			return 0;

		}
		if(check) {
			checks(nexus);
			return 0;
		}

		if(!recompute_error.isEmpty()) {
			recomputeError(nexus, recompute_error);
			return 0;
		}

		if(compress && output.isEmpty()) {
			output = input.left(input.length()-4) + "Z.nxs";
		}
		if(output.isEmpty() && ply.isEmpty()) return 0;
		if(!output.isEmpty() && !ply.isEmpty())  {
			cerr << "The output can be a ply file or a nexus file, not both." << endl;
			return -1;
		}
		if(output == input) {
			cerr << "Output and Input file must be different." << endl;
			return -1;
		}
		Extractor extractor(&nexus);

		if(max_size != 0.0)
			extractor.selectBySize(max_size*(1<<20));

		if(error != -1)
			extractor.selectByError(error);

		if(max_triangles != 0)
			extractor.selectByTriangles(max_triangles);

		if(drop_level)
			extractor.dropLevel();

		if(!ply.isEmpty()) {       //export to ply
			extractor.saveUnifiedPly(ply);

		} else if(!output.isEmpty()) { //export to nexus

			bool invert = false;
			if(!imatrix.isEmpty()) {
				matrix = imatrix;
				invert = true;
			}
			if(!matrix.isEmpty()) {
				QStringList sl = matrix.split(":");
				if(sl.size() != 16) {
					cerr << "Wrong matrix: found only " << sl.size() << " elements.\n";
					exit(-1);
				}
				vcg::Matrix44f m;
				for(int i = 0; i < sl.size(); i++)
					m.V()[i] = sl.at(i).toFloat();
				//if(invert)
				//    m = vcg::Invert(m);

				extractor.setMatrix(m);
			}

			Signature signature = nexus.header.signature;
			if(compress) {
				signature.flags |= Signature::MECO;
				//signature.flags |= Signature::CTM2;

				if(coord_step) {  //global precision, absolute value
					extractor.error_factor = 0.0; //ignore error factor.
					//do nothing
				} else if(position_bits) {
					vcg::Sphere3f &sphere = nexus.header.sphere;
					coord_step = sphere.Radius()/pow(2.0f, position_bits);
					extractor.error_factor = 0.0;

				} else if(error_q) {
					//take node 0:
					int level = 0;
					int node = 0;
					uint32_t sink = nexus.header.n_nodes -1;
					coord_step = error_q*nexus.nodes[0].error/2;
					for(int i = 0; i < sink; i++){
						Node &n = nexus.nodes[i];
						Patch &patch = nexus.patches[n.first_patch];
						if(patch.node != sink)
							continue;
						double e = error_q*n.error/2;
						if(e < coord_step)
							coord_step = e; //we are looking at level1 error, need level0 estimate.
					}
					extractor.error_factor = error_q;
				}
				cout << "Vertex quantization step: " << coord_step << endl;
				cout << "Texture quantization step: " << tex_step << endl;
				extractor.coord_q =(int)log2(coord_step);
				extractor.norm_bits = norm_bits;
				extractor.color_bits[0] = luma_bits;
				extractor.color_bits[1] = chroma_bits;
				extractor.color_bits[2] = chroma_bits;
				extractor.color_bits[3] = alpha_bits;
				extractor.tex_step = tex_step; //was (int)log2(tex_step * pow(2, -12));, moved to per node value
				cout << "tex step: " << extractor.tex_step << endl;
			}

			cout << "saving with flags: " << signature.flags << endl;

			if(!output.endsWith(".nxs") && !output.endsWith(".nxz")) output += ".nxs";

			extractor.save(output, signature);
			//builder.copy(nexus, out, selection);

			/*        builder.create(out, signature);
		if(q_position.toDouble() != 0.0)
			builder.encoder.geometry.options.precision = q_position.toDouble();
		if(position_bits.toInt() != 0)
			builder.encoder.geometry.options.coord_q = position_bits.toInt();
		if(matrix != "") {
			QStringList sl = matrix.toString().split(":");
			if(sl.size() != 16) {
				cerr << "Wrong matrix: found only " << sl.size() << " elements.\n";
				exit(-1);
			}
			vcg::Matrix44f m;
			for(int i = 0; i < sl.size(); i++)
				m.V()[i] = sl.at(i).toFloat();
			builder.setMatrix(m);
		}
		builder.process(nexus, selection); */
		}

		if(projection != "") {
			//call function to actually project color.
			//texProject(NexusBuilder &builder, QString dat_file);
			//for every patch
			//Patch loadPatch(quint32 patch);
			//void unloadPatch(quint32 patch, char *mem); (patch.start)

			//for every image
			//check intersection
			//project

		}

	} catch(QString error) {
		cerr << "ERROR: " << qPrintable(error) << endl;
	}
	return 0;
}

void printInfo(NexusData &nexus) {
	Header &header = nexus.header;
	cout << "Tot vertices: " << header.nvert << endl;
	cout << "Tot faces   : " << header.nface << endl;

	cout << "Components: ";
	if(header.signature.vertex.hasNormals()) cout << " normals";
	if(header.signature.vertex.hasColors()) cout << " colors";
	if(!header.signature.face.hasIndex()) cout << " pointcloud";
	cout << endl;

	cout << "Flags: " << header.signature.flags << endl;
	if(header.signature.flags & Signature::CTM1)
		cout << "Compression; CMT MG1\n";
	vcg::Point3f c = header.sphere.Center();
	float r = header.sphere.Radius();
	cout << "Sphere      : c: [" << c[0] << "," << c[1] << "," << c[2] << "] r: " << r << endl;
	cout << "Nodes       : " << header.n_nodes << endl;
	cout << "Patches     : " << header.n_patches << endl;
	cout << "Textures    : " << header.n_textures << endl << endl;
	cout << endl;


	uint32_t n_nodes = header.n_nodes;
	if(show_dag){
		cout << "Dag dump: \n";
		for(uint i = 0; i < n_nodes-1; i++) {
			nx::Node &node = nexus.nodes[i];
			cout << "Node: " << i << "\t";
			for(uint k = node.first_patch; k < node.last_patch(); k++)
				cout << "[" << nexus.patches[k].node << "] ";
			cout << endl;
		}
		cout << endl;
	}
	int last_level_size =0;
	uint32_t sink = nexus.header.n_nodes -1;
	if(show_nodes) {
		cout << "Node dump:\n";
		double mean = 0;
		for(uint i = 0; i < n_nodes-1; i++) {
			nx::Node &node =  nexus.nodes[i];

			if(nexus.patches[node.first_patch].node == sink)
				last_level_size += node.getEndOffset() - node.getBeginOffset();

			int n = nexus.header.signature.face.hasIndex()?node.nface:node.nvert;
			//compute primitives
			cout << "Node: " << i << "\t  Error: " << node.error << "\t"
				 << " Sphere r: " << node.sphere.Radius() << "\t"
				 << "Primitives: " << n << "\t"
				 << "Size: " << node.getSize() << endl;

			mean += node.nface;
		}
		cout << endl;
		mean /= n_nodes;
		double std = 0;
		for(uint i = 0; i < n_nodes; i++)
			std += pow(mean - nexus.nodes[i].nface, 2);
		std /= n_nodes;
		std = sqrt(std);
		cout << "Mean faces: " << mean << "standard deviation: " << std << endl;
		cout << "Last level size: " << last_level_size/(1024*1024) << "MB" << endl;
	}
	if(show_patches) {
		cout << "Patch dump:\n";
		for(uint i = 0; i < nexus.header.n_patches; i++) {
			nx::Patch &patch = nexus.patches[i];
			cout << "Patch: " << i << "\t Offset: " << patch.triangle_offset << "\t texture: " << patch.texture << endl;

		}
		cout << endl;
	}

	if(show_textures) {
		cout << "Texture dump: \n";
		for(uint i = 0; i < nexus.header.n_textures; i++) {
			nx::Texture &texture = nexus.textures[i];
			cout << "Texture: " << i << "\t Offset: " << texture.getBeginOffset() << " \t size: " << texture.getSize() << endl;
		}
	}


}

void recomputeError(NexusData &nexus, QString error_method) {
	enum Method { AVERAGE, QUADRATIC, LOGARITHMIC, CURVATURE };
	Method method;

	if(error_method == "average") {
		method = AVERAGE;
	} else if(error_method == "quadratic") {
		method = QUADRATIC;
	} else if(error_method == "logarithmic") {
		method = LOGARITHMIC;
	} else if(error_method == "curvature") {
		method = CURVATURE;
	} else {
		cerr << "Not a valid error method: " << qPrintable(error_method) << ". Choose among:\n";
		cerr << "average, quadratic, locaritmic, curvature" << endl;
		exit(0);
	}

	float min_error = 1e20;

	for(uint i = 0; i < nexus.header.n_nodes-1; i++) {
		nx::Node &node = nexus.nodes[i];
		float error = 0;
		int count = 0;
		nexus.loadRam(i);

		NodeData &data = nexus.nodedata[i];
		vcg::Point3f *coords = data.coords();
		uint16_t *faces = data.faces(nexus.header.signature, node.nvert);

		switch(method) {

		case AVERAGE:
			for(int i = 0; i < node.nface; i++) {
				for(int k = 0; k < 3; k++) {
					int v0 = faces[i*3 + k];
					int v1 = faces[i*3 + ((k+1)%3)];
					//this computes average
					float err = (coords[v0] - coords[v1]).SquaredNorm();
					error += sqrt(err);
					count++;
				}
			}
			break;

		case QUADRATIC:
			for(int i = 0; i < node.nface; i++) {
				for(int k = 0; k < 3; k++) {
					int v0 = faces[i*3 + k];
					int v1 = faces[i*3 + ((k+1)%3)];
					error +=(coords[v0] - coords[v1]).SquaredNorm();
					count++;
				}
			}
			break;

		case LOGARITHMIC:
			for(int i = 0; i < node.nface; i++) {
				for(int k = 0; k < 3; k++) {
					int v0 = faces[i*3 + k];
					int v1 = faces[i*3 + ((k+1)%3)];

					float err = (coords[v0] - coords[v1]).SquaredNorm();
					error += log(err); //this is actually 2*error because of the missing square root
					count++;
				}
			}
			break;

		case CURVATURE:
			assert(0);
			break;
		}


		nexus.dropRam(i);
		if(count > 0) {
			switch(method) {
			case AVERAGE: error /= count; break;
			case QUADRATIC: error = sqrt(error/count); break;
			case LOGARITHMIC: error = exp(error/(2*count)); //2 accounts for the square roots
			case CURVATURE: break;
			}
		}
		if(error > 0 && error < min_error)
			min_error = error;
		node.error = error;
	}

	nexus.nodes[nexus.header.n_nodes - 1].error = min_error;

	nexus.file.seek(sizeof(Header));
	nexus.file.write((char *)nexus.nodes, sizeof(nx::Node)* nexus.header.n_nodes);
	//fseek(nexus.file, sizeof(Header), SEEK_SET);
	//fwrite(nexus.nodes, sizeof(nx::Node), nexus.header.n_nodes, nexus.file);
}

void checks(NexusData &nexus) {
	uint32_t n_nodes = nexus.header.n_nodes;
	uint32_t n_patches = nexus.header.n_patches;
	uint32_t n_textures = nexus.header.n_textures;

	for(uint i = 0; i < n_nodes-1; i++) {
		Node &node = nexus.nodes[i];
		if(node.first_patch >= n_patches) {
			cout << "Node: " << i << " of " << n_nodes << " first patch: " << node.first_patch << " of " << n_patches << endl;
			exit(-1);
		}
		assert(node.offset * (quint64)NEXUS_PADDING < nexus.file.size());
	}
	assert(nexus.nodes[n_nodes-1].first_patch = n_patches);

	for(uint i = 0; i < n_patches-1; i++) {
		Patch &patch = nexus.patches[i];
		assert(patch.node < n_nodes);
		if(patch.texture == 0xffffffff) continue;
		if(patch.texture >= n_textures) {
			cout << "Patch: " << i << " of: " << n_patches << " texture: " << patch.texture << " of: " << n_textures << endl;
			exit(-1);
		}
	}
	if(n_textures)
		assert(nexus.patches[n_patches-1].texture = n_textures);

	for(uint i = 0; i < n_textures-1; i++) {
		Texture &texture = nexus.textures[i];
		if(texture.offset * (quint64)NEXUS_PADDING >= nexus.file.size()) {
			cout << "Texture " << i << " offset: " << texture.offset*NEXUS_PADDING << " file size: " << nexus.file.size() << endl;
		}
	}
	if(nexus.textures[n_textures-1].offset*NEXUS_PADDING != nexus.file.size()) {
		cout << "last texture: " << nexus.textures[n_textures-1].offset*NEXUS_PADDING <<
				"file size: " << nexus.file.size() << endl;
	}
	return;
}

//check bounding spheres
void checkSpheres(NexusData &nexus) {

	uint32_t n_nodes = nexus.header.n_nodes;
	for(int i = n_nodes -2; i >= 0; i--) {
		Node &node = nexus.nodes[i];
		vcg::Sphere3f sphere = node.tightSphere();
		nexus.loadRam(i);
		NodeData &data = nexus.nodedata[i];

		vcg::Point3f *coords = data.coords();
		float new_radius = sphere.Radius();
		for(int k = 0; k < node.nvert; k++) {
			vcg::Point3f &p = coords[k];
			float dist = (p - sphere.Center()).Norm();
			if(dist > new_radius)
				new_radius = dist;
		}
		if(node.tight_radius != new_radius) {
			if(node.tight_radius*2 < new_radius) {
				cout << "OFfset: " << node.offset*NEXUS_PADDING << endl;
			}
			cout << "Node: " << i << " radius: " << node.tight_radius << " -> " << new_radius << endl;
		}
		node.tight_radius = new_radius;
		nexus.dropRam(i);
	}

	//write back nodes.
	nexus.file.seek(sizeof(Header));
	nexus.file.write((char *)nexus.nodes, sizeof(nx::Node)* nexus.header.n_nodes);
	//fseek(nexus.file, sizeof(Header), SEEK_SET);
	//fwrite(nexus.nodes, sizeof(Node), n_nodes, nexus.file);
}
