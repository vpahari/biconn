/*
 * Volume.h
 *
 * Created on: July 7, 2018
 * Author: Franz-Benjamin Mocnik <mail@mocnik-science.net>
 */

#ifndef VOLUME_H_
#define VOLUME_H_

#include "../graph/Graph.h"
#include <unordered_map>

namespace NetworKit {

/**
 * @ingroup volume
 */
class Volume {
private:
	/**
	 * For a given node n, the nodes within distance r are returned together with
	 * their distance to node n.
	 */
	static std::unordered_map<node, double> nodesWithinDistance(const Graph &G, double r, node n);

public:
	/**
	* Number of nodes within a given radius; average for many nodes
	*
	* Please find further information about the volume and its meaning in the
	* following publication:
	*
	* Franz-Benjamin Mocnik: "The Polynomial Volume Law of Complex Networks in
	* the Context of Local and Global Optimization", Scientific Reports 8(11274)
	* 2018. doi: 10.1038/s41598-018-29131-0
	*
	* @param G  the graph
	* @param r  the radius
	* @param samples	the number of samples
	*
	**/
	static double volume(const Graph &G, const double r, const count samples);

	/**
	* Number of nodes within different given radii; average for many nodes
	*
	* Please find further information about the volume and its meaning in the
	* following publication:
	*
	* Franz-Benjamin Mocnik: "The Polynomial Volume Law of Complex Networks in
	* the Context of Local and Global Optimization", Scientific Reports 8(11274)
	* 2018. doi: 10.1038/s41598-018-29131-0
	*
	* @param G  the graph
	* @param rs  the radii
	* @param samples	the number of samples
	*
	**/
	static std::vector<double> volume(const Graph &G, const std::vector<double> rs, const count samples);
};

} /* namespace NetworKit */

#endif /* VOLUME_H_ */
