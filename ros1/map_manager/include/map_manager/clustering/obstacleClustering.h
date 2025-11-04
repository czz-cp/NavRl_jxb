/*
	FILE: obstacleClustering.h
	------------------------
	function header of static obstacle clustering
*/
#ifndef OBSTALCE_CLUSTERING_H
#define OBSTALCE_CLUSTERING_H

#include <map_manager/clustering/DBSCAN.h>
#include <map_manager/clustering/Kmeans.h>
#include <thread>
#include <Eigen/Eigen>
#include <math.h>

// 点云聚类
// 表示一个障碍物点云聚类及其基本几何信息
struct pointCluster{
	std::vector<Eigen::Vector3d> points; // 聚类中的点
	int id;					 // 聚类ID	
	Eigen::Vector3d centroid; // 聚类中心
	Eigen::Vector3d clusterMin, clusterMax; // 聚类边界框最小最大点

	pointCluster(){}

	pointCluster(const std::vector<Eigen::Vector3d>& _points, int _id, const Eigen::Vector3d& _centroid, 
		         const Eigen::Vector3d& _clusterMin, const Eigen::Vector3d& _clusterMax){
		points = _points;
		id = _id;
		centroid = _centroid;
		clusterMin = _clusterMin;
		clusterMax = _clusterMax;
	}
};

// 表示一个旋转的3D边界框，包含所有顶点和几何属性
struct bboxVertex{
	std::vector<Eigen::Vector3d> vert;
	Eigen::Vector3d centroid;
	Eigen::Vector3d dimension;
	double angle;
	bboxVertex(){}

	bboxVertex(const Eigen::Vector3d& pmin, const Eigen::Vector3d& pmax, double rotAngle){
		vert.push_back(Eigen::Vector3d (pmax(0), pmax(1), pmax(2)));
		vert.push_back(Eigen::Vector3d (pmin(0), pmax(1), pmax(2)));
		vert.push_back(Eigen::Vector3d (pmin(0), pmin(1), pmax(2)));
		vert.push_back(Eigen::Vector3d (pmax(0), pmin(1), pmax(2)));
		vert.push_back(Eigen::Vector3d (pmax(0), pmax(1), pmin(2)));
		vert.push_back(Eigen::Vector3d (pmin(0), pmax(1), pmin(2)));
		vert.push_back(Eigen::Vector3d (pmin(0), pmin(1), pmin(2)));
		vert.push_back(Eigen::Vector3d (pmax(0), pmin(1), pmin(2)));

		dimension = pmax - pmin;
		centroid = (pmax + pmin)/2.0;
		angle = rotAngle;
	}
};


// 静态障碍物信息结构体
struct staticObstacle{
	Eigen::Vector3d centroid;
	Eigen::Vector3d size;
	double yaw;
};

// 聚类算法：
// 第一级：DBSCAN聚类，得到初始聚类 将相邻的点分组为初步聚类
// 第二级：Kmeans聚类+边界框优化，得到最终聚类结果 对大的聚类进一步细分
class obstacleClustering{
	private:
		std::shared_ptr<DBSCAN> dbscan_;
		std::shared_ptr<KMeans> kmeans_;
		std::vector<pointCluster> initialClusters_;
		std::vector<bboxVertex> refinedRotatedBBoxes_;
		std::vector<bboxVertex> rotatedInitialBBoxes_;
		double res_;

		// DBSCAN paremeters
		double eps_ = 0.5;
		double minPts_ = 15;

		// Kmeans & Refinement parameters
		int treeLevel_ = 3;
		int angleDiscreteNum_ = 20;
		double densityThresh_ = 0.9;
		double improveThresh_ = 1.1;
		int kmeansIterNum_ = 10;


	public:
		obstacleClustering(double res);
		void setParam();
		// 输入局部点云，执行完整的聚类流程
		void run(const std::vector<Eigen::Vector3d>& localCloud);

		void runDBSCAN(const std::vector<Eigen::Vector3d>& localCloud, std::vector<pointCluster>& cloudClusters);
		void runKmeans(const pointCluster& cluster, std::vector<pointCluster>& cloudClusters);
		double getOrientation(const pointCluster& cluster, bboxVertex& vertex);

		// 结果获取函数
		std::vector<pointCluster> getInitialCluster();    // 初始聚类
		std::vector<bboxVertex> getRotatedInitialBBoxes(); // 初始边界框
		std::vector<bboxVertex> getRefinedBBoxes();        // 优化后的边界框
		std::vector<staticObstacle> getStaticObstacles();   // 简化障碍物表示
};


#endif