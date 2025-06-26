#include "efear/voxelgrid.h"


//namespace radar_mapping {

int GetX(idx_grid& idx){return std::get<0>(idx);}

int GetY(idx_grid& idx){return std::get<1>(idx);}

int GetZ(idx_grid& idx){return std::get<2>(idx);}


idx_grid Voxelgrid::GetIndex(const pcl::PointXYZI &p){
  idx_grid k{floor(p.x/resolution_+0.5), floor(p.y/resolution_+0.5), floor(p.z/resolution_+0.5)}; // middle voxel centered aroubnd (0,0,0) with voxel border at 0.5*res in each direction
  return k;
}

idx_grid Voxelgrid::GetIndex(const double x, const double y, const double z){
  idx_grid k{floor(x/resolution_+0.5), floor(y/resolution_+0.5), floor(z/resolution_+0.5)}; // middle voxel centered aroubnd (0,0,0) with voxel border at 0.5*res in each direction
  return k;
}

void Voxelgrid::IdxToCenter(const idx_grid& idx, Eigen::Vector3d& center){
  center(0) = resolution_*std::get<0>(idx);
  center(1) = resolution_*std::get<1>(idx);
  center(2) = resolution_*std::get<2>(idx);
}

Eigen::Vector3d Voxelgrid::IdxToCenterReturn(const idx_grid& idx){
  Eigen::Vector3d center;
  center(0) = resolution_*std::get<0>(idx);
  center(1) = resolution_*std::get<1>(idx);
  center(2) = resolution_*std::get<2>(idx);
  return center;
}

void Voxelgrid::InsertElement(const double x, const double y, const double z, const voxel& obj){
  InsertElement(GetIndex(x,y,z), obj);
}
void Voxelgrid::InsertElement(idx_grid idx, const voxel& obj){
  map_[idx] = obj;
}

voxel* Voxelgrid::GetElement(const idx_grid& idx){
  voxel* obj;
  auto itr = map_.find(idx);
  if(itr != map_.end()){
    obj = &((*itr).second);
  }
  return obj;
}

std::vector<idx_grid> Voxelgrid::GetNeighbours( idx_grid& idx, int n){
  std::vector<idx_grid> neighbours;
  for(int x=GetX(idx)-n ; x<=GetX(idx)+n ; x++)
    for(int y=GetY(idx)-n ; y<=GetY(idx)+n ; y++)
      for(int z=GetZ(idx)-n ; z<=GetZ(idx)+n ; z++)
      neighbours.push_back(std::make_tuple(x,y,z));

  return neighbours;
}

const voxel* Voxelgrid::GetElement(const double x, const double y, const double z){
  return GetElement(GetIndex(x, y, z));
}


bool Voxelgrid::ElementExists(const idx_grid& idx){
  if(map_.find(idx) != map_.end()){
    return true;
  }
  else return false;
}

size_t Voxelgrid::Size(){
  return map_.size();
}

std::vector<idx_grid> Voxelgrid::GetIndecies(){
  std::vector<idx_grid>  full_index(map_.size());
  for(auto it = map_.begin(); it != map_.end(); it++)
    full_index[std::distance(map_.begin(),it)] = it->first;
  return full_index;
}
std::vector<Eigen::Vector3d> Voxelgrid::GetCenters(){
  std::vector<Eigen::Vector3d> centers(map_.size());
  for(auto it = map_.begin(); it != map_.end(); it++)
    centers[std::distance(map_.begin(),it)] = IdxToCenterReturn(it->first);

  return centers;
}
std::vector<voxel> Voxelgrid::GetCells(){
  std::vector<voxel> cells(map_.size());
  for(auto it = map_.begin(); it != map_.end(); it++)
    cells[std::distance(map_.begin(),it)] = it->second;

  return cells;
}

void Voxelgrid::Clear(){
  map_.clear();
}

double Voxelgrid::GetResolution(){
  return resolution_;
}


/********* VOXEL ***********/

voxel::voxel():N_(0),has_gausian_(false){}

voxel::voxel(const Eigen::Vector3d& p):voxel(){
  AddPoint(p);
}

void voxel::AddPoint(const Eigen::Vector3d& p){
  pnts_.resize(pnts_.rows()+1,3);
  pnts_.block<3,1>(pnts_.rows()-1,0) = p ;
  cout<<"pnts: "<<pnts_<<endl;
}
void voxel::ComputeMeanCov(const Eigen::Vector3d& p){
  const int nr_pnts = pnts_.rows();
  if(nr_pnts>=3){
    mean_ = pnts_.colwise().mean(); 

  }
}



/********* NDT GRID ***********/


NDTGrid::NDTGrid(float resolution):map_(resolution){}

NDTGrid::NDTGrid(const pcl::PointCloud<pcl::PointXYZI>& cld, float resolution) : NDTGrid(resolution){
  for (const auto& p : cld) {
    Eigen::Vector3d v;
    v << p.x, p.y, p.z;
    idx_grid idx = map_.GetIndex(p);
    if(map_.ElementExists(idx))
      map_.GetElement(idx)->AddPoint(v);
    else
      map_.InsertElement(idx, voxel(v));
  }
}


//}
