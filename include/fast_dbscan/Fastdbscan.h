#include "KDTreeVectorOfVectorsAdaptor.h"
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/PointIndices.h>

template <typename PointT>
struct PointCloudAdaptor {
    using CloudType = std::vector<PointT, Eigen::aligned_allocator<PointT>>;

    const CloudType& pts;

    PointCloudAdaptor(const CloudType& points) : pts(points) {}

    inline size_t kdtree_get_point_count() const { return pts.size(); }

    inline float kdtree_distance(const float* p1, const size_t idx_p2, size_t /*dim*/) const {
        const auto& p2 = pts[idx_p2];
        float dx = p1[0] - p2.x;
        float dy = p1[1] - p2.y;
        float dz = p1[2] - p2.z;
        return dx * dx + dy * dy + dz * dz;
    }

    inline float kdtree_get_pt(const size_t idx, int dim) const {
        if (dim == 0) return pts[idx].x;
        if (dim == 1) return pts[idx].y;
        return pts[idx].z;
    }

    template <class BBOX>
    bool kdtree_get_bbox(BBOX&) const { return false; }
};

template <typename PointT>
class FastDBSCAN {
public:
    using CloudType = std::vector<PointT, Eigen::aligned_allocator<PointT>>;
    using AdaptorType = PointCloudAdaptor<PointT>;
    using KDTree = nanoflann::KDTreeSingleIndexAdaptor<
        nanoflann::L2_Simple_Adaptor<float, AdaptorType>,
        AdaptorType, 3>;

    void setInputCloud(CloudType& input_cloud) {
        cloud_ = &input_cloud;
    }

    void setCorePointMinPts(int min_pts) {
        core_min_pts_ = min_pts;
    }

    void setClusterTolerance(float eps) {
        eps_ = eps;
    }

    void setMinClusterSize(int min_size) {
        min_cluster_size_ = min_size;
    }

    void setMaxClusterSize(int max_size) {
        max_cluster_size_ = max_size;
    }

    void extract(std::vector<pcl::PointIndices>& cluster_indices) {
        size_t N = cloud_->size();
        std::vector<int> types(N, 0); // 0 = unprocessed, 1 = processing, 2 = processed
        std::vector<bool> is_noise(N, false);

        AdaptorType adaptor(*cloud_);
        KDTree index(3, adaptor, nanoflann::KDTreeSingleIndexAdaptorParams(10));
        index.buildIndex();

        #pragma omp parallel for schedule(dynamic)
        for (int i = 0; i < static_cast<int>(N); ++i) {
            if (types[i] != 0) continue;

            std::vector<std::pair<size_t, float>> matches;
            float query_pt[3] = { (*cloud_)[i].x, (*cloud_)[i].y, (*cloud_)[i].z };
            nanoflann::SearchParams params;
            index.radiusSearch(query_pt, eps_ * eps_, matches, params);

            if (matches.size() < core_min_pts_) {
                is_noise[i] = true;
                continue;
            }

            std::vector<int> seed_queue;
            seed_queue.push_back(i);
            types[i] = 2;

            for (size_t j = 0; j < matches.size(); ++j) {
                size_t idx = matches[j].first;
                if (idx != i && types[idx] == 0) {
                    seed_queue.push_back(idx);
                    types[idx] = 1;
                }
            }

            size_t sq_idx = 1;
            while (sq_idx < seed_queue.size()) {
                int pt_idx = seed_queue[sq_idx];
                float q[3] = { (*cloud_)[pt_idx].x, (*cloud_)[pt_idx].y, (*cloud_)[pt_idx].z };
                std::vector<std::pair<size_t, float>> local_matches;
                index.radiusSearch(q, eps_ * eps_, local_matches, params);

                if (local_matches.size() >= core_min_pts_) {
                    for (const auto& m : local_matches) {
                        int nidx = m.first;
                        if (types[nidx] == 0) {
                            seed_queue.push_back(nidx);
                            types[nidx] = 1;
                        }
                    }
                }

                types[pt_idx] = 2;
                ++sq_idx;
            }

            if (seed_queue.size() >= min_cluster_size_ && seed_queue.size() <= max_cluster_size_) {
                pcl::PointIndices r;
                r.indices = seed_queue;
                std::sort(r.indices.begin(), r.indices.end());
                r.indices.erase(std::unique(r.indices.begin(), r.indices.end()), r.indices.end());
                #pragma omp critical
                cluster_indices.push_back(r);
            }
        }

        std::sort(cluster_indices.rbegin(), cluster_indices.rend(), [](const pcl::PointIndices& a, const pcl::PointIndices& b) {
            return a.indices.size() < b.indices.size();
        });
    }

private:
    CloudType* cloud_;
    int core_min_pts_ = 10;
    float eps_ = 1.0f;
    int min_cluster_size_ = 30;
    int max_cluster_size_ = 10000;
};
